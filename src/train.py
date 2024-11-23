#
# train.py - Создаёт код для обучения нейросети на наборе текста.
#


# Импортируем:
import os
import time
import json
import shutil
import torch
import torch.nn as nn
from tqdm import tqdm


# Сам класс для обучения нейросети:
class TrainGPTLLM:
    def __init__(self,
                 config:       dict,
                 model:        nn.Module,
                 train_data:   torch.tensor,
                 valid_data:   torch.tensor,
                 weight_decay: float = 1e-4) -> None:
        self.batch_size = config["batch_size"]
        self.learn_iter = config["learn_iter"]
        self.learn_rate = config["learn_rate"]
        self.eval_inter = config["eval_inter"]
        self.eval_iters = config["eval_iters"]
        self.model      = model
        self.optimizer  = torch.optim.AdamW(self.model.parameters(), lr=self.learn_rate, weight_decay=weight_decay)
        self.train_data = train_data
        self.valid_data = valid_data
        self.n_ctx      = model.hparams["n_ctx"]

    # Генерирует небольшой пакет данных с входами X и целями Y:
    def get_batch(self, split: str) -> (torch.Tensor, torch.Tensor):
        data = self.train_data if split == "train" else self.valid_data  # Определяем, какие данные использовать.
        ix   = torch.randint(len(data)-self.n_ctx, (self.batch_size,))   # Генерируем случайные индексы для выборки.

        # Формируем входные данные X из выбранных индексов:
        x = torch.stack([data[i:i+self.n_ctx] for i in ix])  # (batch_size, n_ctx).

        # Формируем целевые данные Y, сдвинутые на 1 по времени:
        y = torch.stack([data[i+1:i+self.n_ctx+1] for i in ix])  # (batch_size, n_ctx).

        # Переносим данные на указанное устройство (CPU или GPU), и возвращаем результат:
        x, y = x.to(self.model.device), y.to(self.model.device)
        return x, y

    # Оценка потерь для обучающей и валидационной выборок:
    @torch.no_grad()
    def estimate_loss(self) -> dict:
        out = {}           # Инициализация словаря для хранения результатов.
        self.model.eval()  # Переключаем модель в режим оценки (отключение Dropout и BatchNorm).
        for split in ["train", "valid"]:           # Проходим по обучающей и валидационной выборкам.
            losses = torch.zeros(self.eval_iters)  # Инициализация тензора для хранения потерь.
            for k in range(self.eval_iters):       # Цикл для оценки потерь.
                x, y = self.get_batch(split)       # Получаем пакет данных (входы и цели).
                logits, loss = self.model(x, y)    # Пропускаем данные через модель для получения логитов и потерь.
                losses[k] = loss.item()            # Сохраняем значение потерь в тензор.
            out[split] = losses.mean()             # Сохраняем усредненные потери для данной выборки.
        self.model.train()  # Переключаем модель обратно в режим обучения.
        return out          # Возвращаем словарь с потерями.

    # Загрузить оптимизатор:
    def load(self, file_path: str, weights_only: bool = True) -> None:
        self.optimizer.load_state_dict(torch.load(file_path, weights_only=True))

    # Сохранить оптимизатор:
    def save(self, file_path: str) -> None:
        torch.save(self.optimizer.state_dict(), file_path)

    # Обучить нейросеть:
    def train(self,
              autosave_dir:      str   = None,
              autosave_per_iter: int   = 100,
              diff_threshold:    float = 1.0) -> None:
        # Автосохранение модели и параметров:
        def autosave(dir_path: str, current_iter: int, total_iters: int) -> None:
            if not os.path.exists(dir_path): os.makedirs(dir_path)  # Если папки не существует, создаём её.

            # Сохраняем гиперпараметры:
            with open(os.path.join(dir_path, "hparams.json"), "w+", encoding="utf-8") as f:
                json.dump(self.model.hparams, f, indent=4)

            # Сохраняем веса модели:
            if os.path.isfile(os.path.join(autosave_dir, "model.bin")):
                shutil.copy2(os.path.join(dir_path, "model.bin"), os.path.join(dir_path, "model-backup.bin"))
            self.model.save(os.path.join(dir_path, "model.bin"))

            # Сохраняем состояние оптимизатора:
            if os.path.isfile(os.path.join(autosave_dir, "optimizer.bin")):
                shutil.copy2(os.path.join(dir_path, "optimizer.bin"), os.path.join(dir_path, "optimizer-backup.bin"))
            self.save(os.path.join(dir_path, "optimizer.bin"))

            # Сохраняем процесс обучения:
            with open(os.path.join(dir_path, "train-autosave.json"), "w+", encoding="utf-8") as f:
                json.dump({
                    "last_iter":      current_iter,             # Последняя итерация.
                    "total_iters":    total_iters,              # Всего итераций на момент обучения.
                    "remained_iters": total_iters-current_iter  # Оставшиеся итерации.
                }, f, indent=4)

        # Загружаем автосохранение:
        start_iter = 0
        if autosave_dir is not None:
            autosave_file = os.path.join(autosave_dir, "train-autosave.json")
            if os.path.isfile(autosave_file):
                with open(autosave_file, "r", encoding="utf-8") as f: train_state = json.load(f)
                start_iter = train_state["last_iter"]
                if start_iter > 0: start_iter += 1

            if start_iter > 0:
                model_file = os.path.join(autosave_dir, "model.bin")
                if os.path.isfile(model_file):
                    self.model.load(model_file)

                optimizer_file = os.path.join(autosave_dir, "optimizer.bin")
                if os.path.isfile(optimizer_file):
                    self.load(optimizer_file)

        # Проходимся epoch-количество раз обучая нейросеть:
        diff = 0.0  # Разница в потерях между обучаемой и проверочных данныхы.
        for i in tqdm(range(start_iter, self.learn_iter), "Training", total=self.learn_iter, initial=start_iter):
            # Загружаем батч данных:
            xb, yb = self.get_batch("train")

            # Вычисляем потери:
            logits, loss = self.model(xb, yb)

            # Время от времени оценивайте потери на наборах train_data и valid_data:
            if i % self.eval_inter == 0 or i == self.learn_iter-1:
                train, valid = self.estimate_loss().values()
                diff = valid - train
                print(f"\nLosses [Train|Valid]: T={train:.4f} V={valid:.4f}. Difference: {diff:.4f}\n")

            # Прерываем, если достигаем заданного значения потерь:
            if diff >= diff_threshold:
                print(f"\nDifference threshold reached ({diff} >= {diff_threshold}). Stopping training.\n")
                break

            # Оптимизация (обучение на ошибках):
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # Автосохранение модели:
            if (i % autosave_per_iter == 0 or i == self.learn_iter-1) and i > 0 and autosave_dir is not None:
                print(f"\nAutosaving training and model, iteration {i}/{self.learn_iter}...")
                autosave(autosave_dir, i, self.learn_iter)

        # Удаляем автосохранение обучения и оптимизатора:
        if autosave_dir is not None:
            if os.path.isfile(os.path.join(autosave_dir, "train-autosave.json")):
                os.remove(os.path.join(autosave_dir, "train-autosave.json"))

            if os.path.isfile(os.path.join(autosave_dir, "model-backup.bin")):
                os.remove(os.path.join(autosave_dir, "model-backup.bin"))

            if os.path.isfile(os.path.join(autosave_dir, "optimizer.bin")):
                os.remove(os.path.join(autosave_dir, "optimizer.bin"))

            if os.path.isfile(os.path.join(autosave_dir, "optimizer-backup.bin")):
                os.remove(os.path.join(autosave_dir, "optimizer-backup.bin"))
