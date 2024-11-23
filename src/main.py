#
# main.py - Основной скрипт. Просто тестируем ии.
#


# Импортируем:
import os
import json
import torch

# Импортируем функционал из других скриптов:
from gpt import GPTLLM
from tokenizer import TokenizerSymbol, TokenizerWord
from train import TrainGPTLLM


""" Структуры конфигураторов:
    # Гиперпараметры модели:
    hparams = {
        "n_vocab": 120,  # Размер словаря. Сколько токенов знает модель и их значение.
        "n_ctx":   256,  # Размер контекста который модель может учитывать при генерации след. токена.
        "n_embd":  384,  # Размер вектора для токена. Сколько чисел определяют токен в пространстве.
        "n_head":  6,    # Количество голов внимания. Определяет количество точек внимания в контексте.
        "n_layer": 6,    # Количество слоёв трансформеров. Определяет обширность понимания контекста.
        "dropout": 0.15  # Вероятность "выключения" нейронов во время обучения (необязательный параметр).
    }

    # Конфиг тренировки модели:
    train = {
        # Прочие параметры:
        "batch_size": 64,    # Размер пакета данных, используемый для обучения в одной итерации.
        "learn_iter": 5000,  # Количество итераций для обучения.
        "learn_rate": 3e-4,  # Определяет, насколько сильно обновляются веса в процессе обучения.
        "eval_inter": 500,   # Через сколько итераций обучения, модель будет оцениваться на валидационных данных.
        "eval_iters": 100    # Сколько итераций будет использоваться для оценки ошибки во время валидации.
    }
"""


# Загрузить параметры модели:
def load_parameters(dir_path: str) -> [dict, dict, dict]:
    vocab = None
    with open(os.path.join(dir_path, "vocab.json"), "r+", encoding="utf-8") as f:
        vocab = json.load(f)

    hparams = None
    with open(os.path.join(dir_path, "hparams.json"), "r+", encoding="utf-8") as f:
        hparams = json.load(f)

    train = None
    if os.path.isfile(os.path.join(dir_path, "train.json")):
        with open(os.path.join(dir_path, "train.json"), "r+", encoding="utf-8") as f:
            train = json.load(f)

    return vocab, hparams, train


# Сохранить параметры модели:
def save_parameters(dir_path: str, vocab: dict, hparams: dict, train_cfg: dict, indent: int = 4) -> None:
    with open(os.path.join(dir_path, "vocab.json"), "w+", encoding="utf-8") as f:
        json.dump(vocab, f, indent=indent)

    with open(os.path.join(dir_path, "hparams.json"), "w+", encoding="utf-8") as f:
        json.dump(hparams, f, indent=indent)

    with open(os.path.join(dir_path, "train.json"), "w+", encoding="utf-8") as f:
        json.dump(train_cfg, f, indent=indent)


# Путь до папки с моделью:
model_path = "data/models/shakespeare/"


# Загружаем модель:
vocab, hparams, train_cfg = load_parameters(model_path)


# Загружаем текст для тренировки нейросети:
train_text = ""
with open("./data/texts/shakespeare.txt", "r+", encoding="utf-8") as f: train_text = f.read()

# Создаём токенизатор:
tokenizer = TokenizerSymbol(text=train_text)
# tokenizer = TokenizerWord(text=train_text)  # Токенизатор по словам.
print(f"Vocab size: {tokenizer.vocab_size} tokens.")

""" Изменяем параметры только если хотим с нуля обучить нейросеть:
vocab = tokenizer.vocab

hparams = {
    "n_vocab": tokenizer.vocab_size,
    "n_ctx": 256,
    "n_embd": 384,
    "n_head": 6,
    "n_layer": 6,
    "dropout": 0.15
}

train_cfg = {
    "batch_size": 64,
    "learn_iter": 5000,
    "learn_rate": 3e-4,
    "eval_inter": 500,
    "eval_iters": 100
}
"""

# Создаём нейросеть:
gpt = GPTLLM(hparams, "cuda")
print(f"Device: {gpt.device}")

# Выводим размер модели:
print(sum(p.numel() for p in gpt.parameters())/1_000_000, "M parameters.")

"""
# Подготавливаем текст для тренировки нейросети:
data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long).to(gpt.device)
n = int(0.9*len(data))  # Первые 90% - это тренировка, остальные 10% - проверочные.
train_data = data[:n]
valid_data = data[n:]

# Тренеруем нейросеть:
trainer = TrainGPTLLM(train_cfg, gpt, train_data, valid_data)
trainer.train(model_path, 100, 3.0)

# Сохраняем натренированную модель:
save_parameters(model_path, vocab, hparams, train_cfg)
gpt.save(os.path.join(model_path, "model.bin"))
"""


gpt.load(os.path.join(model_path, "model.bin"))  # Загружаем веса.


# Функция генерации ответа от ии:
request = lambda txt, mxt, tmp, tpk: tokenizer.decode(gpt.generate(tokenizer.encode(txt), mxt, tmp, tpk))

text = """
ОНИ:
Это очень интересно!
"""
print(f"Prompt: {text}\n\nAI: {request(text, 1000, 0.4, 0)}")
