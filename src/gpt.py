#
# gpt.py - Основной код нейросети.
#


# Импортируем:
import random
import torch
import torch.nn as nn
from torch.nn import functional as TF
from tqdm import tqdm


# Устанавливаем seed для случайных чисел:
seed = random.randint(0, +1_000_000_000)
print(f"Initial seed: {seed}")
torch.manual_seed(seed)


# Реализует одну "голову" внимания:
class Head(nn.Module):
    def __init__(self, head_size: int, hparams: dict) -> None:
        super().__init__()

        # Линейные слои для преобразования входных данных в ключи, запросы и значения:
        self.key   = nn.Linear(hparams["n_embd"], head_size, bias=False)  # (C -> head_size).
        self.query = nn.Linear(hparams["n_embd"], head_size, bias=False)  # (C -> head_size).
        self.value = nn.Linear(hparams["n_embd"], head_size, bias=False)  # (C -> head_size).

        # Нижнетреугольная матрица для маскирования при внимании: запрещает внимание на будущие токены:
        self.register_buffer("tril", torch.tril(torch.ones(hparams["n_ctx"], hparams["n_ctx"])))

        # Dropout для регулирования обучения и уменьшения переобучения:
        self.dropout = nn.Dropout(hparams["dropout"] if "dropout" in hparams else 0.0)

    # Выполняет механизм само-внимания для данной входной последовательности:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # Преобразуем входные данные в ключи (key), запросы (query) и значения (value):
        k = self.key(x)    # Размерность (B, T, head_size).
        q = self.query(x)  # Размерность (B, T, head_size).

        # Вычисляем скоринг внимания, чтобы определить, насколько каждый токен связан с другими токенами:
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, T).

        # Используем нижнетреугольную маску, чтобы запретить внимание на будущие токены:
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T).

        # Применяем softmax, чтобы получить вероятности внимания:
        wei = TF.softmax(wei, dim=-1)  # (B, T, T).

        # Применяем dropout к вероятностям внимания
        wei = self.dropout(wei)
        
        # Преобразуем входные данные в значения (value):
        v = self.value(x)  # (B, T, head_size).
        
        # Агрегируем значения с учетом вероятностей внимания:
        out = wei @ v  # (B, T, head_size).

        return out


# Механизм многошагового внимания, который комбинирует результаты нескольких "голов" внимания:
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size: int, hparams: dict) -> None:
        super().__init__()

        # Cписок голов внимания:
        self.heads = nn.ModuleList([Head(head_size, hparams) for _ in range(hparams["n_head"])])

        # Линейный слой для проекции выходов голов:
        self.proj = nn.Linear(head_size * hparams["n_head"], hparams["n_embd"])

        # Слой регуляризации для предотвращения переобучения:
        self.dropout = nn.Dropout(hparams["dropout"] if "dropout" in hparams else 0.0)

    # Выполняет многошаговое внимание на входной последовательности:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# Простая полносвязная нейронная сеть с нелинейностью для обработки эмбеддингов:
class FeedFoward(nn.Module):
    def __init__(self, hparams: dict) -> None:
        super().__init__()
        n_embd, dropout = hparams["n_embd"], hparams["dropout"] if "dropout" in hparams else 0.0
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Линейный слой, который увеличивает размерность в 4 раза.
            nn.GELU(),                      # Функция активации (ReLU, GELU).
            nn.Linear(4 * n_embd, n_embd),  # Линейный слой для снижения размерности обратно до n_embd.
            nn.Dropout(dropout),            # Слой дропаут для регуляризации и предотвращения переобучения.
        )

    # Пропускает входные данные через полносвязную сеть:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # Применяет последовательные слои к входным данным.


# Блок трансформера, который включает в себя слой внимания и слой feed-forward:
class Block(nn.Module):
    def __init__(self, hparams: dict) -> None:
        super().__init__()
        head_size = hparams["n_embd"] // hparams["n_head"]  # Размер каждой головы внимания.
        self.sa   = MultiHeadAttention(head_size, hparams)  # Слой многоголового внимания.
        self.ffwd = FeedFoward(hparams)                     # Слой feed-forward.
        self.ln1  = nn.LayerNorm(hparams["n_embd"])         # Слой нормализации для первого внимания.
        self.ln2  = nn.LayerNorm(hparams["n_embd"])         # Слой нормализации для feed-forward.

    # Пропускает входные данные через блок трансформера:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))  # Применяет слой нормализации, затем многоголовое внимание и добавл. остат. связь.
        x = x + self.ffwd(self.ln2(x))  # Применяет слой нормализации, затем feed-forward и добавляет остаточную связь.
        return x  # Возвращает обработанные данные.


# Основной класс нейросети (генеративная предобученная трансформерная модель - GPT):
class GPTLLM(nn.Module):
    def __init__(self, hparams: dict, device: str = None) -> None:
        super().__init__()

        n_vocab, n_ctx, n_embd, n_head, n_layer = list(hparams.values())[:5]
        self.token_embedding_table    = nn.Embedding(n_vocab, n_embd)            # Эмбеддинг токенов.
        self.position_embedding_table = nn.Embedding(n_ctx, n_embd)              # Эмбеддинг позиций.
        self.blocks  = nn.Sequential(*[Block(hparams) for _ in range(n_layer)])  # Последовательность блоков.
        self.lm_head = nn.Linear(n_embd, n_vocab)  # Линейный слой для предсказания токенов.
        self.ln_f    = nn.LayerNorm(n_embd)        # Нормализация на выходе.

        # Генерируемые токены:
        self.gen_tokens = ""

        # Инициализация весов:
        self.apply(self._init_weights_)

        # Гиперпараметры:
        self.hparams = hparams

        # Вычислительное устройство:
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)  # Переносим эту модель на устройство.

    # Инициализация весов для линейных слоев и эмбеддингов нормальным распределением:
    def _init_weights_(self, module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Прямое распространение данных через модель:
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple:
        B, T = idx.shape

        # idx и targets оба являются тензорами (B, T) целых чисел:
        tok_emb = self.token_embedding_table(idx)                                     # (B, T, C).
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T, C).
        x = tok_emb + pos_emb     # (B, T, C).
        x = self.blocks(x)        # (B, T, C).
        x = self.ln_f(x)          # (B, T, C).
        logits = self.lm_head(x)  # (B, T, vocab_size).

        if targets is None: loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)            # Приводим логи в нужный формат для потерь.
            targets = targets.view(B * T)             # Приводим targets в нужный формат.
            loss = TF.cross_entropy(logits, targets)  # Вычисляем потери.

        return logits, loss

    # Загрузить модель:
    def load(self, file_path: str, weights_only: bool = True, strict: bool = True) -> None:
        self.load_state_dict(torch.load(file_path, map_location=self.device, weights_only=weights_only), strict=strict)
        self.to(self.device)

    # Сохранить модель:
    def save(self, file_path: str) -> None:
        torch.save(self.state_dict(), file_path)

    # Генерация текста на основе входного промпта:
    @torch.no_grad()
    def generate(self, prompt, max_tokens: int, temp: float = 1.0, top_k: int = 0, tqdm_use: bool = True) -> list:
        # Преобразуем промпт в тензор:
        idx = torch.tensor(prompt, dtype=torch.long, device=self.device).unsqueeze(0)

        self.gen_tokens = []  # Инициализируем список для сгенерированных токенов.

        # Цикл генерации токенов:
        for _ in tqdm(range(max_tokens), "Generating") if tqdm_use else range(max_tokens):
            # Обрезаем последовательность, если она превышает n_ctx:
            idx_cond = idx if idx.size(1) <= self.hparams["n_ctx"] else idx[:, -self.hparams["n_ctx"]:]

            # Получаем предсказания логитов
            logits, _ = self(idx_cond)  # Предсказания на основе текущего контекста.
            logits = logits[:, -1, :]   # (B, C) берем только последние логиты.

            # Применяем температуру:
            logits = logits / (temp if temp != 0.0 else 1.0)  # Нормализуем логиты с учетом температуры.

            # Опционально обрезаем логиты до top_k лучших вариантов:
            if top_k is not None and top_k > 0 and isinstance(top_k, int):
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')  # Убираем значения вне top_k.

            # Применяем softmax для получения вероятностей:
            probs = TF.softmax(logits, dim=-1)  # (B, C).

            # Выбираем следующий токен:
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1) выбираем следующий токен.

            # Дополняем массив токенов:
            self.gen_tokens.append(idx_next.item())  # Добавляем токен к сгенерированным.

            # Добавляем индекс следующего токена к текущей последовательности:
            idx = torch.cat((idx, idx_next), dim=1)  # Обновляем входные данные.

        # Возвращаем сгенерированные токены:
        return self.gen_tokens
