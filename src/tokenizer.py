#
# tokenizer.py - Создаёт простые функции для токенизации.
#


# Импортируем:
import re
from functools import lru_cache


# Класс символьного токенизатора:
class TokenizerSymbol:
    def __init__(self, text: str = None, vocab: dict = None) -> None:
        self.vocab = sorted(list(set(text))) if vocab is None else sorted(list(vocab.keys()))
        self.vocab_size = len(self.vocab)

        # Токен и его индекс:
        self.stoi = {c:i for i, c in enumerate(self.vocab)}

        # Индекс и его токен:
        self.itos = {i:c for i, c in enumerate(self.vocab)}

        # Функции для превращения строки в список токенов и наоборот:
        self.encode = lambda s: [self.stoi[c] if c in self.stoi else -1 for c in s]
        self.decode = lambda l: "".join([self.itos[i] if i != -1 else "⍰" for i in l])


# Класс словарного токенизатора:
class TokenizerWord:
    def __init__(self, text: str = None, vocab: dict = None) -> None:
        # Инициализация словаря с токеном <unk>, если словарь не передан:
        self.vocab = {"<unk>": 0} if vocab is None else vocab

        if text is not None:
            # Собираем все уникальные слова и сортируем их (спецсимволы самые первые):
            unique_words = sorted(
                set(text.split()),
                key=lambda word: (1, word) if word[0].isalnum() else (0, word)
            )

            # Генерируем словарный запас: ключом является токен, а значением — его идентификатор:
            self.vocab.update({word: index + 1 for index, word in enumerate(unique_words)})

    # Получить размер словарного запаса:
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # Закодировать текст в токены:
    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(word, self.vocab["<unk>"]) for word in text.strip().split()]

    # Декодировать токены в текст:
    def decode(self, ids: list[int]) -> str:
        # Мы инвертируем vocab для поиска слова по идентификатору
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return " ".join(inv_vocab.get(i, "<unk>") for i in ids) if ids is not None else ""
