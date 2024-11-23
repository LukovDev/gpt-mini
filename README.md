 # GPT-Mini - Мини Генеративная Предварительно-обученная Трансформенная модель.

Это ИИ на основе GPT-2 но с рядом модификаций и улучшений.

Эта база кода позволяет как обучать модель (с нуля или дообучать), так и генерировать текст.</br>

В этом коде реализовано 2 токенизатора - Посимвольный (```TokenizerSymbol```) и пословный (```TokenizerWord```).</br>
Для нормальной модели лучше использовать токенизатор на основе ```BPE```.</br>

Класс ```GPTLLM``` работает на ```GELU``` функции, как самой универсальной. Вы можете заменить её на ```ReLU```, если того требует модель, или на что-то другое.

#

Если доработать этот код, то возможно, на нём можно будет запускать другие модели, например GPT-2 или SberGPT.

Если вы столкнулись с проблемой когда ```torch``` не хочет работать на ```cuda```, хотя вы уверены что ваша видеокарта поддерживает это, попробуйте это:</br>
```pip install torchvision==0.20.0+cu118 torchaudio==2.5.0+cu118 --index-url https://download.pytorch.org/whl/cu118```

#

### Как использовать ИИ:
```python
# Загружаем модель:
vocab, hparams, train_cfg = load_parameters(model_path)

# Создаём токенизатор:
tokenizer = TokenizerSymbol(vocab=vocab)

# Создаём нейросеть:
gpt = GPTLLM(hparams, "cuda")  # Первый параметр - Гиперпараметры нейросети. Второй параметр - Вычислительное устройство.
print(f"Device: {gpt.device}")

gpt.load(os.path.join(model_path, "model.bin"))  # Загружаем веса обученной нейросети в нашу нейросеть.

# Генерируем текст:

# Для начала превращаем наш текст в токены:
tokens = tokenizer.encode("Привет ИИ!")

# Генерируем предсказанные токены:
# Первый параметр - Набор токенов для генерации. Второй параметр - Максимальное количество генерируемых токенов.
# Третий параметр - Температура генерации. Четвёртый параметр - Top_K коэф.
# Пятый параметр - Использование шкалы прогресса tqdm при генерации токенов.
gen_tokens = gpt.generate(tokens, max_tokens=1000, temp=0.4, top_k=0, tqdm_use=True)

# Декодируем токены и выводим результат:
print(tokenizer.decode(gen_tokens))
```

### Как обучать модель:
```python
import torch

# Загружаем текст для тренировки нейросети:
train_text = ""
with open("data/texts/shakespeare.txt", "r+", encoding="utf-8") as f: train_text = f.read()

# Создаём токенизатор:
tokenizer = TokenizerSymbol(text=train_text)

# Новые кастомные параметры модели:
vocab = tokenizer.stoi

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

# Сохраняем параметры нейросети:
save_parameters(model_path, vocab, hparams, train_cfg)

# Создаём нейросеть:
gpt = GPTLLM(hparams, "cuda")  # Первый параметр - Гиперпараметры нейросети. Второй параметр - Вычислительное устройство.
print(f"Device: {gpt.device}")

# Подготавливаем текст для тренировки нейросети:
data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long).to(gpt.device)
n = int(0.9*len(data))  # Первые 90% - это тренировочные данные, остальные 10% - Проверочные данные.
train_data = data[:n]
valid_data = data[n:]

# Класс для обучения нейросети:
# Первый параметр - Конфигурация тренеровщика. Второй параметр - Наша нейросеть.
# Третий параметр - Тренировочные данные. Четвёртый параметр - Проверочные данные.
trainer = TrainGPTLLM(train_cfg, gpt, train_data, valid_data)

# Тренеруем нейросеть:
# Первый параметр - Путь до папки с автосохранением модели.
# Второй параметр - Частота автосохранения модели (в итерациях).
# Третий параметр - Уровень разности потерь между тренировочными данными и проверочными.
# В случае превышения этого значения, обучение прекращается.
trainer.train(model_path, 100, 3.0)  # Все параметры необязательны.

# Сохраняем натренированную модель:
gpt.save(os.path.join(model_path, "model.bin"))
```

#

Код в целом простой и в нём можно легко разобраться. Можете модифицировать код внутренних классов и улучшать, например, код обучения или генерации текста.

#

Пример генерации текста на маленькой модели обученной на этой же базе кода, на наборе данных произведений Шекспира:</br>
![](https://github.com/user-attachments/assets/875a041b-9355-429b-b2fb-99df8f475e5d)

#

Характеристики маленькой модели:
- Модель использует посимвольную токенизацию и генерацию.
- Модель работает на GELU функции активации.
- Модель состоит из 10.8 миллионов параметров.
- Размер учитываемого контекста при генерации составляет 256 токенов.
- Размер векторного представления токена составляет 384.
- Количество голов внимания и слоёв трансфо
- Итоговый размер модели составляет 50.3 мб.

Во время обучения:
- Размер пакета составляет 64.
- Количество итераций обучения составляет 5000.
- Шаг обучения составляет 3e-4.
- Уровень сброса нейрона (dropout) составляет 0.15.
- Обучалась на GTX 1660 около 50 минут.
