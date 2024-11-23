import torch  # pip install torchvision==0.20.0+cu118 torchaudio==2.5.0+cu118 --index-url https://download.pytorch.org/whl/cu118
import torch.nn as nn
from torch.nn import functional as TF
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size    = 64  # Размер пакета данных, используемый для обучения в одной итерации.
block_size    = 256  # Максимальная длина последовательности (количество токенов) для одного входного примера.
max_iters     = 2000  # Количество итераций для обучения.
eval_interval = 500  # Как часто модель будет оцениваться на валидационном наборе.
learning_rate = 0.0001  # Определяет, насколько сильно обновляются веса в процессе обучения.
eval_iters    = 200  # Сколько итераций будет использоваться для оценки ошибки во время валидации.
n_embd        = 384  # Размер векторного представления для токенов.
n_head        = 6  # Количество "голов" в механизме многошагового внимания (multi-head attention).
n_layer       = 6  # Количество блоков в трансформере. Каждый блок состоит из слоя внимания и слоя feed-forward.
dropout       = 0.1  # Вероятность "выключения" нейронов во время обучения для предотвращения переобучения.

training      = False
save_full     = False
file_name     = "model.bin"


print(f"Device: {device}")


train_text = ""
with open("train-text.txt", "r+", encoding="utf-8") as f:
    train_text = f.read()


chars = sorted(list(set(train_text)))
vocab_size = len(chars)

stoi = { c:i for i,c in enumerate(chars) }
itos = { i:c for i,c in enumerate(chars) }
encode = lambda s: [stoi[c] if c in stoi else -1 for c in s]
decode = lambda l: "".join([itos[i] if i != -1 else "⍰" for i in l])

print(vocab_size)
print(stoi)

# Train and test splits
data = torch.tensor(encode(train_text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# Функция генерирует пакеты данных для обучения и валидации:
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Реализует одну "голову" внимания:
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = TF.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out



# Класс создает несколько "голов" внимания, каждая из которых работает параллельно:
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# Простая нейронная сеть с двумя слоями:
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Один блок трансформера, который состоит из слоя внимания и слоя "вычислений" (feed-forward):
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = TF.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, prompt, max_tokens, temperature=1.0, threshold=0.01):
        if temperature == 0.0: temperature = 1e-10
        idx = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
        text = ""

        for _ in tqdm(range(max_tokens), "Generating"):
            idx_cond = idx[:, -block_size:]

            # Получаем предсказания логитов
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, C)

            # Применяем температуру
            logits = logits / temperature  # Нормализуем логиты с учетом температуры.

            # Применяем softmax для получения вероятностей
            probs = TF.softmax(logits, dim=-1)  # (B, C)

            # Проверяем, все ли вероятности ниже порога
            if torch.max(probs) < threshold:
                print("\nBreaked")
                break  # Останавливаем генерацию, если максимальная вероятность ниже порога

            # Выбираем следующий токен
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Дополняем ответ:
            text += decode([idx_next.item()])

            # Добавляем индекс следующего токена к текущей последовательности
            idx = torch.cat((idx, idx_next), dim=1)
        return text

model = GPTLanguageModel()
if not training:
    if save_full: torch.load(file_name, map_location=device, weights_only=True)
    else: model.load_state_dict(torch.load(file_name, map_location=device, weights_only=True))
model = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, "M parameters.")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

if training:
    for iter in tqdm(range(max_iters), "Training"):

        # every once in a while evaluate the loss on train and val sets
        # if iter % eval_interval == 0 or iter == max_iters - 1:
        #     losses = estimate_loss()
        #     print(f" | train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", end="")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        if iter % eval_interval == 0 or iter == max_iters - 1: print(loss.item())
        if loss.item() <= 1.0: break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if save_full: torch.save(model, file_name)
    else: torch.save(model.state_dict(), file_name)

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)

prompt = "Андрей:\nСегодня я был в школе."
out = model.generate(encode(prompt), max_tokens=2048, temperature=1.0, threshold=0.01)
print(prompt+out)
