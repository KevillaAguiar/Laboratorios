from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim

#Tarefa 1
dataset = load_dataset("bentrevett/multi30k")
subset = dataset["train"].select(range(1000))

#Tarefa 2
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def tokenizar_dataset(subset):
    pares = []
    for item in subset:
        en = tokenizer.encode(item["en"])
        de = tokenizer.encode(item["de"])
        pares.append((en, de))
    return pares

def aplicar_padding(pares):
    # garante que todas as sequências tenham o mesmo comprimento
    max_len = max(max(len(en), len(de)) for en, de in pares)
    pares_padded = []
    for en, de in pares:
        en_padded = en + [0] * (max_len - len(en))
        de_padded = de + [0] * (max_len - len(de))
        pares_padded.append((en_padded, de_padded))
    return pares_padded, max_len

pares_tokenizados = tokenizar_dataset(subset)
pares_padded, max_len = aplicar_padding(pares_tokenizados)

src_tensor = torch.tensor([en for en, de in pares_padded])
tgt_tensor = torch.tensor([de for en, de in pares_padded])

#Tarefa 3
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None:
        scores = scores + mask
    attention_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, V)

class EncoderBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        att = scaled_dot_product_attention(Q, K, V)
        x = self.norm1(x + att)
        x = self.norm2(x + self.ffn(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_q1 = nn.Linear(d_model, d_model)
        self.W_k1 = nn.Linear(d_model, d_model)
        self.W_v1 = nn.Linear(d_model, d_model)
        self.W_q2 = nn.Linear(d_model, d_model)
        self.W_k2 = nn.Linear(d_model, d_model)
        self.W_v2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model)

    def forward(self, y, Z):
        seq_len = y.shape[1]
        # máscara causal impede o decoder de ver tokens futuros
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)

        Q = self.W_q1(y)
        K = self.W_k1(y)
        V = self.W_v1(y)
        att1 = scaled_dot_product_attention(Q, K, V, mask=mask)
        y = self.norm1(y + att1)

        Q = self.W_q2(y)
        K = self.W_k2(Z)
        V = self.W_v2(Z)
        att2 = scaled_dot_product_attention(Q, K, V)
        y = self.norm2(y + att2)

        y = self.norm3(y + self.ffn(y))
        return y

class Transformer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = EncoderBlock(d_model)
        self.decoder = DecoderBlock(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.encoder_embedding(src)
        tgt = self.decoder_embedding(tgt)
        Z = self.encoder(src)
        out = self.decoder(tgt, Z)
        return self.linear(out)

vocab_size = tokenizer.vocab_size
modelo = Transformer(d_model=128, vocab_size=vocab_size)
criterio = nn.CrossEntropyLoss(ignore_index=0)
otimizador = optim.Adam(modelo.parameters(), lr=0.0001)

batch_size = 16

for epoca in range(10):
    loss_total = 0
    num_batches = 0

    for i in range(0, len(src_tensor), batch_size):
        src = src_tensor[i:i+batch_size]
        tgt_input = tgt_tensor[i:i+batch_size, :-1]
        tgt_output = tgt_tensor[i:i+batch_size, 1:]

        otimizador.zero_grad()
        previsao = modelo(src, tgt_input)
        loss = criterio(previsao.reshape(-1, vocab_size), tgt_output.reshape(-1))
        loss.backward()
        otimizador.step()

        loss_total += loss.item()
        num_batches += 1

    print(f"Época {epoca+1}/10 — Loss médio: {loss_total/num_batches:.4f}")

#Tarefa 4
src_one = src_tensor[0:1]
tgt_one = tgt_tensor[0:1]

print("Frase original EN:", subset[0]["en"])
print("Tradução esperada DE:", subset[0]["de"])

for epoca in range(50):
    otimizador.zero_grad()
    previsao = modelo(src_one, tgt_one[:, :-1])
    loss = criterio(previsao.reshape(-1, vocab_size), tgt_one[:, 1:].reshape(-1))
    loss.backward()
    otimizador.step()

    if (epoca+1) % 10 == 0:
        print(f"Época {epoca+1}/50 — Loss: {loss.item():.4f}")