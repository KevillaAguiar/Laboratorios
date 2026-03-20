# Laboratório 5 — Treinamento Fim-a-Fim do Transformer

Este laboratório conecta a arquitetura construída nos Labs 1 a 4 a um dataset real do Hugging Face, implementando o loop completo de treinamento com backpropagation e o otimizador Adam.

## Componentes Implementados

- **`FeedForwardNetwork(nn.Module)`** — rede densa com expansão de dimensão e ativação ReLU, migrada do Lab 4 para PyTorch
- **`scaled_dot_product_attention`** — função genérica de atenção, migrada do Lab 4 para PyTorch
- **`EncoderBlock(nn.Module)`** — bloco completo do Encoder com pesos treináveis
- **`DecoderBlock(nn.Module)`** — bloco completo do Decoder com Masked Self-Attention, Cross-Attention e FFN
- **`Transformer(nn.Module)`** — modelo completo com embeddings, Encoder, Decoder e projeção final
- **Loop de treinamento** — Forward, Loss, Backward, Step com mini-batches
- **Overfitting Test** — prova de que os gradientes fluem corretamente pela arquitetura

## Dataset

Dataset utilizado: `bentrevett/multi30k` (pares inglês → alemão)

Subconjunto de treinamento: 1.000 frases das 29.000 disponíveis

Tokenizador: `bert-base-multilingual-cased` (Hugging Face)

## Fluxo do Treinamento

```
1. Forward  → passa os dados pelo modelo e obtém a previsão
2. Loss     → CrossEntropyLoss compara previsão com resposta esperada
3. Backward → calcula os gradientes
4. Step     → Adam atualiza os pesos
5. Zera     → limpa os gradientes para a próxima rodada
```

## Resultados

### Training Loop (10 épocas, 1.000 frases, batch_size=16)
```
Época  1 → Loss: 11.07
Época  2 → Loss:  9.54
Época  3 → Loss:  8.08
Época  4 → Loss:  6.98
Época  5 → Loss:  6.32
Época  6 → Loss:  5.98
Época  7 → Loss:  5.80
Época  8 → Loss:  5.67
Época  9 → Loss:  5.56
Época 10 → Loss:  5.45
```

### Overfitting Test (50 épocas, 1 frase)
```
Época 10 → Loss: 4.32
Época 20 → Loss: 3.25
Época 30 → Loss: 2.30
Época 40 → Loss: 1.56
Época 50 → Loss: 1.04
```

O Loss caiu mais de 75% na frase isolada, provando que os gradientes fluem corretamente por toda a arquitetura.

## Como Executar

```bash
pip install datasets transformers torch
python laboratorio_5.py
```

---

*Partes geradas/complementadas com IA (Claude - Anthropic), revisadas por Kévilla.*