# Laboratório 4 — O Transformer Completo "From Scratch"

Este laboratório integra todos os componentes construídos nos Labs 1, 2 e 3 em uma arquitetura Encoder-Decoder completa, capaz de realizar inferência auto-regressiva fim-a-fim usando apenas NumPy.

## Componentes Implementados

- **`scaled_dot_product_attention(Q, K, V, mask=None)`** — mecanismo de atenção genérico, reutilizado para Self-Attention, Masked Self-Attention e Cross-Attention
- **`LayerNorm(x, sublayer_output)`** — conexão residual seguida de normalização de camada
- **`FeedForwardNetwork(output)`** — rede densa com expansão de dimensão e ativação ReLU
- **`create_causal_mask(seq_len)`** — gera a máscara triangular superior com `-inf` para impedir que o Decoder veja tokens futuros
- **`EncoderBlock(x)`** — bloco completo do Encoder
- **`DecoderBlock(y, Z)`** — bloco completo do Decoder com Cross-Attention acoplado à memória do Encoder
- **Loop de inferência auto-regressivo** — gera tokens até encontrar `<EOS>`

## Fluxo da Arquitetura

### Encoder
```
x → Self-Attention(x, x, x) → Add & Norm → FFN → Add & Norm → Z
```

### Decoder
```
y → Masked Self-Attention(y, y, y, mask) → Add & Norm
  → Cross-Attention(Q=norm, K=Z, V=Z)   → Add & Norm
  → FFN                                  → Add & Norm
  → Linear → Softmax → probabilidades
```

### Inferência
1. O `encoder_input` (simulando "Thinking Machines") passa pelo `EncoderBlock` e gera a memória `Z`
2. O Decoder é inicializado com o token `<START>`
3. A cada iteração, o Decoder recebe a sequência gerada até o momento e prevê o próximo token via `argmax` das probabilidades
4. O novo token é concatenado à entrada do Decoder
5. O loop encerra quando o token `<EOS>` é gerado

## Como Executar

```bash
python laboratorio_4.py
```

Não há dependências externas além do NumPy.

---

*Partes geradas/complementadas com IA (Claude - Anthropic), revisadas por Kévilla.*