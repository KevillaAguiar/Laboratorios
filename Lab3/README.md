# Laboratórios

## Laboratório 3 – Implementando o Decoder

### Como executar

1. Instalar dependências

O projeto utiliza apenas bibliotecas básicas:

pip install numpy

2. Executar o script

No terminal, dentro da pasta do laboratório:

python decoder.py

### Tarefa 1 – Máscara Causal (Look-Ahead Mask)

A máscara causal impede que um token na posição `i` atenda tokens em posições futuras `i+1, i+2, ...` durante o treinamento.

A matriz retornada tem shape `[seq_len, seq_len]` onde a diagonal principal e o triângulo inferior contêm `0` e o triângulo superior contém `-inf`.

Após somar a máscara aos scores e aplicar o softmax, as posições futuras se tornam estritamente `0.0`, pois:

$$e^{-\infty} = 0$$

A equação completa é:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

### Exemplo

Chamada:

```python
create_causal_mask(seq_len=5)
```

Saída esperada (Prova Real): matriz `(5, 5)` onde a parte superior é estritamente `0.0` após o softmax.

```
[[1.     0.     0.     0.     0.    ]
 [0.306  0.693  0.     0.     0.    ]
 [0.353  0.125  0.521  0.     0.    ]
 [0.038  0.657  0.137  0.166  0.    ]
 [0.387  0.056  0.128  0.337  0.091 ]]
```

### Tarefa 2 – Cross-Attention (Ponte Encoder-Decoder)

No Cross-Attention, Q vem do Decoder e K, V vem do Encoder. Isso permite que o Decoder consulte a frase original a cada passo da geração.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Diferente da Tarefa 1, não há máscara causal — o Decoder tem permissão para olhar para a frase do Encoder por completo.

### Exemplo

Tensores de entrada:

```python
encoder_output = np.random.randn(1, 10, 512)  # frase em francês
decoder_state  = np.random.randn(1, 4, 512)   # tokens já gerados em inglês
```

Chamada:

```python
cross_attention(encoder_output, decoder_state, W_q, W_k, W_v)
```

Saída esperada: tensor de shape `(1, 4, 512)`.


### Tarefa 3 – Loop de Inferência Auto-Regressivo

O Decoder gera uma palavra por vez. A cada iteração, o token gerado é adicionado à sequência e usado como entrada na próxima chamada. O loop para quando o modelo gera o token `<EOS>`.

### Exemplo

Sequência inicial:

```python
current_sequence = ["<start>", "O", "rato"]
```

A cada iteração:

1. `generate_next_token` retorna um vetor de probabilidades de tamanho `10.000`
2. `np.argmax` seleciona o índice com maior probabilidade
3. O índice é adicionado à sequência
4. Se o índice for `EOS_INDEX = 9999`, o loop para e a frase é impressa