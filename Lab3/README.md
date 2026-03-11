# Laboratório 3 — Implementando o Decoder

## Descrição

Este laboratório consiste na implementação didática dos **blocos matemáticos centrais do Decoder Transformer** utilizando apenas a biblioteca `numpy`.

O objetivo é compreender como o Decoder gera texto de forma auto-regressiva, garantindo que o modelo não "olhe para o futuro" durante o processamento. O programa implementa a máscara causal, a ponte de comunicação entre Encoder e Decoder via Cross-Attention, e o loop de inferência que gera tokens um a um.

A implementação inclui os seguintes componentes principais:

* **Máscara Causal (Look-Ahead Mask)** — impede que tokens atendam a posições futuras
* **Scaled Dot-Product Attention com máscara** — prova real de que posições futuras viram `0.0` após o softmax
* **Cross-Attention (Encoder-Decoder Attention)** — ponte entre a memória do Encoder e o estado atual do Decoder
* **Loop de Inferência Auto-Regressivo** — geração de tokens iterativa com condição de parada `<EOS>`

## Como executar o código

### 1. Instalar dependências

O projeto utiliza apenas bibliotecas básicas do Python:

```
pip install numpy
```

### 2. Executar o programa

No terminal, dentro da pasta do laboratório, execute:

```
python laboratorio_3.py
```

O script irá:

1. Criar a máscara causal e imprimir a **Prova Real** — uma matriz onde as posições futuras são estritamente `0.0` após o softmax
2. Criar tensores fictícios simulando a saída do Encoder (`encoder_output`) e o estado atual do Decoder (`decoder_state`)
3. Executar o **Cross-Attention**, cruzando as representações do Encoder com as do Decoder
4. Exibir o shape do tensor de saída do Cross-Attention
5. Iniciar o **loop de inferência**, gerando tokens até encontrar o token de parada `<EOS>`
6. Imprimir a frase final gerada