# Laboratório 2 — Encoder Transformer Simplificado

## Descrição

Este laboratório consiste na implementação didática de um **Encoder Transformer simplificado** utilizando apenas a biblioteca `numpy`.

O objetivo é compreender como funciona o mecanismo de **Self-Attention** introduzido no artigo *Attention Is All You Need*. O programa recebe uma frase, transforma as palavras em **embeddings vetoriais** e processa esses vetores através de **6 camadas de encoder**, produzindo representações contextualizadas para cada palavra.

A implementação inclui os seguintes componentes principais:

* Conversão de palavras em **IDs**
* **Embedding** de palavras
* **Scaled Dot-Product Self-Attention**
* **Conexões residuais + Layer Normalization**
* **Feed Forward Network**
* Empilhamento de **6 camadas do encoder**

O resultado final é um tensor contendo os vetores contextualizados de cada token da frase.


## Como executar o código

### 1. Instalar dependências

O projeto utiliza apenas bibliotecas básicas do Python:

```
pip install numpy pandas
```

ou, caso exista um arquivo de dependências:

```
pip install -r requirements.txt
```

### 2. Executar o programa

No terminal, dentro da pasta do laboratório, execute:

```
python encoder.py
```

O script irá:

1. Criar um vocabulário simples
2. Converter a frase em tokens numéricos
3. Gerar embeddings
4. Processar os vetores pelas **6 camadas do encoder**
5. Exibir o tensor final com as representações vetoriais contextualizadas.
