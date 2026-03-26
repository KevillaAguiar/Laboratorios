# Laboratórios — Tópicos Avançados em Inteligência Artificial
Este repositório reúne as implementações desenvolvidas durante a disciplina **Tópicos Avançados em Inteligência Artificial**.
O objetivo das atividades é compreender, na prática, os principais mecanismos utilizados em **modelos modernos de linguagem baseados em Transformers**, implementando seus componentes fundamentais utilizando **Python e NumPy**.
Os laboratórios são organizados em pastas separadas, onde cada atividade explora uma parte específica da arquitetura.
---
# Estrutura do Repositório
```
.
├── Lab1
├── Lab2
├── Lab3
├── Lab4
├── Lab5
├── Lab6
└── README.md
```
Cada pasta contém o código do laboratório e um README específico com explicações e instruções de execução.
---
# Laboratórios Implementados

## Lab1 — Self-Attention
No primeiro laboratório foi implementado o mecanismo de **Self-Attention**, que é o componente central da arquitetura Transformer.
Nesta atividade foram explorados os conceitos de:
* **Query (Q)**
* **Key (K)**
* **Value (V)**
* Produto escalar entre vetores
* Aplicação de **Softmax** para gerar pesos de atenção

O objetivo foi entender como cada palavra de uma sequência pode **atribuir diferentes níveis de importância às outras palavras da frase**, produzindo representações contextualizadas.

---

## Lab2 — Encoder Transformer
Neste laboratório foi implementada uma versão simplificada do **Encoder do Transformer**.
O objetivo foi compreender como o modelo extrai **representações contextuais completas de uma frase** utilizando múltiplas camadas de atenção.
Componentes implementados:
* Conversão de palavras em **IDs**
* **Tabela de Embeddings**
* **Scaled Dot-Product Self-Attention**
* **Conexões Residuais + Layer Normalization**
* **Feed Forward Network**
* Empilhamento de **6 camadas do Encoder**

O resultado final é um tensor contendo os **vetores contextualizados de cada token da frase**.

---

## Lab3 — Decoder Transformer
Neste laboratório foi implementada a parte responsável pela **geração de texto** em um modelo Transformer.
Foram implementados três conceitos principais:

### Máscara Causal (Look-Ahead Mask)
Criação de uma matriz que impede que o modelo **acesse palavras futuras** durante o processamento da sequência.

### Cross-Attention
Implementação da atenção entre **Encoder e Decoder**, permitindo que o Decoder utilize as representações produzidas pelo Encoder como memória durante a geração do texto.

### Loop Auto-Regressivo
Simulação do processo de geração de texto onde o modelo produz **um token por vez**, utilizando a saída anterior como contexto para prever a próxima palavra.
O processo continua até que o modelo gere o token especial **`<EOS>`**, indicando o fim da frase.

---

## Lab4 — Transformer Completo (Encoder-Decoder)
Neste laboratório todos os componentes anteriores foram integrados em uma **arquitetura Encoder-Decoder completa**, capaz de realizar inferência auto-regressiva fim-a-fim usando apenas NumPy.

Componentes refatorados e integrados:
* **`scaled_dot_product_attention(Q, K, V, mask=None)`** — função genérica reutilizada para Self-Attention, Masked Self-Attention e Cross-Attention
* **`LayerNorm`** — conexão residual + normalização de camada
* **`FeedForwardNetwork`** — rede densa com expansão de dimensão e ativação ReLU
* **`create_causal_mask`** — máscara triangular com `-inf` para impedir acesso a tokens futuros

Blocos construídos:
* **`EncoderBlock(x)`** — Self-Attention → Add & Norm → FFN → Add & Norm
* **`DecoderBlock(y, Z)`** — Masked Self-Attention → Add & Norm → Cross-Attention → Add & Norm → FFN → Add & Norm → Linear → Softmax

O teste de inferência simula a tradução da frase **"Thinking Machines"**, onde o Decoder gera tokens auto-regressivamente a partir do token `<START>` até produzir o token `<EOS>`.

---

## Lab5 — Treinamento Fim-a-Fim do Transformer
Neste laboratório final da Unidade I, a arquitetura do Lab 4 foi migrada para **PyTorch** e conectada a um dataset real do Hugging Face, implementando o loop completo de treinamento com backpropagation.

Dataset utilizado: `bentrevett/multi30k` (pares inglês → alemão), subconjunto de 1.000 frases.

Componentes implementados:
* Carregamento e tokenização do dataset com `datasets` e `transformers` do Hugging Face
* Padding das sequências para comprimento uniforme
* Migração dos blocos do Lab 4 para classes `nn.Module` do PyTorch
* **Loop de treinamento** com `CrossEntropyLoss` e otimizador `Adam`
* **Overfitting Test** — prova de convergência forçando o modelo a memorizar uma frase

Resultados obtidos:
* Loss caiu de **11.07 → 5.45** em 10 épocas no dataset completo
* Loss caiu de **4.32 → 1.04** em 50 épocas no teste de overfitting

---

## Lab6 — Tokenizador BPE e WordPiece
Neste laboratório foi implementado o algoritmo **Byte Pair Encoding (BPE)** do zero e explorado o funcionamento do **WordPiece** na prática.

Componentes implementados:
* **`get_stats(vocab)`** — conta a frequência de pares de símbolos adjacentes no corpus
* **`merge_vocab(pair, v_in)`** — funde o par mais frequente em todas as palavras do vocabulário
* **Loop de treinamento BPE** — 5 iterações formando tokens morfológicos como `est</w>`
* **WordPiece** — tokenização com `bert-base-multilingual-cased` do Hugging Face

O algoritmo BPE descobriu sozinho o sufixo `est</w>` após 3 iterações, sem nenhuma regra linguística explícita. O símbolo `##` no WordPiece indica continuação de token — mecanismo que impede o travamento do modelo diante de palavras desconhecidas.

---

# Integridade Acadêmica
O desenvolvimento deste repositório contou com o auxílio da ferramenta de IA generativa **Claude (Anthropic)** nas seguintes frentes:

* Revisão e correção do código escrito pela aluna, com explicações sobre cada ajuste
* Esclarecimento de dúvidas conceituais sobre arquitetura Transformer e manipulação de tensores com NumPy e PyTorch
* Implementação do loop de inferência auto-regressivo (Lab4 — Tarefa 4)
* Implementação guiada do loop de treinamento e overfitting test (Lab5 — Tarefas 3 e 4)
* Implementação da função `merge_vocab` com expressões regulares (Lab6 — Tarefa 2)
* Redação dos arquivos README

Todo o restante do código foi desenvolvido manualmente pela aluna com base nos conceitos estudados na disciplina.

*Partes geradas/complementadas com IA (Claude - Anthropic), revisadas por Kévilla.*