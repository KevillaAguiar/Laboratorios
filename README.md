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
# Integridade Acadêmica

Ferramentas de **IA generativa** foram utilizadas apenas para esclarecimento de dúvidas conceituais e de sintaxe do `numpy`, conforme permitido nas instruções das atividades.

Todo o código foi desenvolvido manualmente com base nos conceitos estudados na disciplina.
