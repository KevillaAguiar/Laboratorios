# Laboratório 6 — Construindo um Tokenizador BPE e Explorando o WordPiece

Neste laboratório foi implementado o algoritmo Byte Pair Encoding (BPE) do zero e explorado o funcionamento do WordPiece na prática utilizando o tokenizador multilíngue do BERT.

## Componentes Implementados

- **`get_stats(vocab)`** — varre o corpus e conta a frequência de todos os pares de símbolos adjacentes
- **`merge_vocab(pair, v_in)`** — funde o par mais frequente em todas as palavras do vocabulário
- **Loop de treinamento BPE** — executa `get_stats` e `merge_vocab` sucessivamente por 5 iterações
- **WordPiece na prática** — tokenização de frase em português com o `bert-base-multilingual-cased`

## Resultados do BPE (5 iterações)

```
Iteração 1: par fundido → ('e', 's')
Iteração 2: par fundido → ('es', 't')
Iteração 3: par fundido → ('est', '</w>')   ← sufixo morfológico formado!
Iteração 4: par fundido → ('l', 'o')
Iteração 5: par fundido → ('lo', 'w')
```

O algoritmo descobriu sozinho que `est</w>` é um sufixo recorrente em inglês, sem nenhuma regra linguística explícita.

## Resultado da Tokenização WordPiece

Frase tokenizada: `"Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."`

```
['Os', 'hip', '##er', '-', 'par', '##âm', '##etros', 'do', 'transform', '##er',
 'são', 'in', '##cons', '##tit', '##uc', '##ional', '##mente', 'di', '##f',
 '##í', '##cei', '##s', 'de', 'aj', '##usta', '##r', '.']
```

## O que significa o `##`?

O símbolo `##` indica que aquele fragmento é uma **continuação** do token anterior — ele não começa uma nova palavra. Por exemplo, `transform` + `##er` reconstrói "transformer", e `aj` + `##usta` + `##r` reconstrói "ajustar".

O WordPiece não corta as palavras seguindo regras gramaticais, mas sim nos pontos que aparecem com **maior frequência** no corpus de treinamento. Isso tem uma consequência importante: quando o modelo encontra uma palavra completamente desconhecida, ele não trava. Em vez de gerar um token `[UNK]` (desconhecido), ele decompõe a palavra em sub-partes que já conhece. Por exemplo, uma palavra nova como "transformerização" seria quebrada em pedaços familiares como `transform`, `##er`, `##iza`, `##ção` — permitindo que o modelo processe o significado aproximado mesmo sem ter visto a palavra antes.

## Como Executar

```bash
pip install transformers
python laboratorio_6.py
```

---

## Integridade Acadêmica

A função `merge_vocab` utiliza expressões regulares (`re`) para localizar e substituir os pares de símbolos nas strings do vocabulário. O padrão com `re.escape`, `(?<!\S)` e `(?!\S)` foi gerado com auxílio da IA generativa **Claude (Anthropic)** e revisado por Kévilla.

*Partes geradas/complementadas com IA (Claude - Anthropic), revisadas por Kévilla.*