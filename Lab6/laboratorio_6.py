import re
from transformers import AutoTokenizer
#Tarefa 1
vocab = {
    'l o w </w>': 5,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3
}

def get_stats(vocab):
    pairs = {}
    for word, freq in vocab.items():
        symbols = word.split()  # separa em lista: ['n','e','w','e','s','t','</w>']
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs

stats = get_stats(vocab)
print(max(stats, key=stats.get), stats[('e', 's')])

#Tarefa 2
def merge_vocab(pair, v_in):
    v_out = {}
    # cria um padrão que encontra o par com espaço entre eles
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        # substitui o par pelo token fundido
        new_word = pattern.sub(''.join(pair), word)
        v_out[new_word] = v_in[word]
    return v_out

for i in range(5):
    stats = get_stats(vocab)
    best = max(stats, key=stats.get)
    vocab = merge_vocab(best, vocab)
    print(f"Iteração {i+1}: par fundido → {best}")
    print(f"Vocab: {vocab}\n")

#Tarefa 3
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

frase = "Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."
tokens = tokenizer.tokenize(frase)
print(tokens)