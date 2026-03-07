import numpy as np
import pandas as pd 

#passo 1
def preparar_vocab():
    vocabulario = pd.DataFrame({
        "palavra": ["estou", "com", "muita", "fome"],
        "id": [0, 1, 2, 3]
    })

    frase = "estou com muita fome"

    lista_id = [
        vocabulario[vocabulario["palavra"] == palavra]["id"].values[0]
        for palavra in frase.split()
    ]

    return vocabulario, lista_id

def embedding(lista_id):
    d_model = 3

    tb_embeddings = np.random.rand(len(lista_id), d_model)

    X = tb_embeddings[lista_id]

    X = np.expand_dims(X, axis=0)

    return X

#passo 2

def scaled_dot_product_attention(X):

    d_model = 3

    Wq = np.random.rand(d_model, d_model)
    Wk = np.random.rand(d_model, d_model)
    Wv = np.random.rand(d_model, d_model)

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    product_scalar = np.matmul(Q, K.transpose(0,2,1))

    #d_k = K.shape[1]

    scaled_scores = product_scalar/np.sqrt(d_model)

    exp_scores = np.exp(scaled_scores)
    sums = np.sum(exp_scores, axis=-1, keepdims=True)
    attention_weights = exp_scores/sums

    output = attention_weights @ V

    return output

#Conexões Residuais e Normalização

def LayerNorm(X, output):

    Add = X + output

    eps = 1e-6
    mean = np.mean(Add, axis=-1, keepdims=True)
    var = np.var(Add, axis=-1, keepdims=True)

    LNormr = (Add - mean)/(np.sqrt(var + eps))

    return LNormr

def FeedForwardNetwork(output):

    d_model = 3
    d_ff = d_model*4

    W1 = np.random.rand(d_model, d_ff)
    W2 = np.random.rand(d_ff, d_model)
    b1 = np.zeros(d_ff)
    b2 = np.zeros(d_model)

    T_Linear_1 = (output @ W1) + b1

    FFN = (np.maximum(0, T_Linear_1) @ W2) + b2

    return FFN

#passo 3

vocabulario, lista_id = preparar_vocab()
X = embedding(lista_id)

for i in range(1, 7):
    # Self Attention
    X_att = scaled_dot_product_attention(X)

    # Residual + LayerNorm
    X_norm1 = LayerNorm(X, X_att)

    # Feed Forward
    X_ffn = FeedForwardNetwork(X_norm1)

    # Residual + LayerNorm
    X_out = LayerNorm(X_norm1, X_ffn)

    # Output vira input da próxima camada
    X = X_out
    
print(X)
