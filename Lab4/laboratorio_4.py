import numpy as np

#Tarefa 1
def scaled_dot_product_attention(Q, K, V, mask=None):

    product_scalar = np.matmul(Q, K.transpose(0, 2, 1))

    d_k = K.shape[-1]

    scaled_scores = product_scalar/np.sqrt(d_k)

    if mask is not None:
        scaled_scores = scaled_scores + mask

    exp_scores = np.exp(scaled_scores)
    sums = np.sum(exp_scores, axis=-1, keepdims=True)
    attention_weights = exp_scores/sums

    output = attention_weights @ V

    return output

def LayerNorm(X, output):

    Add = X + output

    eps = 1e-6
    mean = np.mean(Add, axis=-1, keepdims=True)
    var = np.var(Add, axis=-1, keepdims=True)

    LNormr = (Add - mean)/(np.sqrt(var + eps))

    return LNormr

def FeedForwardNetwork(output):

    d_model = output.shape[-1]
    d_ff = d_model*4

    W1 = np.random.rand(d_model, d_ff)
    W2 = np.random.rand(d_ff, d_model)
    b1 = np.zeros(d_ff)
    b2 = np.zeros(d_model)

    T_Linear_1 = (output @ W1) + b1

    FFN = (np.maximum(0, T_Linear_1) @ W2) + b2

    return FFN

def create_causal_mask(seq_len):
    matriz = np.ones((seq_len, seq_len))
    casual_mask = np.triu(matriz, k=1)
    M = np.where(casual_mask > 0, -np.inf, 0.0)

    return M

#Tarefa 2
def EncoderBlock(x):
    # Self Attention
    X_att = scaled_dot_product_attention(x, x, x)

    # Residual + LayerNorm
    X_norm1 = LayerNorm(x, X_att)

    # Feed Forward
    X_ffn = FeedForwardNetwork(X_norm1)

    # Residual + LayerNorm
    X_out = LayerNorm(X_norm1, X_ffn)

    # Output vira input da próxima camada
    X = X_out

    return X

#Tarefa 3
def DecoderBlock(y, Z):

    mask = create_causal_mask(y.shape[1])
    Masked_Self_Attention = scaled_dot_product_attention(y, y, y, mask=mask) 

    Normalization = LayerNorm(y, Masked_Self_Attention)
    Cross_Attention = scaled_dot_product_attention(Normalization, Z, Z)

    Normalization_2 = LayerNorm(Normalization, Cross_Attention)
    ffn = FeedForwardNetwork(Normalization_2)

    out = LayerNorm(Normalization_2, ffn)

    vocab_size = 100
    W_out = np.random.rand(out.shape[-1], vocab_size)
    logits = out @ W_out
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    return probs