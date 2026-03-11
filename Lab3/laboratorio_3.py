import numpy as np

def create_causal_mask(seq_len):
    matriz = np.ones((seq_len, seq_len))

    casual_mask = np.triu(matriz, k=1)

    M = np.where(casual_mask > 0, -np.inf, 0.0)

    return print(M)

def cross_attention(encoder_output, decoder_state, W_q, W_k, W_v):
    Q = decoder_state @ W_q
    K= encoder_output @ W_k
    V = encoder_output @ W_v

    product_scalar = np.matmul(Q, K.transpose(0, 2, 1))

    d_k = K.shape[-1]

    scaled_scores = product_scalar/np.sqrt(d_k)

    exp_scores = np.exp(scaled_scores)
    sums = np.sum(exp_scores, axis=-1, keepdims=True)
    attention_weights = exp_scores/sums

    output = attention_weights @ V

    return output




encoder_output = np.random.randn(1, 10, 512)
decoder_state = np.random.randn(1, 4, 512)

W_q = np.random.randn([512, 512])
W_k = np.random.randn([512, 512])
W_v = np.random.randn([512, 512])