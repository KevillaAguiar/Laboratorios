import numpy as np

def scaled_dot_product_attention(Q, K, V):

    product_scalar = np.matmul(Q, K.T)

    d_k = K.shape[1]

    scaled_scores = product_scalar/np.sqrt(d_k)

    exp_scores = np.exp(scaled_scores)
    sums = np.sum(exp_scores, axis=1, keepdims=True)
    attention_weights = exp_scores/sums

    output = attention_weights @ V

    return output


