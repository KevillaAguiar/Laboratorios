import numpy as np

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


