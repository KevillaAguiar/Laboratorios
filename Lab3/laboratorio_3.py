import numpy as np

#Tarefa 1
def create_causal_mask(seq_len):
    matriz = np.ones((seq_len, seq_len))
    casual_mask = np.triu(matriz, k=1)
    M = np.where(casual_mask > 0, -np.inf, 0.0)

    d_k = 64
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)

    product_scalar = Q @ K.T
    scaled = (product_scalar/np.sqrt(d_k)) + M
    exp_scores = np.exp(scaled)
    sums = np.sum(exp_scores, axis=-1, keepdims=True)
    attention = exp_scores/sums

    print("Prova Real - Softmax com máscara causal:")
    print(attention)

    return M

create_causal_mask(seq_len=5)

#Tarefa 2
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

W_q = np.random.randn(512, 512)
W_k = np.random.randn(512, 512)
W_v = np.random.randn(512, 512)

cross_attention_output = cross_attention(encoder_output, decoder_state, W_q, W_k, W_v)
print("\nSaída do Cross-Attention:", cross_attention_output.shape)

def generate_next_token(current_sequence, encoder_output):

    probability_vector = np.random.randn(10000)
    exp = np.exp(probability_vector)
    probs = exp/np.sum(exp)

    return probs

current_sequence = ["<start>", "O", "rato"]
EOS_INDEX = 9999

while True:

    token = np.argmax(generate_next_token(current_sequence, encoder_output))
    
    current_sequence.append(str(token))

    if token == EOS_INDEX:
        print("\nFrase final:", current_sequence)
        break