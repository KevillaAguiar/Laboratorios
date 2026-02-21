import numpy as np
from laboratorio_1 import scaled_dot_product_attention

X = np.array([
    [0.1, 0.3, 0.5],
    [0.6, 0.4, 0.3],
    [0.1, 0.1, 0.8],
    [0.2, 0.3, 0.5]
])

output = scaled_dot_product_attention(X, X, X)

print(output)