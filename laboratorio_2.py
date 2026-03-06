import numpy as np
import pandas as pd 

#passo 1
vocabulario = pd.DataFrame({
    "palavra": ["estou", "com", "muita", "fome"],
    "id": [0, 1, 2, 3]
})

frase = "estou com muita fome"

lista_id = [
    vocabulario[vocabulario["palavra"] == palavra]["id"].values[0]
    for palavra in frase.split()
]

d_model = 3

tb_embeddings = [np.random.rand(len(lista_id), d_model)]

X = tb_embeddings[lista_id]

X = np.expand_dims(X, axis=0)