# Laboratórios

## Laboratório 1 – Scaled Dot-Product Attention

### Como executar

1. Criar ambiente virtual:

```bash
python -m venv venv
```

2. Ativar:

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

3. Instalar dependências:

```bash
pip install -r requirements.txt
```

4. Executar:

```bash
python test_lab_1.py
```

### Normalização

Após calcular ( QK^T ), o resultado é dividido por ( \sqrt{d_k} ), onde ( d_k ) é a dimensão dos vetores.
Isso evita valores muito altos antes da aplicação do softmax, que é feito linha a linha.

### Exemplo

Entrada:

```python
X = np.array([
    [1, 3, 5],
    [6, 4, 3],
    [1, 1, 7],
    [2, 3, 5]
])
```

Chamada:

```python
scaled_dot_product_attention(X, X, X)
```

Saída esperada: matriz de dimensão `(4, 3)` com valores normalizados.
