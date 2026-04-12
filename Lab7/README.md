# Laboratório 7 — Fine-Tuning com LoRA e QLoRA

Neste laboratório foi construído um pipeline completo de **fine-tuning** de um modelo de linguagem fundacional utilizando técnicas de eficiência de parâmetros (**PEFT/LoRA**) e quantização (**QLoRA**) para viabilizar o treinamento em hardwares limitados.

## Componentes Implementados

- **Engenharia de Dados Sintéticos** — geração de 50 pares instrução-resposta no domínio de culinária via API GPT-3.5-turbo, divididos em 90% treino e 10% teste, salvos em `.jsonl`
- **Quantização QLoRA** — carregamento do modelo base em 4-bits com tipo `nf4` (NormalFloat) e compute dtype `float16`
- **Configuração LoRA** — rank=64, alpha=16, dropout=0.1, task_type=CAUSAL_LM
- **Otimizador Paginado** — `paged_adamw_32bit` para transferência de picos de memória da GPU para a CPU
- **Learning Rate Scheduler** — tipo `cosine` com warmup ratio de 0.03
- **Pipeline de Treinamento** — `SFTTrainer` da biblioteca `trl`
- **Salvamento do Adaptador** — modelo fine-tuned salvo em `./lora_model/`

## Configurações de Fine-Tuning

### Quantização (QLoRA)
- Precisão: **4-bits**
- Tipo: **nf4** (NormalFloat 4-bit)
- Compute Dtype: **float16**

### Adaptador LoRA
- Rank (r): **64**
- Alpha: **16**
- Dropout: **0.1**
- Task Type: **CAUSAL_LM**
- Target Modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`

### Otimizador e Scheduler
- Otimizador: **paged_adamw_32bit**
- Learning Rate: **2e-4**
- Scheduler: **cosine**
- Warmup Ratio: **0.03**

## Estrutura do Projeto

```
Lab7/
├── laboratorio_7.py     # Geração do dataset sintético via API
├── train_lora.py        # Pipeline de fine-tuning com LoRA/QLoRA
├── inference.py         # Script de teste e inferência
├── train.jsonl          # Dataset de treino (45 pares)
├── test.jsonl           # Dataset de teste (5 pares)
├── lora_model/          # Adaptadores LoRA treinados
└── README.md
```

## Como Executar

```bash
pip install torch transformers peft bitsandbytes trl openai datasets accelerate
```

Configure sua chave da API num arquivo `.env`:
```
OPENAI_API_KEY=sk-sua_chave_aqui
```

Em seguida rode os scripts na ordem:
```bash
python laboratorio_7.py   # gera o dataset
python train_lora.py      # treina o modelo
python inference.py       # testa o modelo (opcional)
```

---

## Integridade Acadêmica

O desenvolvimento deste laboratório contou com o auxílio da ferramenta de IA generativa **Claude (Anthropic)** nas seguintes frentes:

* Estruturação do pipeline e recomendações de bibliotecas
* Geração de templates de código para as configurações de LoRA, BitsAndBytesConfig e SFTTrainer
* Redação dos arquivos README

Todas as configurações obrigatórias (rank, alpha, dropout, otimizador, scheduler) foram validadas e o código foi revisado pela aluna.

*Partes geradas/complementadas com IA (Claude - Anthropic), revisadas por Kévilla.*