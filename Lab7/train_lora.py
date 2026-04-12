import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import os

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
)

device_map = "auto" if torch.cuda.is_available() else "cpu"
use_quantization = torch.cuda.is_available()

print(f"Device: {device_map}")
print(f"GPU disponível: {torch.cuda.is_available()}")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Carregando {model_name}...")

if use_quantization:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

print("\nConfig LoRA:")
print(f"  - Rank (r): {lora_config.r}")
print(f"  - Alpha: {lora_config.lora_alpha}")
print(f"  - Dropout: {lora_config.lora_dropout}")

model = get_peft_model(model, lora_config)

print("Carregando dataset...")
dataset = load_dataset("json", data_files={"train": "train.jsonl", "test": "test.jsonl"})

dataset = dataset.rename_column("response", "completion")

def formatting_func(examples):
    output_texts = []
    for i in range(len(examples["prompt"])):
        text = f"### Instrução:\n{examples['prompt'][i]}\n\n### Resposta:\n{examples['completion'][i]}"
        output_texts.append(text)
    return {"text": output_texts}

dataset = dataset.map(formatting_func, batched=True)

training_config = SFTConfig(
    output_dir="./lora_model",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    num_train_epochs=1,
    max_length=256,
    dataset_text_field="text",
    eval_strategy="steps",
    eval_steps=5,
    save_steps=10,
    logging_steps=2,
    push_to_hub=False,
    use_cpu=True,
    seed=42,
)

print("\nConfig de Treinamento:")
print(f"  - Otimizador: {training_config.optim}")
print(f"  - LR Scheduler: {training_config.lr_scheduler_type}")
print(f"  - Learning Rate: {training_config.learning_rate}")

trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

print("\nIniciando treinamento...")
trainer.train()

print("\nSalvando modelo...")
trainer.model.save_pretrained("./lora_model")
tokenizer.save_pretrained("./lora_model")

print("Treinamento concluído! Modelo salvo em ./lora_model")
