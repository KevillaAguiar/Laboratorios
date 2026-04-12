"""
Script de Inferência: Teste do modelo fine-tuned com LoRA
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def carregar_modelo_com_lora():
    print("Carregando config LoRA...")
    peft_config = PeftConfig.from_pretrained("./lora_model")
    
    model_name = peft_config.base_model_name_or_path
    print(f"Modelo base: {model_name}")
    
    print("Carregando modelo...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    
    print("Carregando adaptadores LoRA...")
    model = PeftModel.from_pretrained(model, "./lora_model")
    
    print("Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def fazer_inferencia(model, tokenizer, prompt, max_length=150):
    formatted_prompt = f"### Instrução:\n{prompt}\n\n### Resposta:\n"
    
    print(f"\nPrompt: {prompt}")
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Resposta:" in response:
        response = response.split("### Resposta:")[1].strip()
    
    print(f"Resposta:\n{response}\n")
    
    return response

def main():
    print("=" * 60)
    print("Inferência - Fine-tuning com LoRA")
    print("=" * 60)
    
    model, tokenizer = carregar_modelo_com_lora()
    model.eval()
    
    prompts_teste = [
        "Como fazer um bolo de chocolate delicioso?",
        "Qual é a diferença entre fermento em pó e fermento químico?",
        "Como temperar um prato corretamente?",
        "Qual a melhor forma de cozinhar carne vermelha?",
        "Como fazer uma sopa caseira saudável?",
    ]
    
    print("\nTestando com 5 prompts...\n")
    
    for i, prompt in enumerate(prompts_teste, 1):
        print(f"[{i}/{len(prompts_teste)}]")
        fazer_inferencia(model, tokenizer, prompt)
    
    print("=" * 60)


if __name__ == "__main__":
    main()
    print("="*60)

if __name__ == "__main__":
    main()
