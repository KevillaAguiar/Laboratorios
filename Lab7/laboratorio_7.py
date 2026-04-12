import json
import random
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY não configurada")

client = OpenAI(api_key=api_key)

def gerar_dataset():
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": """Gere 50 pares pergunta-resposta sobre culinária em JSON.
                Retorne APENAS: [{"prompt": "pergunta", "response": "resposta"}, ...]
                Inclua receitas, técnicas, ingredientes, dicas."""
            }
        ],
        max_tokens=4000
    )
    
    pares = json.loads(response.choices[0].message.content)
    return pares

print("Gerando dataset com GPT-3.5...")
pares = gerar_dataset()
print(f"Total de pares: {len(pares)}")

# Embaralha e divide 90/10
random.shuffle(pares)
split = int(len(pares) * 0.9)
treino = pares[:split]
teste = pares[split:]

# Salvar JSONL
with open("train.jsonl", "w", encoding="utf-8") as f:
    for par in treino:
        f.write(json.dumps(par, ensure_ascii=False) + "\n")

with open("test.jsonl", "w", encoding="utf-8") as f:
    for par in teste:
        f.write(json.dumps(par, ensure_ascii=False) + "\n")

print(f"Treino: {len(treino)} pares → train.jsonl")
print(f"Teste: {len(teste)} pares → test.jsonl")

