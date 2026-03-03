import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "aaron-lora-out"
SYSTEM_PROMPT = "You are Aaron. Speak in first person as if you are Aaron (use 'I', 'me', 'my'). Pretend you are Aaron talking to a sde recruiter."
DEFAULT_PROMPT = "Introduce yourself in one line."

tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ),
    device_map=None,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)
if torch.cuda.is_available():
    base = base.to(device)

model = PeftModel.from_pretrained(base, ADAPTER, low_cpu_mem_usage=True)
model = model.to(device)
model.eval()

def generate(user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.05,
        )

    generated_ids = out[0][inputs["input_ids"].shape[-1]:]
    return tok.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:]).strip()
    else:
        user_prompt = input("Prompt: ").strip() or DEFAULT_PROMPT

    print(generate(user_prompt))


if __name__ == "__main__":
    main()