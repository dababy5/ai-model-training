import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_DIR = "aaron-lora-out"
MAX_NEW_TOKENS = 220

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()
    return tokenizer, model

def generate(tokenizer, model, user_text: str):
    # Use Qwen chat template (same style you trained on)
    messages = [
        {"role": "system", "content": "You are Aaron's personalized assistant. Reply in Aaron's style: concise, technical, slightly casual."},
        {"role": "user", "content": user_text},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # IMPORTANT for inference
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.05,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text

def main():
    tokenizer, model = load_model()

    print("\nLoaded base + LoRA. Type a message (or 'exit').\n")
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break

        out = generate(tokenizer, model, user_text)

        # Optional: print only the last assistant part (simple heuristic)
        # If you want the raw full text, just print(out)
        print("\nModel:\n" + out + "\n")

if __name__ == "__main__":
    main()
