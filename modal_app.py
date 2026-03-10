import modal

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_LOCAL = "aaron-lora-out"
ADAPTER_REMOTE = "/root/aaron-lora-out"
SYSTEM_PROMPT = (
    "You are Aaron. Speak in first person as if you are Aaron "
    "(use 'I', 'me', 'my'). Pretend you are Aaron talking to a sde recruiter."
)

def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(BASE_MODEL)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "fastapi[standard]",
        "pydantic",
        "huggingface_hub",
    )
    .run_function(download_model)
    .add_local_dir(ADAPTER_LOCAL, ADAPTER_REMOTE)
)

app = modal.App("aaron-ai", image=image)


@app.cls(
    gpu="T4",
    timeout=600,
    scaledown_window=300,
    allow_concurrent_inputs=10,
)
class Model:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            device_map="auto",
        )

        self.model = PeftModel.from_pretrained(base, ADAPTER_REMOTE, low_cpu_mem_usage=True)
        self.model.eval()

    @modal.method()
    def generate(self, user_prompt: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.05,
            )

        generated_ids = out[0][inputs["input_ids"].shape[-1]:]
        return self.tok.decode(generated_ids, skip_special_tokens=True).strip()

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel as PydanticModel

        web_app = FastAPI()
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["https://aaron-portfolio.work"],
            allow_methods=["POST"],
            allow_headers=["Content-Type"],
        )

        class Query(PydanticModel):
            question: str

        @web_app.post("/ask")
        def ask(query: Query):
            answer = self.generate.local(query.question)
            return {"answer": answer}

        return web_app


@app.local_entrypoint()
def main(prompt: str = "Introduce yourself in one line."):
    """Run a quick test inference from the command line.

    Usage:
        modal run modal_app.py
        modal run modal_app.py --prompt "What projects have you worked on?"
    """
    model = Model()
    print(model.generate.remote(prompt))
