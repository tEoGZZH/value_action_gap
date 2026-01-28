# hf_backend.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class HFChatModel:
    def __init__(self, model_name_or_path: str):
        hf_token = os.getenv("HF_TOKEN", None)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            token=hf_token,
            use_fast=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            token=hf_token,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.model.eval()

    @torch.inference_mode()
    def chat(self, user_prompt: str, temperature: float = 0.2, max_new_tokens: int = 256) -> str:
        # Llama-3 Instruct: chat template
        messages = [{"role": "user", "content": user_prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        do_sample = temperature > 0

        out = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        gen = out[0, input_ids.shape[-1]:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        return text.strip()
