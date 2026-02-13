# hf_backend.py
import os
import torch
from typing import Optional
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
from enum import Enum


EVAL_JSON_SCHEMA = {
    "name": "task2_output",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["Option 1", "Option 2"],
            },
            "reason": {
                "type": "string",
            },
        },
        "required": ["action", "reason"],
        "additionalProperties": False,
    },
}


class Options(str, Enum):
    option1 = "Option 1"
    option2 = "Option 2"

class Task2Output(BaseModel):
    action: Options
    reason: str


class HFChatModel:
    def __init__(self, model_name_or_path: str):
        # If VLLM_BASE_URL is set, use vLLM OpenAI-compatible server.
        self.vllm_base_url: Optional[str] = os.getenv("VLLM_BASE_URL")
        self.vllm_model: str = os.getenv("VLLM_MODEL", model_name_or_path)
        self.vllm_timeout_s: int = int(os.getenv("VLLM_TIMEOUT_S", "300"))
        self.use_vllm: bool = self.vllm_base_url is not None
        
        # Only load local model/tokenizer if we are NOT using vLLM.
        print(f"Using vLLM: {self.use_vllm}")
        if self.use_vllm:
            self.tokenizer = None
            self.model = None
            return
        

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


    def _chat_vllm(self, user_prompt: str, temperature: float, max_new_tokens: int) -> str:
        """Call a vLLM OpenAI-compatible server and return assistant message text."""

        base = (self.vllm_base_url or "").rstrip("/")
        url = f"{base}/v1/chat/completions"

        payload = {
            "model": self.vllm_model,
            "messages": [{"role": "user", "content": user_prompt}],
            # OpenAI naming: max_tokens
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": 0.9,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "task2_output",
                    "strict": True,
                    "schema": Task2Output.model_json_schema(),
                }
            },
            "max_tokens": 512,
        }

        resp = requests.post(url, json=payload, timeout=self.vllm_timeout_s)
        # vLLM returns OpenAI-style error JSON; include it in exception for debugging.
        if resp.status_code >= 400:
            raise RuntimeError(f"vLLM request failed ({resp.status_code}): {resp.text}")

        data = resp.json()

        # Info for debugging: print the full response and the content we will return.
        choice0 = data["choices"][0]
        print("finish_reason:", choice0.get("finish_reason"))
        print("content repr:", repr(choice0["message"]["content"]))

        return data["choices"][0]["message"]["content"].strip()
    

    @torch.inference_mode()
    def chat(self, user_prompt: str, temperature: float = 0.2, max_new_tokens: int = 1024) -> str:
        if self.use_vllm:
            return self._chat_vllm(user_prompt, temperature, max_new_tokens)
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
