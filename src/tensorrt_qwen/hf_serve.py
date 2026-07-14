from __future__ import annotations

import time
import uuid
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from tensorrt_qwen.config import load_config


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.2
    stream: bool = False


def build_app() -> FastAPI:
    config = load_config()
    model_id = config.model.id
    model_dir = str(config.model.model_dir)

    print(f"Loading transformers model from {model_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Transformers model ready", flush=True)

    app = FastAPI(title="TensorRT Qwen HF fallback", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "backend": "transformers", "model": model_id}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "owned_by": "local-transformers"}],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> JSONResponse:
        if req.stream:
            return JSONResponse(
                {"error": "stream is not supported by transformers fallback"},
                status_code=400,
            )

        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        max_new = max(1, min(int(req.max_tokens), 1024))
        do_sample = req.temperature is not None and float(req.temperature) > 0

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=do_sample,
                temperature=float(req.temperature) if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        created = int(time.time())
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": created,
                "model": req.model or model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                    "completion_tokens": int(generated.numel()),
                    "total_tokens": int(inputs["input_ids"].shape[-1] + generated.numel()),
                },
            }
        )

    return app


def main() -> int:
    uvicorn.run(build_app(), host="0.0.0.0", port=8000, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
