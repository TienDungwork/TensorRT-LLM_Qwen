from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response


def _load_config() -> dict[str, Any]:
    config_path = Path(os.getenv("TENSORRT_QWEN_CONFIG", "/workspace/resource/main.yaml"))
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


CONFIG = _load_config()
MODEL_CONFIG = CONFIG.get("model", {})
API_CONFIG = CONFIG.get("api", {})


TRTLLM_BASE_URL = os.getenv("TRTLLM_BASE_URL", "http://server:8000/v1").rstrip("/")
DEFAULT_MODEL = (
    os.getenv("DEFAULT_MODEL")
    or MODEL_CONFIG.get("id")
    or "Qwen/Qwen2.5-7B-Instruct"
)
TIMEOUT_SECONDS = float(
    os.getenv("API_TIMEOUT_SECONDS") or API_CONFIG.get("timeout_seconds") or "180"
)

app = FastAPI(title="TensorRT Qwen API", version="0.1.0")


async def _post_upstream(path: str, payload: dict[str, Any]) -> Response:
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        upstream = await client.post(f"{TRTLLM_BASE_URL}{path}", json=payload)
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            upstream = await client.get(f"{TRTLLM_BASE_URL.removesuffix('/v1')}/health")
        upstream_ok = upstream.status_code < 500
    except httpx.HTTPError:
        upstream_ok = False
    return {"status": "ok", "model": DEFAULT_MODEL, "upstream_ok": upstream_ok}


@app.get("/v1/models")
async def models() -> JSONResponse:
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": DEFAULT_MODEL,
                    "object": "model",
                    "owned_by": "local-tensorrt-qwen",
                }
            ],
        }
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    payload = await request.json()
    payload.setdefault("model", DEFAULT_MODEL)
    payload.setdefault("temperature", 0.2)
    payload.setdefault("max_tokens", 2048)
    return await _post_upstream("/chat/completions", payload)


@app.post("/api/chat")
async def ollama_style_chat(request: Request) -> JSONResponse:
    payload = await request.json()
    messages = payload.get("messages", [])
    response = await _post_upstream(
        "/chat/completions",
        {
            "model": payload.get("model", DEFAULT_MODEL),
            "messages": messages,
            "temperature": payload.get("temperature", 0.2),
            "max_tokens": payload.get("max_tokens", 2048),
        },
    )
    data = response.body
    parsed = httpx.Response(200, content=data).json()
    content = parsed.get("choices", [{}])[0].get("message", {}).get("content", "")
    return JSONResponse({"message": {"role": "assistant", "content": content}})
