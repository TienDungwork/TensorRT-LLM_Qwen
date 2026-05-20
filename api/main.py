from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response


def _read_config_value(section: str, key: str) -> str | None:
    config_path = Path(os.getenv("TENSORRT_QWEN_CONFIG", "/workspace/resource/main.yaml"))
    if not config_path.exists():
        return None

    current_section = ""
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue

        name, value = line.split(":", 1)
        name = name.strip()
        value = value.split(" #", 1)[0].strip()
        if indent == 0 and not value:
            current_section = name
            continue
        if indent == 2 and current_section == section and name == key:
            return value.strip("'\"") or None
    return None


TRTLLM_BASE_URL = os.getenv("TRTLLM_BASE_URL", "http://server:8000/v1").rstrip("/")
DEFAULT_MODEL = (
    os.getenv("DEFAULT_MODEL")
    or _read_config_value("model", "id")
    or "Qwen/Qwen2.5-7B-Instruct"
)
TIMEOUT_SECONDS = float(
    os.getenv("API_TIMEOUT_SECONDS") or _read_config_value("api", "timeout_seconds") or "180"
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
