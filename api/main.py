from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


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
MAX_INPUT_CHARS = int(os.getenv("ANTHROPIC_COMPAT_MAX_INPUT_CHARS", "10000"))
MAX_OUTPUT_TOKENS = int(os.getenv("ANTHROPIC_COMPAT_MAX_OUTPUT_TOKENS", "768"))
NO_THINK_SUFFIX = "\n/no_think"

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


def _anthropic_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _anthropic_to_openai_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system = payload.get("system")
    if system:
        messages.append({"role": "system", "content": _anthropic_content_to_text(system)})

    for message in payload.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = message.get("role", "user")
        messages.append({
            "role": "assistant" if role == "assistant" else "user",
            "content": _anthropic_content_to_text(message.get("content")),
        })
    return messages


def _clamp_max_tokens(value: Any) -> int:
    try:
        requested = int(value or MAX_OUTPUT_TOKENS)
    except (TypeError, ValueError):
        requested = MAX_OUTPUT_TOKENS
    return max(1, min(requested, MAX_OUTPUT_TOKENS))


def _trim_messages_for_engine(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    remaining = MAX_INPUT_CHARS
    trimmed_reversed: list[dict[str, str]] = []

    # Keep the most recent conversational context first. If the system prompt is
    # huge, keep its beginning and trim the middle pressure off the local engine.
    for message in reversed(messages):
        content = message.get("content", "")
        if remaining <= 0:
            continue

        if len(content) > remaining:
            if message.get("role") == "system":
                content = content[:remaining]
            else:
                content = content[-remaining:]

        trimmed_reversed.append({**message, "content": content})
        remaining -= len(content)

    return list(reversed(trimmed_reversed))


def _strip_think_blocks(text: str) -> str:
    while True:
        start = text.find("<think>")
        if start < 0:
            return text.replace("</think>", "")
        end = text.find("</think>", start)
        if end < 0:
            return text[:start]
        text = text[:start] + text[end + len("</think>"):]


class ThinkBlockFilter:
    def __init__(self) -> None:
        self.in_think = False
        self.pending = ""

    def feed(self, text: str) -> str:
        data = self.pending + text
        self.pending = ""
        output: list[str] = []

        while data:
            if self.in_think:
                end = data.find("</think>")
                if end < 0:
                    keep = min(len(data), len("</think>") - 1)
                    self.pending = data[-keep:] if keep else ""
                    return "".join(output)
                data = data[end + len("</think>"):]
                self.in_think = False
                continue

            start = data.find("<think>")
            if start >= 0:
                output.append(data[:start])
                data = data[start + len("<think>"):]
                self.in_think = True
                continue

            keep = 0
            max_keep = min(len(data), len("<think>") - 1)
            for size in range(max_keep, 0, -1):
                if "<think>".startswith(data[-size:]):
                    keep = size
                    break
            if keep:
                output.append(data[:-keep])
                self.pending = data[-keep:]
            else:
                output.append(data)
            break

        return "".join(output)


def _build_openai_payload(payload: dict[str, Any], *, stream: bool = False) -> dict[str, Any]:
    messages = _trim_messages_for_engine(_anthropic_to_openai_messages(payload))
    for message in reversed(messages):
        if message.get("role") == "user":
            message["content"] = f"{message.get('content', '')}{NO_THINK_SUFFIX}"
            break

    return {
        "model": payload.get("model") or DEFAULT_MODEL,
        "messages": messages,
        "temperature": payload.get("temperature", 0.2),
        "max_tokens": _clamp_max_tokens(payload.get("max_tokens") or payload.get("max_tokens_to_sample")),
        "stream": stream,
    }


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _anthropic_messages_stream(payload: dict[str, Any]) -> AsyncIterator[str]:
    message_id = f"msg_{uuid.uuid4().hex}"
    model = payload.get("model") or DEFAULT_MODEL

    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })
    yield _sse("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })

    upstream_payload = _build_openai_payload(payload, stream=True)

    think_filter = ThinkBlockFilter()
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        async with client.stream("POST", f"{TRTLLM_BASE_URL}/chat/completions", json=upstream_payload) as upstream:
            if upstream.status_code >= 400:
                error_text = await upstream.aread()
                yield _sse("error", {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": error_text.decode("utf-8", errors="replace"),
                    },
                })
                return
            async for line in upstream.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line.removeprefix("data:").strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    delta = think_filter.feed(delta)
                if delta:
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": delta},
                    })

    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 0},
    })
    yield _sse("message_stop", {"type": "message_stop"})


@app.post("/v1/messages")
async def anthropic_messages(request: Request) -> Response:
    payload = await request.json()
    if payload.get("stream"):
        return StreamingResponse(_anthropic_messages_stream(payload), media_type="text/event-stream")

    upstream_payload = _build_openai_payload(payload)
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        upstream = await client.post(f"{TRTLLM_BASE_URL}/chat/completions", json=upstream_payload)
    if upstream.status_code >= 400:
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    data = upstream.json()
    choice = data.get("choices", [{}])[0]
    usage = data.get("usage") or {}
    content = _strip_think_blocks(choice.get("message", {}).get("content", "")).strip()
    return JSONResponse({
        "id": data.get("id") or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": data.get("model") or upstream_payload["model"],
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    })


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
