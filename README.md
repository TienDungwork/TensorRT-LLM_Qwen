# TensorRT Qwen Service

This folder is a standalone Qwen service for `ai_marketing_core`.

It contains:

- `docker-compose.yml`: build/serve TensorRT-LLM and expose an API gateway.
- `resource/main.yaml`: central model/runtime config. Environment variables override it.
- `scripts/build_engine.sh`: downloads Qwen HF weights, converts to TensorRT-LLM checkpoint, builds INT4 engine.
- `scripts/serve_engine.sh`: serves the built engine with `trtllm-serve`.
- `api/main.py`: small FastAPI gateway exposing `/v1/chat/completions` for `ai_marketing_core`.

## Build Engine

```bash
cd TensorRT-LLM_Qwen
docker compose --profile build run --rm builder
```

Default model is set in `resource/main.yaml`:

```yaml
model:
  id: Qwen/Qwen3-4B
```

For a one-off override:

```bash
QWEN_MODEL_ID=Qwen/Qwen2.5-3B-Instruct docker compose --profile build run --rm builder
```

Default engine limits are `max_input_len=3072`, `max_seq_len=4096`, `max_num_tokens=3072`. This is sized for the current `ai_marketing_core` prompt/RAG shape on RTX A4000 16GB.

Runtime VRAM is controlled mainly by TensorRT-LLM KV cache preallocation, not only by engine size. The compose default sets:

```bash
TENSORRT_QWEN_KV_CACHE_FREE_GPU_MEMORY_FRACTION=0.25
```

Increase it for more concurrent/long requests, decrease it if the GPU must share VRAM with other processes.

## Run API

```bash
cd TensorRT-LLM_Qwen
docker compose up api
```

API gateway: `http://localhost:8010/v1`.

Direct TensorRT-LLM server: `http://localhost:8000/v1`.

## Test

```bash
curl http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Viet mot cau ve can dien tu cong nghiep"}],
    "max_tokens": 96,
    "temperature": 0.2
  }'
```

## Connect `ai_marketing_core`

Run the marketing API with:

```bash
AI_MARKETING_MODEL_BACKEND=tensorrt \
TENSORRT_LLM_BASE_URL=http://localhost:8010/v1 \
TENSORRT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
TENSORRT_LLM_MAX_TOKENS=256 \
python run_api.py
```
