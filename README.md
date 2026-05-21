# TensorRT Qwen Service

This folder is a standalone Qwen service for `services.rag_service`.

It contains:

- `docker-compose.yml`: build/serve TensorRT-LLM and expose an API gateway.
- `resource/main.yaml`: central model/runtime config. Environment variables override it.
- `scripts/build_engine_int4`: one-off INT4 engine build script.
- `src/tensorrt_qwen/`: runtime code for loading and serving the TensorRT engine.
- `api/main.py`: small FastAPI gateway exposing `/v1/chat/completions` for `services.rag_service`.

## Build Engine

```bash
cd services/model_service
docker compose --profile build run --rm builder
```

Default model is set in `resource/main.yaml`:

```yaml
model:
  id: Qwen/Qwen3-4B
  engine_dir: engines/Qwen3-4B-int4
```

For a one-off override:

```bash
QWEN_MODEL_ID=Qwen/Qwen2.5-3B-Instruct docker compose --profile build run --rm builder
```

Default engine limits are `max_input_len=3072`, `max_seq_len=4096`, `max_num_tokens=3072`. This is sized for the current `services.rag_service` prompt/RAG shape on RTX A4000 16GB.

Runtime VRAM is controlled mainly by TensorRT-LLM KV cache preallocation, not only by engine size. The compose default sets:

```yaml
tensorrt:
  kv_cache_free_gpu_memory_fraction: 0.25
```

Increase it for more concurrent/long requests, decrease it if the GPU must share VRAM with other processes.

## Run API

```bash
cd services/model_service
docker compose up -d
```

API gateway: `http://localhost:8010/v1`.

Direct TensorRT-LLM server: `http://localhost:8000/v1`.

Public deploy qua nginx tập trung của project: `/home/atin/ntiendung/projects/ai-business/nginx/conf.d/default.conf` -> `/qwen/v1`.

## Test

```bash
curl http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B",
    "messages": [{"role": "user", "content": "Viet mot cau ve can dien tu cong nghiep"}],
    "max_tokens": 96,
    "temperature": 0.2
  }'
```

## Connect `services.rag_service`

Run the marketing API with:

```bash
AI_MARKETING_MODEL_BACKEND=tensorrt \
TENSORRT_LLM_BASE_URL=http://localhost:8010/v1 \
TENSORRT_LLM_MODEL=Qwen/Qwen3-4B \
TENSORRT_LLM_MAX_TOKENS=256 \
python run_api.py
```
