#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${TENSORRT_QWEN_CONFIG:-/workspace/resource/main.yaml}"
if [ -f "${CONFIG_PATH}" ]; then
  eval "$(python3 /workspace/scripts/config_env.py "${CONFIG_PATH}")"
fi

MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_SLUG="$(basename "${MODEL_ID}")"
DATA_DIR="${TENSORRT_QWEN_DATA_DIR:-/workspace/data}"
MODEL_DIR="${TENSORRT_QWEN_MODEL_DIR:-${DATA_DIR}/hf/${MODEL_SLUG}}"
ENGINE_DIR="${TENSORRT_QWEN_ENGINE_DIR:-${DATA_DIR}/engines/${MODEL_SLUG}-int4}"
KV_CACHE_FREE_GPU_MEMORY_FRACTION="${TENSORRT_QWEN_KV_CACHE_FREE_GPU_MEMORY_FRACTION:-0.25}"

if [ ! -d "${ENGINE_DIR}" ]; then
  echo "Engine directory not found: ${ENGINE_DIR}" >&2
  echo "Build it first: docker compose --profile build run --rm builder" >&2
  exit 1
fi

exec trtllm-serve serve "${ENGINE_DIR}" \
  --backend tensorrt \
  --tokenizer "${MODEL_DIR}" \
  --kv_cache_free_gpu_memory_fraction "${KV_CACHE_FREE_GPU_MEMORY_FRACTION}" \
  --host 0.0.0.0 \
  --port 8000
