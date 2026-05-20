#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${TENSORRT_QWEN_CONFIG:-/workspace/resource/main.yaml}"
if [ -f "${CONFIG_PATH}" ]; then
  eval "$(python3 /workspace/scripts/config_env.py "${CONFIG_PATH}")"
fi

MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_SLUG="$(basename "${MODEL_ID}")"
DATA_DIR="${TENSORRT_QWEN_DATA_DIR:-/workspace/data}"
MODEL_DIR="${DATA_DIR}/hf/${MODEL_SLUG}"
CHECKPOINT_DIR="${DATA_DIR}/checkpoints/${MODEL_SLUG}-int4"
ENGINE_DIR="${DATA_DIR}/engines/${MODEL_SLUG}-int4"
DTYPE="${TENSORRT_QWEN_DTYPE:-float16}"
MAX_BATCH_SIZE="${TENSORRT_QWEN_MAX_BATCH_SIZE:-1}"
MAX_INPUT_LEN="${TENSORRT_QWEN_MAX_INPUT_LEN:-3072}"
MAX_SEQ_LEN="${TENSORRT_QWEN_MAX_SEQ_LEN:-4096}"
MAX_NUM_TOKENS="${TENSORRT_QWEN_MAX_NUM_TOKENS:-3072}"
LOAD_MODEL_ON_CPU="${TENSORRT_QWEN_LOAD_MODEL_ON_CPU:-1}"
WEIGHT_ONLY_PRECISION="${TENSORRT_QWEN_WEIGHT_ONLY_PRECISION:-int4}"

QWEN_EXAMPLE_DIR="${QWEN_EXAMPLE_DIR:-/app/tensorrt_llm/examples/models/core/qwen}"
if [ ! -f "${QWEN_EXAMPLE_DIR}/convert_checkpoint.py" ]; then
  QWEN_EXAMPLE_FILE="$(find /app -path "*/examples/models/core/qwen/convert_checkpoint.py" -print -quit)"
  QWEN_EXAMPLE_DIR="$(dirname "${QWEN_EXAMPLE_FILE}")"
fi
if [ ! -f "${QWEN_EXAMPLE_DIR}/convert_checkpoint.py" ]; then
  echo "Cannot find TensorRT-LLM Qwen convert_checkpoint.py in the container." >&2
  exit 1
fi

mkdir -p "${DATA_DIR}/hf" "${DATA_DIR}/checkpoints" "${DATA_DIR}/engines"
if [ ! -f "${MODEL_DIR}/config.json" ]; then
  if command -v hf >/dev/null 2>&1; then
    hf download "${MODEL_ID}" --local-dir "${MODEL_DIR}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_DIR}"
  else
    python3 -m pip install -q "huggingface_hub>=0.34,<1"
    hf download "${MODEL_ID}" --local-dir "${MODEL_DIR}"
  fi
fi

rm -rf "${CHECKPOINT_DIR}" "${ENGINE_DIR}"
mkdir -p "${CHECKPOINT_DIR}" "${ENGINE_DIR}"

cd "${QWEN_EXAMPLE_DIR}"
CONVERT_ARGS=(
  --model_dir "${MODEL_DIR}" \
  --output_dir "${CHECKPOINT_DIR}" \
  --dtype "${DTYPE}" \
  --use_weight_only \
  --weight_only_precision "${WEIGHT_ONLY_PRECISION}" \
  --workers 1
)
if [ "${LOAD_MODEL_ON_CPU}" = "1" ]; then
  CONVERT_ARGS+=(--load_model_on_cpu)
fi

python3 convert_checkpoint.py "${CONVERT_ARGS[@]}"

trtllm-build \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --output_dir "${ENGINE_DIR}" \
  --gemm_plugin "${DTYPE}" \
  --gpt_attention_plugin "${DTYPE}" \
  --max_batch_size "${MAX_BATCH_SIZE}" \
  --max_input_len "${MAX_INPUT_LEN}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --max_num_tokens "${MAX_NUM_TOKENS}" \
  --workers 1 \
  --monitor_memory

echo "Built TensorRT-LLM engine at ${ENGINE_DIR}"
