#!/usr/bin/env python3
"""Emit shell exports from the local resource/main.yaml.

This intentionally supports only the small, nested YAML shape used by this
project so containers do not need PyYAML.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path


CONFIG_TO_ENV = {
    ("model", "id"): "QWEN_MODEL_ID",
    ("model", "dir"): "TENSORRT_QWEN_MODEL_DIR",
    ("model", "engine_dir"): "TENSORRT_QWEN_ENGINE_DIR",
    ("tensorrt", "dtype"): "TENSORRT_QWEN_DTYPE",
    ("tensorrt", "weight_only_precision"): "TENSORRT_QWEN_WEIGHT_ONLY_PRECISION",
    ("tensorrt", "max_batch_size"): "TENSORRT_QWEN_MAX_BATCH_SIZE",
    ("tensorrt", "max_input_len"): "TENSORRT_QWEN_MAX_INPUT_LEN",
    ("tensorrt", "max_seq_len"): "TENSORRT_QWEN_MAX_SEQ_LEN",
    ("tensorrt", "max_num_tokens"): "TENSORRT_QWEN_MAX_NUM_TOKENS",
    ("tensorrt", "load_model_on_cpu"): "TENSORRT_QWEN_LOAD_MODEL_ON_CPU",
    (
        "tensorrt",
        "kv_cache_free_gpu_memory_fraction",
    ): "TENSORRT_QWEN_KV_CACHE_FREE_GPU_MEMORY_FRACTION",
    ("api", "timeout_seconds"): "API_TIMEOUT_SECONDS",
}


def _strip_value(value: str) -> str:
    value = value.split(" #", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_flat_config(path: Path) -> dict[tuple[str, str], str]:
    section = ""
    values: dict[tuple[str, str], str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if indent == 0 and not value:
            section = key
            continue
        if indent == 2 and section:
            values[(section, key)] = _strip_value(value)

    return values


def main() -> int:
    config_path = Path(
        sys.argv[1] if len(sys.argv) > 1 else "/workspace/resource/main.yaml"
    )
    for config_key, env_key in CONFIG_TO_ENV.items():
        value = load_flat_config(config_path).get(config_key)
        if value and not os.environ.get(env_key):
            print(f"export {env_key}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
