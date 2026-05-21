from __future__ import annotations

import os

from tensorrt_qwen.config import ServiceConfig, load_config


def serve_engine(config: ServiceConfig) -> None:
    if not config.model.engine_dir.is_dir():
        raise FileNotFoundError(
            f"Engine directory not found: {config.model.engine_dir}. "
            "Build it first: docker compose --profile build run --rm builder"
        )

    os.execvp(
        "trtllm-serve",
        [
            "trtllm-serve",
            "serve",
            str(config.model.engine_dir),
            "--backend",
            "tensorrt",
            "--tokenizer",
            str(config.model.model_dir),
            "--kv_cache_free_gpu_memory_fraction",
            str(config.tensorrt.kv_cache_free_gpu_memory_fraction),
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
    )


def main() -> int:
    serve_engine(load_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
