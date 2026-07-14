from __future__ import annotations

import os

from pathlib import Path

from tensorrt_qwen.config import ServiceConfig, env_value, load_config


def serve_engine(config: ServiceConfig) -> None:
    backend = env_value("TENSORRT_QWEN_BACKEND", "tensorrt")
    kv_fraction = str(config.tensorrt.kv_cache_free_gpu_memory_fraction)
    common_args = [
        "--kv_cache_free_gpu_memory_fraction",
        kv_fraction,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]

    # Tesla T4 (sm75) is not reliably supported by TensorRT-LLM 1.3
    # (INT4 GEMM missing, TRT engine hang, pytorch Attention/warmup crash).
    # Use HuggingFace transformers as a working fallback on this GPU.
    if backend in {"transformers", "hf", "huggingface"}:
        if not config.model.model_dir.is_dir():
            raise FileNotFoundError(
                f"Model directory not found: {config.model.model_dir}. "
                "Download it first or run the builder profile."
            )
        os.execvp("python3", ["python3", "-m", "tensorrt_qwen.hf_serve"])

    if backend == "pytorch":
        if not config.model.model_dir.is_dir():
            raise FileNotFoundError(
                f"Model directory not found: {config.model.model_dir}. "
                "Download it first or run the builder profile."
            )
        serve_config = Path(
            env_value(
                "TENSORRT_QWEN_SERVE_CONFIG",
                "/workspace/resource/pytorch_serve.yaml",
            )
        )
        pytorch_args = [
            "trtllm-serve",
            "serve",
            str(config.model.model_dir),
            "--backend",
            "pytorch",
            "--tokenizer",
            str(config.model.model_dir),
            *common_args,
        ]
        if serve_config.is_file():
            pytorch_args.extend(["--config", str(serve_config)])
        os.execvp("trtllm-serve", pytorch_args)

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
            *common_args,
        ],
    )


def main() -> int:
    serve_engine(load_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
