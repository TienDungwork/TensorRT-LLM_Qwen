#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tensorrt_qwen.config import DEFAULT_CONFIG_PATH, ServiceConfig, env_value, load_config


DEFAULT_QWEN_EXAMPLE_DIR = Path("/app/tensorrt_llm/examples/models/core/qwen")
UNSUPPORTED_LEGACY_MODEL_TYPES = {"qwen3_5"}


def run_command(args: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def find_qwen_example_dir() -> Path:
    configured = Path(env_value("QWEN_EXAMPLE_DIR", str(DEFAULT_QWEN_EXAMPLE_DIR)))
    if (configured / "convert_checkpoint.py").is_file():
        return configured

    for root, _, files in os.walk("/app"):
        if "convert_checkpoint.py" in files and root.endswith("examples/models/core/qwen"):
            return Path(root)

    raise FileNotFoundError("Cannot find TensorRT-LLM Qwen convert_checkpoint.py")


def is_lfs_pointer(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size > 1024:
        return False

    try:
        return path.read_text(encoding="utf-8").startswith(
            "version https://git-lfs.github.com/spec/v1"
        )
    except UnicodeDecodeError:
        return False


def needs_model_download(model_dir: Path) -> bool:
    if not (model_dir / "config.json").is_file():
        return True
    return any(is_lfs_pointer(path) for path in model_dir.glob("*.safetensors"))


def download_model(config: ServiceConfig) -> None:
    config.model.model_dir.parent.mkdir(parents=True, exist_ok=True)
    if not needs_model_download(config.model.model_dir):
        return

    if shutil.which("hf"):
        run_command(["hf", "download", config.model.id, "--local-dir", str(config.model.model_dir)])
        return
    if shutil.which("huggingface-cli"):
        run_command(
            [
                "huggingface-cli",
                "download",
                config.model.id,
                "--local-dir",
                str(config.model.model_dir),
            ]
        )
        return

    run_command(["python3", "-m", "pip", "install", "-q", "huggingface_hub>=0.34,<1"])
    run_command(["hf", "download", config.model.id, "--local-dir", str(config.model.model_dir)])


def validate_legacy_int4_support(config: ServiceConfig) -> None:
    config_path = config.model.model_dir / "config.json"
    if not config_path.is_file():
        return

    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = str(model_config.get("model_type", ""))
    if model_type in UNSUPPORTED_LEGACY_MODEL_TYPES:
        raise SystemExit(
            f"{config.model.id} has model_type={model_type!r}, which is not supported by "
            "the legacy TensorRT-LLM INT4 engine builder in this image. Use the "
            "PyTorch backend with a TensorRT-LLM image that supports Qwen3.5, or keep "
            "using Qwen/Qwen3-4B for this INT4 engine workflow."
        )


def use_weight_only(precision: str) -> bool:
    return precision.lower() in {"int4", "int8"}


def build_engine_int4(config: ServiceConfig) -> None:
    precision = config.tensorrt.weight_only_precision.lower()
    if precision not in {"int4", "int8", "none", "fp16", "float16"}:
        raise ValueError(
            "Unsupported weight_only_precision. Use int4, int8, or none/fp16. "
            f"Current value: {config.tensorrt.weight_only_precision}"
        )

    qwen_example_dir = find_qwen_example_dir()
    download_model(config)
    if precision == "int4":
        validate_legacy_int4_support(config)

    config.model.checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    config.model.engine_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(config.model.checkpoint_dir, ignore_errors=True)
    shutil.rmtree(config.model.engine_dir, ignore_errors=True)
    config.model.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.model.engine_dir.mkdir(parents=True, exist_ok=True)

    convert_args = [
        "python3",
        "convert_checkpoint.py",
        "--model_dir",
        str(config.model.model_dir),
        "--output_dir",
        str(config.model.checkpoint_dir),
        "--dtype",
        config.tensorrt.dtype,
        "--workers",
        "1",
    ]
    # INT4 weight-only is unsupported on Turing (T4/sm75). Prefer none/fp16 there.
    if use_weight_only(precision):
        convert_args.extend(
            ["--use_weight_only", "--weight_only_precision", precision]
        )
    if config.tensorrt.load_model_on_cpu:
        convert_args.append("--load_model_on_cpu")
    run_command(convert_args, cwd=qwen_example_dir)

    build_args = [
        "trtllm-build",
        "--checkpoint_dir",
        str(config.model.checkpoint_dir),
        "--output_dir",
        str(config.model.engine_dir),
        "--gemm_plugin",
        config.tensorrt.dtype,
        "--gpt_attention_plugin",
        config.tensorrt.dtype,
        "--max_batch_size",
        str(config.tensorrt.max_batch_size),
        "--max_input_len",
        str(config.tensorrt.max_input_len),
        "--max_seq_len",
        str(config.tensorrt.max_seq_len),
        "--max_num_tokens",
        str(config.tensorrt.max_num_tokens),
        "--workers",
        "1",
        "--monitor_memory",
    ]
    # Turing (T4, sm75) and older GPUs do not support fused context FMHA.
    if env_value("TENSORRT_QWEN_CONTEXT_FMHA", "auto") == "disable":
        build_args.extend(["--context_fmha", "disable"])
    run_command(build_args)
    label = precision if use_weight_only(precision) else "fp16"
    print(f"Built TensorRT-LLM {label} engine at {config.model.engine_dir}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the configured Qwen INT4 engine.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(env_value("TENSORRT_QWEN_CONFIG", str(DEFAULT_CONFIG_PATH))),
    )
    args = parser.parse_args()

    build_engine_int4(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
