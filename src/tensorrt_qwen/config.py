from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path("/workspace/resource/main.yaml")
DEFAULT_DATA_DIR = Path("/workspace/data")
DEFAULT_MODEL_ID = "Qwen/Qwen3-4B"


def env_value(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def resolve_under(base: Path, value: str | None, fallback: Path) -> Path:
    if not value:
        return fallback

    path = Path(value)
    return path if path.is_absolute() else base / path


@dataclass(frozen=True)
class ModelConfig:
    id: str
    model_dir: Path
    checkpoint_dir: Path
    engine_dir: Path

    @property
    def slug(self) -> str:
        return self.id.rstrip("/").split("/")[-1]


@dataclass(frozen=True)
class TensorRTConfig:
    dtype: str
    weight_only_precision: str
    max_batch_size: int
    max_input_len: int
    max_seq_len: int
    max_num_tokens: int
    load_model_on_cpu: bool
    kv_cache_free_gpu_memory_fraction: float


@dataclass(frozen=True)
class ServiceConfig:
    data_dir: Path
    model: ModelConfig
    tensorrt: TensorRTConfig


def load_config(config_path: Path | None = None) -> ServiceConfig:
    config_path = config_path or Path(env_value("TENSORRT_QWEN_CONFIG", str(DEFAULT_CONFIG_PATH)))
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model_raw = raw.get("model", {})
    trt_raw = raw.get("tensorrt", {})

    data_dir = Path(env_value("TENSORRT_QWEN_DATA_DIR", str(DEFAULT_DATA_DIR)))
    model_id = env_value("QWEN_MODEL_ID", model_raw.get("id", DEFAULT_MODEL_ID))
    model_slug = model_id.rstrip("/").split("/")[-1]
    weight_precision = env_value(
        "TENSORRT_QWEN_WEIGHT_ONLY_PRECISION",
        str(trt_raw.get("weight_only_precision", "int4")),
    )

    model = ModelConfig(
        id=model_id,
        model_dir=resolve_under(
            data_dir,
            env_value("TENSORRT_QWEN_MODEL_DIR", model_raw.get("dir")),
            data_dir / "hf" / model_slug,
        ),
        checkpoint_dir=resolve_under(
            data_dir,
            model_raw.get("checkpoint_dir"),
            data_dir / "checkpoints" / f"{model_slug}-{weight_precision}",
        ),
        engine_dir=resolve_under(
            data_dir,
            env_value("TENSORRT_QWEN_ENGINE_DIR", model_raw.get("engine_dir")),
            data_dir / "engines" / f"{model_slug}-{weight_precision}",
        ),
    )
    tensorrt = TensorRTConfig(
        dtype=env_value("TENSORRT_QWEN_DTYPE", str(trt_raw.get("dtype", "float16"))),
        weight_only_precision=weight_precision,
        max_batch_size=int(
            env_value("TENSORRT_QWEN_MAX_BATCH_SIZE", str(trt_raw.get("max_batch_size", 1)))
        ),
        max_input_len=int(
            env_value("TENSORRT_QWEN_MAX_INPUT_LEN", str(trt_raw.get("max_input_len", 3072)))
        ),
        max_seq_len=int(
            env_value("TENSORRT_QWEN_MAX_SEQ_LEN", str(trt_raw.get("max_seq_len", 4096)))
        ),
        max_num_tokens=int(
            env_value(
                "TENSORRT_QWEN_MAX_NUM_TOKENS",
                str(trt_raw.get("max_num_tokens", 3072)),
            )
        ),
        load_model_on_cpu=str(
            env_value(
                "TENSORRT_QWEN_LOAD_MODEL_ON_CPU",
                str(trt_raw.get("load_model_on_cpu", 1)),
            )
        ).lower()
        in {"1", "true", "yes"},
        kv_cache_free_gpu_memory_fraction=float(
            env_value(
                "TENSORRT_QWEN_KV_CACHE_FREE_GPU_MEMORY_FRACTION",
                str(trt_raw.get("kv_cache_free_gpu_memory_fraction", 0.25)),
            )
        ),
    )

    return ServiceConfig(data_dir=data_dir, model=model, tensorrt=tensorrt)
