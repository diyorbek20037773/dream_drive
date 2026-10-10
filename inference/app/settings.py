"""Environment configuration for the inference service (env names: docs/IMPLEMENTATION_SPEC.md section 2)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# inference/app/settings.py -> parents[2] is the repository root (or /app inside the image).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "config" / "models.resolved.yaml"


class Settings(BaseSettings):
    """All values come from environment variables of the same (upper-case) name."""

    model_config = SettingsConfigDict(extra="ignore", protected_namespaces=(), case_sensitive=False)

    model_id: str = "yolopv2"
    # Second detector (docs/IMPLEMENTATION_SPEC.md section 2.1). Both models are required for readiness.
    rfdetr_model_id: str = "rfdetr_nano"
    model_root: Path = Path("/models")
    model_manifest_path: Path = DEFAULT_MANIFEST_PATH

    inference_device: Literal["cpu", "cuda"] = "cpu"
    inference_precision: Literal["fp32", "fp16"] = "fp32"
    # Requests allowed to wait while one forward pass runs; more concurrent requests get 503 inference_busy.
    inference_max_waiters: int = Field(default=4, ge=0, le=64)
    # A waiter that cannot obtain the model lock within this time is answered with 503 inference_busy.
    inference_lock_timeout_s: float = Field(default=30.0, gt=0, le=600)
    inference_max_image_bytes: int = Field(default=8 * 1024 * 1024, ge=1024, le=64 * 1024 * 1024)
    inference_max_image_side: int = Field(default=4096, ge=64, le=8192)
    inference_min_image_side: int = Field(default=16, ge=1, le=64)
    # 0 keeps torch's default intra-op thread count.
    inference_torch_threads: int = Field(default=0, ge=0, le=256)
    inference_warmup_runs: int = Field(default=2, ge=0, le=10)

    model_init_connect_timeout_s: float = Field(default=15.0, gt=0, le=300)
    model_init_read_timeout_s: float = Field(default=60.0, gt=0, le=600)
    model_init_total_timeout_s: float = Field(default=1800.0, gt=0, le=7200)

    log_level: str = "INFO"
    git_commit: str = "unknown"
    build_version: str = "dev"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
