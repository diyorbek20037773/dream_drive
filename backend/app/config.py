"""Runtime configuration. Environment variable names are the contract documented in .env.example."""

from __future__ import annotations

import re
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_HEX_COMMIT = re.compile(r"^[a-f0-9]{40,64}$")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore", case_sensitive=False)

    app_env: Literal["development", "test", "production"] = "development"
    app_secret: str = ""
    admin_password: str = ""
    public_origin: str = "http://localhost:8080"
    default_locale: Literal["uz", "en"] = "uz"

    max_upload_bytes: int = Field(default=104_857_600, gt=0)
    max_video_seconds: float = Field(default=120, gt=0)
    max_video_width: int = Field(default=1920, gt=0)
    max_video_height: int = Field(default=1080, gt=0)
    max_queued_per_session: int = Field(default=2, ge=1)
    max_queued_total: int = Field(default=10, ge=1)
    max_requests_per_minute: int = Field(default=10, ge=1)
    max_simulation_wall_seconds: float = Field(default=120, gt=0)
    artifact_retention_hours: float = Field(default=24, gt=0)
    session_ttl_hours: float = Field(default=24, gt=0)

    analysis_fps: float = Field(default=5.0, gt=0, le=30)
    inference_conf_threshold: float = Field(default=0.1, ge=0.01, le=0.9)
    # Display/reporting threshold (DETECTION_REPORT_MIN_SCORE). Inference keeps boxes down to
    # inference_conf_threshold because ByteTrack's second association needs low-score boxes; job summaries,
    # report.json/html and the UI count only detections/tracks with score >= this value as findings and list the
    # rest separately as low-confidence. timeline.json keeps every detection with its score.
    detection_report_min_score: float = Field(default=0.4, ge=0.0, le=1.0)
    inference_base_url: str = "http://inference:8001"
    inference_timeout_seconds: float = Field(default=30, gt=0)
    sim_inference_timeout_seconds: float = Field(default=5, gt=0)

    simulator: Literal["lite", "carla"] = "lite"
    sim_perception_source: Literal["synthetic_rgb_cv", "learned_model"] = "synthetic_rgb_cv"
    sim_fixed_dt_seconds: float = Field(default=0.05, gt=0, le=0.2)
    learned_sim_mode_enabled: bool = False
    carla_host: str = ""

    redis_url: str = "redis://redis:6379/0"
    sqlite_path: Path = Path("/data/dreamdrive.db")
    data_root: Path = Path("/data")
    artifact_root: Path = Path("/data/artifacts")

    log_level: str = "INFO"
    git_commit: str = "unknown"
    build_version: str = "dev"
    build_dirty: bool = False
    worker_metrics_port: int = 9101
    enable_mlflow: bool = False
    mlflow_tracking_uri: str = "http://mlflow:5000"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def commit_is_valid(self) -> bool:
        return bool(_HEX_COMMIT.match(self.git_commit))

    @property
    def tmp_root(self) -> Path:
        return self.data_root / "tmp"

    @property
    def evaluation_root(self) -> Path:
        return self.data_root / "evaluations"

    def effective_secret(self) -> str:
        if self.app_secret:
            return self.app_secret
        if self.is_production:
            raise RuntimeError("APP_SECRET must be set when APP_ENV=production")
        return _ephemeral_secret()


@lru_cache(maxsize=1)
def _ephemeral_secret() -> str:
    return secrets.token_urlsafe(48)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
