"""Pydantic mirror of contracts/run-report.schema.json (schema_version 1.0)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.world_state import PerceptionSource

MetricStatus = Literal["measured", "not_evaluated", "unsupported"]


class Metric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float | None
    unit: str = Field(min_length=1)
    sample_count: int = Field(ge=0)
    status: MetricStatus
    note: str

    @model_validator(mode="after")
    def _measured_rules(self) -> Metric:
        if self.status == "measured":
            if self.value is None or self.sample_count < 1:
                raise ValueError("measured metric needs a numeric value and sample_count >= 1")
        else:
            if self.value is not None or not self.note:
                raise ValueError("non-measured metric must have value=null and an explanatory note")
        return self

    @classmethod
    def measured(cls, value: float, unit: str, sample_count: int, note: str = "") -> Metric:
        return cls(value=float(value), unit=unit, sample_count=sample_count, status="measured", note=note)

    @classmethod
    def missing(cls, unit: str, note: str, status: MetricStatus = "not_evaluated", sample_count: int = 0) -> Metric:
        return cls(value=None, unit=unit, sample_count=sample_count, status=status, note=note)


class RunReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    kind: Literal["video_analysis", "simulation", "evaluation"]
    status: Literal["succeeded", "failed", "cancelled", "interrupted", "stopped", "arrived"]
    perception_source: PerceptionSource
    code_commit: str = Field(pattern=r"^[a-f0-9]{40,64}$")
    model_sha256: str | None = Field(pattern=r"^[a-f0-9]{64}$")
    input_manifest_sha256: str | None = Field(pattern=r"^[a-f0-9]{64}$")
    generated_utc: datetime
    metrics: dict[str, Metric]
    limitations: list[str]
    evaluation_truth_used: bool
    runtime_used_dynamic_ground_truth: Literal[False] = False
