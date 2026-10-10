"""Pydantic mirror of contracts/world-state.schema.json (schema_version 1.0).

Dynamic simulator ground truth is intentionally absent: extra fields are forbidden.
`tests/unit/backend/test_contracts.py` validates dumps against the JSON Schema.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PerceptionSource = Literal["learned_model", "synthetic_rgb_cv", "classical_lane_fallback", "unavailable"]
QualityStatus = Literal["valid", "degraded", "missing", "invalid", "stale"]
LaneStatus = Literal["valid", "uncertain", "unsupported", "unavailable"]
DecisionState = Literal[
    "observe", "follow_route", "slow", "controlled_stop", "safety_stop", "paused", "arrived", "failed"
]
ActionKind = Literal["analytical_only", "simulator_control"]
Mode = Literal["video_analysis", "simulation"]

_SHA256 = r"^[a-f0-9]{64}$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FrameInfo(_Strict):
    id: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    clock: Literal["video_pts", "simulation_time"]
    acquired_utc: datetime
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    quality_status: QualityStatus


class Detection(_Strict):
    track_id: int | None = Field(default=None, ge=0)
    class_id: int = Field(ge=0)
    class_name: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)
    xyxy_pixels: list[float] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def _ordered(self) -> Detection:
        x1, y1, x2, y2 = self.xyxy_pixels
        if min(self.xyxy_pixels) < 0 or x2 < x1 or y2 < y1:
            raise ValueError("xyxy_pixels must be non-negative and ordered x1<=x2, y1<=y2")
        return self


class Perception(_Strict):
    source: PerceptionSource
    model_id: str | None
    model_sha256: str | None = Field(pattern=_SHA256)
    detections: list[Detection]
    lane_mask_ref: str | None
    drivable_mask_ref: str | None
    lane_status: LaneStatus
    image_lane_offset_normalized: float | None = Field(ge=-1, le=1)
    image_obstacle_cue: float | None = Field(ge=0, le=1)
    ttc_seconds: float | None = Field(gt=0)
    uncertainty_reasons: list[str]

    @model_validator(mode="after")
    def _learned_identity(self) -> Perception:
        if self.source == "learned_model" and (not self.model_id or not self.model_sha256):
            raise ValueError("learned_model perception requires model_id and model_sha256")
        return self


class EgoTelemetry(_Strict):
    source: Literal["ego_telemetry"] = "ego_telemetry"
    x_m: float
    y_m: float
    yaw_rad: float
    speed_mps: float = Field(ge=0)


class Decision(_Strict):
    state: DecisionState
    reason_codes: list[str] = Field(min_length=1)
    action_kind: ActionKind


class Command(_Strict):
    steer: float = Field(ge=-1, le=1)
    throttle: float = Field(ge=0, le=1)
    brake: float = Field(ge=0, le=1)
    input_frame_id: int = Field(ge=0)
    source: Literal["dreamdrive_controller"] = "dreamdrive_controller"
    safety_override: bool
    reason_codes: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _exclusive(self) -> Command:
        if self.throttle > 0 and self.brake > 0:
            raise ValueError("throttle and brake must not be applied together")
        return self


class WorldState(_Strict):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    mode: Mode
    frame: FrameInfo
    perception: Perception
    ego: EgoTelemetry | None
    decision: Decision
    command: Command | None

    @model_validator(mode="after")
    def _mode_rules(self) -> WorldState:
        if self.mode == "video_analysis":
            if self.ego is not None or self.command is not None:
                raise ValueError("video_analysis has no ego localization or actuator command")
            if self.decision.action_kind != "analytical_only":
                raise ValueError("video_analysis decisions are analytical_only")
        return self
