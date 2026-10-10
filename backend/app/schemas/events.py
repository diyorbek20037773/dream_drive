"""SSE envelope (contracts/event.schema.json) with discriminated payloads per event type."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.api import ManeuverLateral, ManeuverLongitudinal, ManeuverPrimary

EventType = Literal["job_state", "job_progress", "run_state", "frame_ready", "decision", "capability_changed", "error"]


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JobStatePayload(_Payload):
    kind: Literal["job_state"] = "job_state"
    job_id: str
    state: str
    error_code: str | None = None


class JobProgressPayload(_Payload):
    kind: Literal["job_progress"] = "job_progress"
    job_id: str
    progress: float = Field(ge=0, le=1)
    stage: str
    frames_done: int = Field(ge=0)
    frames_total: int = Field(ge=0)


class RunStatePayload(_Payload):
    kind: Literal["run_state"] = "run_state"
    run_id: str
    state: str
    outcome: str | None = None


class FrameReadyPayload(_Payload):
    kind: Literal["frame_ready"] = "frame_ready"
    run_id: str
    frame_id: int = Field(ge=0)
    sim_time_s: float = Field(ge=0)


class DecisionCommand(_Payload):
    steer: float
    throttle: float
    brake: float


class DecisionManeuver(_Payload):
    """Compact Run.live.maneuver (app.control.maneuver): what the simulated vehicle is doing."""

    primary: ManeuverPrimary
    longitudinal: ManeuverLongitudinal
    lateral: ManeuverLateral
    since_sim_time_s: float = Field(ge=0, description="Simulated time (s) at which this maneuver began.")


class DecisionPayload(_Payload):
    kind: Literal["decision"] = "decision"
    run_id: str
    frame_id: int = Field(ge=0)
    state: str
    reason_codes: list[str]
    command: DecisionCommand | None = None
    # Additive, optional (older producers omit them).
    sim_time_s: float | None = Field(default=None, ge=0, description="Simulated time of frame_id, s.")
    maneuver: DecisionManeuver | None = None


class CapabilityChangedPayload(_Payload):
    kind: Literal["capability_changed"] = "capability_changed"
    video_analysis_available: bool
    reason: str | None = None


class ErrorPayload(_Payload):
    kind: Literal["error"] = "error"
    code: str
    message: str


EventPayload = Annotated[
    JobStatePayload
    | JobProgressPayload
    | RunStatePayload
    | FrameReadyPayload
    | DecisionPayload
    | CapabilityChangedPayload
    | ErrorPayload,
    Field(discriminator="kind"),
]


class EventEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    event_id: str = Field(min_length=1)
    timestamp_utc: datetime
    resource_id: str = Field(min_length=1)
    type: EventType
    payload: EventPayload
