"""REST request/response models (docs/IMPLEMENTATION_SPEC.md §3.4). The OpenAPI schema is generated from these.

Conventions:
- Request models forbid unknown fields; response models ignore extras so internal dicts can be validated directly.
- Response models mark defaulted fields as required in the serialization schema, so generated TypeScript types
  are non-optional for fields the server always sends.
- Timestamps are timezone-aware datetimes normalised to UTC; units are explicit in names (_s, _m, _mps, _utc, _ms).
- Named enums use ``TypeAliasType`` so they appear as reusable components (e.g. ``JobState``) in the OpenAPI.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any, Literal, get_args

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, field_validator
from typing_extensions import TypeAliasType

import app.schemas.world_state as ws
from app.schemas.inference import ModelIdentity, ModelInfo
from app.schemas.reason_codes import ReasonCode
from app.schemas.report import Metric, RunReport

if TYPE_CHECKING:
    from app.config import Settings
    from app.simulators.base import RoadGraph

# ---------------------------------------------------------------------------------------------------------------------
# Shared constants (engineering defaults from docs/PRODUCT.md and the spec; not measured capacity)
# ---------------------------------------------------------------------------------------------------------------------
RESOURCE_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
SHA256_PATTERN = r"^[a-f0-9]{64}$"
COMMIT_PATTERN = r"^[a-f0-9]{40,64}$"
ACCEPTED_VIDEO_EXTENSIONS: tuple[str, ...] = ("mp4", "mov")
MAX_OBSTACLES_PER_RUN = 5
MAX_ACTIVE_SIMULATIONS = 1
MAX_EVALUATION_SEEDS = 20
SEED_MAX = 1_000_000
UNSUPPORTED_CLASS_EXAMPLES: tuple[str, ...] = ("person", "traffic_light", "traffic_sign")
ROAD_GRAPH_FRAME = "ENU x-east y-north yaw CCW"


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]

# ---------------------------------------------------------------------------------------------------------------------
# Named enums (exposed as OpenAPI components)
# ---------------------------------------------------------------------------------------------------------------------
JobState = TypeAliasType(
    "JobState",
    Literal["queued", "validating", "running", "cancelling", "cancelled", "succeeded", "failed", "interrupted"],
)
RunState = TypeAliasType(
    "RunState", Literal["initializing", "running", "paused", "stopping", "stopped", "arrived", "failed"]
)
EvaluationState = TypeAliasType("EvaluationState", Literal["queued", "running", "succeeded", "failed", "interrupted"])
EvaluationKind = TypeAliasType(
    "EvaluationKind", Literal["synthetic_scenario_suite", "model_benchmark", "video_ml_accuracy"]
)
SessionRole = TypeAliasType("SessionRole", Literal["anonymous", "operator"])
ArtifactName = TypeAliasType(
    "ArtifactName", Literal["input", "annotated.mp4", "timeline.json", "report.json", "report.html", "poster.jpg"]
)
SimulatorId = TypeAliasType("SimulatorId", Literal["lite", "carla"])
SimPerceptionSource = TypeAliasType("SimPerceptionSource", Literal["synthetic_rgb_cv", "learned_model"])
ScenarioId = TypeAliasType(
    "ScenarioId", Literal["normal", "bend", "obstacle_stop", "camera_blackout", "stale_frame", "pause_resume"]
)
RunAction = TypeAliasType("RunAction", Literal["pause", "resume", "stop"])
# Re-exported world-state literals so the OpenAPI names them.
PerceptionSource = TypeAliasType("PerceptionSource", ws.PerceptionSource)
DecisionState = TypeAliasType("DecisionState", ws.DecisionState)
ActionKind = TypeAliasType("ActionKind", ws.ActionKind)
QualityStatus = TypeAliasType("QualityStatus", ws.QualityStatus)
LaneStatus = TypeAliasType("LaneStatus", ws.LaneStatus)
# Simulation explainability (app.control.maneuver / app.control.basis; a unit test keeps these in sync).
ManeuverPrimary = TypeAliasType(
    "ManeuverPrimary",
    Literal["STOP", "BRAKE", "SLOW_DOWN", "ACCELERATE", "CRUISE", "TURN_LEFT", "TURN_RIGHT", "HOLD"],
)
ManeuverLongitudinal = TypeAliasType(
    "ManeuverLongitudinal",
    Literal["accelerating", "cruising", "decelerating", "braking_to_stop", "stopped", "holding"],
)
ManeuverLateral = TypeAliasType("ManeuverLateral", Literal["straight", "turning_left", "turning_right"])
ManeuverSignalSource = TypeAliasType("ManeuverSignalSource", Literal["ego_telemetry_window", "command_model", "none"])
DecisionBasisSource = TypeAliasType(
    "DecisionBasisSource",
    Literal[
        "synthetic_rgb_cv",
        "learned_model",
        "ego_telemetry",
        "route_graph",
        "operator",
        "safety_supervisor",
        "controller",
    ],
)
DecisionBasisEffect = TypeAliasType(
    "DecisionBasisEffect",
    Literal["slow", "controlled_stop", "safety_stop", "steer_correction", "speed_limit", "pause", "arrive", "fail"],
)
DecisionBasisComparator = TypeAliasType("DecisionBasisComparator", Literal[">=", ">", "<=", "<", "==", "!="])

# Plain tuples of the enum values, for runtime checks (derived from the aliases above, never duplicated).
JOB_STATES: tuple[str, ...] = get_args(JobState.__value__)
JOB_TERMINAL_STATES: frozenset[str] = frozenset({"cancelled", "succeeded", "failed", "interrupted"})
RUN_STATES: tuple[str, ...] = get_args(RunState.__value__)
RUN_TERMINAL_STATES: frozenset[str] = frozenset({"stopped", "arrived", "failed"})
ARTIFACT_NAMES: tuple[str, ...] = get_args(ArtifactName.__value__)
SCENARIO_IDS: tuple[str, ...] = get_args(ScenarioId.__value__)


class ErrorCode(StrEnum):
    """Machine-readable error codes used in ``ErrorEnvelope.code``. The frontend maps them to localized text."""

    # Generic
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"
    METHOD_NOT_ALLOWED = "method_not_allowed"
    HTTP_ERROR = "http_error"
    INTERNAL_ERROR = "internal_error"
    NOT_IMPLEMENTED = "not_implemented"
    SERVICE_UNAVAILABLE = "service_unavailable"
    # Session / authorization
    SESSION_REQUIRED = "session_required"
    CSRF_FAILED = "csrf_failed"
    OPERATOR_REQUIRED = "operator_required"
    INVALID_CREDENTIALS = "invalid_credentials"
    OPERATOR_LOGIN_DISABLED = "operator_login_disabled"
    # Quotas / admission
    RATE_LIMITED = "rate_limited"
    SESSION_QUEUE_FULL = "session_queue_full"
    QUEUE_FULL = "queue_full"
    SIMULATION_BUSY = "simulation_busy"
    EVALUATION_BUSY = "evaluation_busy"
    # Upload / video validation
    UPLOAD_TOO_LARGE = "upload_too_large"
    UPLOAD_EMPTY = "upload_empty"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    VIDEO_INVALID = "video_invalid"
    VIDEO_TOO_LONG = "video_too_long"
    VIDEO_TOO_LARGE_DIMENSIONS = "video_too_large_dimensions"
    VIDEO_NO_STREAM = "video_no_stream"
    VIDEO_DECODE_FAILED = "video_decode_failed"
    # Model / inference
    MODEL_NOT_READY = "model_not_ready"
    INFERENCE_UNAVAILABLE = "inference_unavailable"
    # Jobs
    RESULTS_UNAVAILABLE = "results_unavailable"
    ARTIFACT_UNAVAILABLE = "artifact_unavailable"
    RANGE_NOT_SATISFIABLE = "range_not_satisfiable"
    # Simulation runs
    SIMULATOR_UNAVAILABLE = "simulator_unavailable"
    UNSUPPORTED_MODE = "unsupported_mode"
    UNSUPPORTED_SCENARIO = "unsupported_scenario"
    UNKNOWN_NODE = "unknown_node"
    DESTINATION_UNREACHABLE = "destination_unreachable"
    INVALID_TRANSITION = "invalid_transition"
    RUN_NOT_ACTIVE = "run_not_active"
    OBSTACLE_LIMIT_REACHED = "obstacle_limit_reached"
    REPORT_UNAVAILABLE = "report_unavailable"
    # Evaluations
    EVALUATION_UNAVAILABLE = "evaluation_unavailable"


# ---------------------------------------------------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------------------------------------------------
class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResponseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", json_schema_serialization_defaults_required=True)


# ---------------------------------------------------------------------------------------------------------------------
# Errors and health
# ---------------------------------------------------------------------------------------------------------------------
class ErrorEnvelope(ResponseModel):
    """Every non-2xx JSON response. Never contains filesystem paths or stack traces."""

    code: ErrorCode
    message: str = Field(description="Human-readable English message; the UI localizes by `code`.")
    request_id: str = Field(description="Correlates with the X-Request-ID response header and server logs.")
    details: dict[str, Any] = Field(default_factory=dict, description="Structured, code-specific context.")


class HealthCheck(ResponseModel):
    ok: bool
    reason: str | None = None


class HealthStatus(ResponseModel):
    status: Literal["ok", "unavailable"]
    checks: dict[str, HealthCheck] = Field(
        default_factory=dict, description="Readiness dependencies (database, queue). Model readiness is separate."
    )
    checked_utc: UtcDatetime


# ---------------------------------------------------------------------------------------------------------------------
# Session and limits
# ---------------------------------------------------------------------------------------------------------------------
class Limits(ResponseModel):
    """Configured engineering limits (not measured server capacity)."""

    max_upload_bytes: int = Field(gt=0)
    max_video_duration_s: float = Field(gt=0)
    max_video_width_px: int = Field(gt=0)
    max_video_height_px: int = Field(gt=0)
    accepted_extensions: list[str]
    max_queued_per_session: int = Field(ge=1)
    max_queued_total: int = Field(ge=1)
    max_expensive_requests_per_minute: int = Field(ge=1)
    max_simulation_wall_s: float = Field(gt=0)
    max_active_simulations: int = Field(ge=1)
    max_obstacles_per_run: int = Field(ge=0)
    artifact_retention_hours: float = Field(gt=0)
    analysis_fps: float = Field(gt=0)

    @classmethod
    def from_settings(cls, settings: Settings) -> Limits:
        return cls(
            max_upload_bytes=settings.max_upload_bytes,
            max_video_duration_s=settings.max_video_seconds,
            max_video_width_px=settings.max_video_width,
            max_video_height_px=settings.max_video_height,
            accepted_extensions=list(ACCEPTED_VIDEO_EXTENSIONS),
            max_queued_per_session=settings.max_queued_per_session,
            max_queued_total=settings.max_queued_total,
            max_expensive_requests_per_minute=settings.max_requests_per_minute,
            max_simulation_wall_s=settings.max_simulation_wall_seconds,
            max_active_simulations=MAX_ACTIVE_SIMULATIONS,
            max_obstacles_per_run=MAX_OBSTACLES_PER_RUN,
            artifact_retention_hours=settings.artifact_retention_hours,
            analysis_fps=settings.analysis_fps,
        )


class SessionCreate(RequestModel):
    admin_password: str | None = Field(
        default=None, min_length=1, max_length=256, description="Operator login; omit for an anonymous session."
    )


class SessionInfo(ResponseModel):
    session_id_suffix: str = Field(description="Last characters of the session id, for display/support only.")
    role: SessionRole
    csrf_token: str = Field(description="Send as X-CSRF-Token on every POST/DELETE except POST /session.")
    expires_utc: UtcDatetime
    limits: Limits


# ---------------------------------------------------------------------------------------------------------------------
# Video jobs
# ---------------------------------------------------------------------------------------------------------------------
class JobInput(ResponseModel):
    sha256: str = Field(pattern=SHA256_PATTERN, description="SHA256 of the uploaded bytes.")
    bytes: int = Field(ge=0)
    ext: Literal["mp4", "mov"] = Field(description="Accepted container extension, without a dot.")


class JobVideo(ResponseModel):
    """ffprobe-derived properties of the uploaded video."""

    duration_s: float = Field(ge=0)
    width: int = Field(ge=1, description="Display width in pixels after rotation.")
    height: int = Field(ge=1, description="Display height in pixels after rotation.")
    fps_avg: float | None = Field(default=None, ge=0, description="Average frame rate reported by the container.")
    rotation: int = Field(default=0, description="Rotation in degrees applied for display.")
    codec: str


class Job(ResponseModel):
    id: str
    kind: Literal["video_analysis"] = "video_analysis"
    state: JobState
    stage: str | None = Field(default=None, description="Current pipeline stage name, e.g. probe/decode/encode.")
    progress: float = Field(ge=0, le=1)
    frames_done: int = Field(ge=0)
    frames_total: int | None = Field(default=None, ge=0, description="Planned analyzed frames, once known.")
    error_code: str | None = None
    error_message: str | None = None
    created_utc: UtcDatetime
    started_utc: UtcDatetime | None = None
    finished_utc: UtcDatetime | None = None
    input: JobInput
    video: JobVideo | None = None
    model: ModelIdentity | None = Field(default=None, description="Immutable identity of the model used for this job.")
    analysis_fps: float = Field(gt=0)


class JobList(ResponseModel):
    items: list[Job]
    next_cursor: str | None = Field(default=None, description="Opaque cursor for the next page; null at the end.")


class JobTimingMs(ResponseModel):
    """Measured wall-clock timings in milliseconds. Inference percentiles are null when no frame was analyzed."""

    decode: float = Field(ge=0)
    inference_p50: float | None = Field(default=None, ge=0)
    inference_p95: float | None = Field(default=None, ge=0)
    postprocess: float = Field(ge=0)
    encode: float = Field(ge=0)
    total_wall: float = Field(ge=0)


class TrackSummary(ResponseModel):
    track_id: int = Field(ge=0)
    class_name: str
    first_t: float = Field(ge=0, description="First video PTS (seconds) the track was seen.")
    last_t: float = Field(ge=0, description="Last video PTS (seconds) the track was seen.")
    frames: int = Field(ge=1)
    max_score: float = Field(ge=0, le=1, description="Highest detector confidence (not a safety probability).")


class JobSummary(ResponseModel):
    analyzed_frames: int = Field(ge=0)
    decoded_frames: int = Field(ge=0)
    coverage_ratio: float = Field(ge=0, le=1)
    duration_s: float = Field(ge=0)
    effective_analyzed_fps: float = Field(ge=0, description="Analyzed frames divided by total wall time.")
    timing_ms: JobTimingMs
    decision_counts: dict[str, int] = Field(description="Analytical decision state -> analyzed frame count.")
    reason_counts: dict[str, int] = Field(description="Reason code -> analyzed frame count.")
    tracks: list[TrackSummary]
    class_counts: dict[str, int] = Field(description="Class name -> detection count over analyzed frames.")
    audio: Literal["removed"] = "removed"


class JobArtifact(ResponseModel):
    name: ArtifactName
    url: str = Field(description="Same-origin, owner-checked relative URL.")
    content_type: str
    bytes: int = Field(ge=0)


class JobResults(ResponseModel):
    job_id: str
    state: JobState
    mode: Literal["video_analysis"] = "video_analysis"
    perception_source: PerceptionSource = Field(description="Source of every perception in this job's results.")
    model: ModelIdentity | None = Field(default=None, description="Model that produced the results, if learned.")
    summary: JobSummary
    artifacts: list[JobArtifact]
    timeline_url: str | None = None
    limitations: list[str]


class DeleteResult(ResponseModel):
    id: str
    status: Literal["deleted", "deletion_scheduled"] = Field(
        description="deletion_scheduled: the job was in progress; cancellation was requested and bytes are removed "
        "once the worker stops."
    )


# ---------------------------------------------------------------------------------------------------------------------
# Simulators and road graph
# ---------------------------------------------------------------------------------------------------------------------
class PerceptionModeInfo(ResponseModel):
    id: SimPerceptionSource
    available: bool
    reason: str | None = Field(default=None, description="Why the mode is unavailable; null when available.")


class ScenarioInfo(ResponseModel):
    id: ScenarioId
    available: bool
    reason: str | None = None


class RoadNode(ResponseModel):
    id: str
    x: float = Field(description="East coordinate, metres.")
    y: float = Field(description="North coordinate, metres.")
    label_uz: str
    label_en: str
    reachable_from_default: bool


class RoadEdge(ResponseModel):
    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True, serialize_by_alias=True)

    id: str
    from_: str = Field(alias="from")
    to: str
    polyline: list[Annotated[list[float], Field(min_length=2, max_length=2)]] = Field(
        description="[[x, y], ...] in metres from `from` to `to`."
    )


class RoadGraphInfo(ResponseModel):
    nodes: list[RoadNode]
    edges: list[RoadEdge]
    lane_width_m: float = Field(gt=0)
    units: Literal["m"] = "m"
    frame: Literal["ENU x-east y-north yaw CCW"] = "ENU x-east y-north yaw CCW"

    @classmethod
    def from_graph(cls, graph: RoadGraph, reachable_from_default: set[str] | frozenset[str]) -> RoadGraphInfo:
        """Build the wire shape from the static simulator topology (app.simulators.base.RoadGraph)."""
        return cls(
            nodes=[
                RoadNode(
                    id=n.id,
                    x=n.x,
                    y=n.y,
                    label_uz=n.label_uz,
                    label_en=n.label_en,
                    reachable_from_default=n.id in reachable_from_default,
                )
                for n in graph.nodes
            ],
            edges=[
                RoadEdge.model_validate(
                    {"id": e.id, "from": e.a, "to": e.b, "polyline": [[float(x), float(y)] for x, y in e.polyline]}
                )
                for e in graph.edges
            ],
            lane_width_m=graph.lane_width_m,
        )


class Simulator(ResponseModel):
    id: SimulatorId
    name: str
    available: bool
    reason: str | None = None
    perception_modes: list[PerceptionModeInfo]
    scenarios: list[ScenarioInfo]
    road_graph: RoadGraphInfo | None = Field(default=None, description="Null when the simulator is unavailable.")
    default_start: str | None = None


class SimulatorList(ResponseModel):
    items: list[Simulator]


# ---------------------------------------------------------------------------------------------------------------------
# Simulation runs
# ---------------------------------------------------------------------------------------------------------------------
class RunCreate(RequestModel):
    simulator: SimulatorId = "lite"
    perception_source: SimPerceptionSource = "synthetic_rgb_cv"
    start_node: str = Field(min_length=1, max_length=64)
    destination_node: str = Field(min_length=1, max_length=64)
    scenario: ScenarioId = "normal"
    seed: int = Field(default=42, ge=0, le=SEED_MAX)


class RunRoute(ResponseModel):
    node_ids: list[str]
    polyline: list[Annotated[list[float], Field(min_length=2, max_length=2)]] = Field(
        description="Lane-centre path [[x, y], ...] in metres."
    )
    length_m: float = Field(ge=0)


class LiveEgo(ResponseModel):
    """Ego localization telemetry from the simulator (allowed runtime input), ENU frame."""

    source: Literal["ego_telemetry"] = "ego_telemetry"
    x_m: float
    y_m: float
    yaw_rad: float = Field(description="Radians, counter-clockwise from +x (east).")
    speed_mps: float = Field(ge=0)


class LiveDecision(ResponseModel):
    state: DecisionState
    reason_codes: list[ReasonCode] = Field(min_length=1)
    action_kind: ActionKind


class LiveCommand(ResponseModel):
    """Simulator-only actuation computed by the deterministic controller; never sent to a physical vehicle."""

    steer: float = Field(ge=-1, le=1, description="+1 = full left.")
    throttle: float = Field(ge=0, le=1)
    brake: float = Field(ge=0, le=1)
    input_frame_id: int = Field(ge=0)
    source: Literal["dreamdrive_controller"] = "dreamdrive_controller"
    safety_override: bool
    reason_codes: list[ReasonCode] = Field(min_length=1)


class LivePerception(ResponseModel):
    source: PerceptionSource
    quality_status: QualityStatus
    lane_status: LaneStatus
    image_lane_offset_normalized: float | None = Field(default=None, ge=-1, le=1)
    image_obstacle_cue: float | None = Field(default=None, ge=0, le=1)
    obstacle_boxes: list[Annotated[list[float], Field(min_length=4, max_length=4)]] = Field(
        description="Perceived obstacle boxes [[x1, y1, x2, y2], ...] in frame pixels."
    )
    lane_points: list[Annotated[list[float], Field(min_length=2, max_length=2)]] = Field(
        description="Perceived lane points [[x, y], ...] in frame pixels."
    )
    frame_width: int = Field(ge=1)
    frame_height: int = Field(ge=1)


class LiveManeuver(ResponseModel):
    """What the simulated vehicle is doing on this tick: a deterministic description classified from the applied
    command (after the safety supervisor) and ego telemetry (app.control.maneuver). It is not a controller."""

    primary: ManeuverPrimary = Field(
        description="Display code; precedence HOLD > STOP > BRAKE > TURN_* > SLOW_DOWN > ACCELERATE > CRUISE."
    )
    longitudinal: ManeuverLongitudinal
    lateral: ManeuverLateral = Field(description="Turn direction; left = counter-clockwise yaw (steer > 0).")
    since_sim_time_s: float = Field(ge=0, description="Simulated time (s) at which the current primary began.")
    input_frame_id: int = Field(ge=0, description="Input frame id of the command this maneuver describes.")
    speed_mps: float = Field(ge=0, description="Ego telemetry speed, m/s.")
    accel_mps2: float | None = Field(
        default=None, description="Longitudinal acceleration used for the label, m/s^2; null when no command applied."
    )
    yaw_rate_radps: float | None = Field(
        default=None, description="Yaw rate used for the label, rad/s (+ = left); null when no command applied."
    )
    signal_source: ManeuverSignalSource = Field(
        description="ego_telemetry_window: measured over the last 0.5 s of simulated time; command_model: LiteRoad "
        "pedal/bicycle model of the applied command (first ticks); none: no command (hold)."
    )


class DecisionBasisItem(ResponseModel):
    """One input that the safety supervisor, FSM or controller actually evaluated for this tick's decision, with
    the threshold it was compared against. Never contains evaluator truth."""

    key: str = Field(description="Stable input name, e.g. image_obstacle_cue, frame_wall_age_s, target_speed_mps.")
    value: bool | int | float | str | None = Field(description="Evaluated value; null when the input was unavailable.")
    unit: str | None = Field(default=None, description="SI unit of value/threshold (s, m, m/s, 1/m) or null.")
    source: DecisionBasisSource
    threshold: float | None = Field(default=None, description="Compared limit; null for informational inputs.")
    comparator: DecisionBasisComparator | None = Field(
        default=None, description="Trigger condition is `value <comparator> threshold`."
    )
    triggered: bool = Field(description="True when this input currently drives `effect` (gate: condition holds).")
    effect: DecisionBasisEffect | None = Field(default=None, description="What the input drives when triggered.")
    input_frame_id: int = Field(ge=0, description="Equals the command's input_frame_id (same camera frame).")


class RunLive(ResponseModel):
    frame_id: int = Field(ge=0)
    sim_time_s: float = Field(ge=0)
    wall_time_s: float = Field(ge=0)
    realtime_factor: float | None = Field(default=None, ge=0, description="Measured sim time / wall time.")
    ego: LiveEgo
    decision: LiveDecision
    command: LiveCommand | None = None
    perception: LivePerception
    route_progress: float = Field(ge=0, le=1)
    acquisition_age_s: float | None = Field(default=None, ge=0)
    maneuver: LiveManeuver | None = Field(
        default=None, description="Current maneuver; null for live states recorded before this field existed."
    )
    decision_basis: list[DecisionBasisItem] = Field(
        default_factory=list,
        description="Inputs evaluated for this frame's decision, in pipeline order (frame checks, perception "
        "checks, ego/limits, FSM, controller, command).",
    )


class Run(ResponseModel):
    id: str
    state: RunState
    outcome: str | None = Field(
        default=None,
        description="Terminal outcome, e.g. arrived, stopped_operator, stopped_before_obstacle, "
        "stopped_camera_invalid, failed_collision, failed_out_of_bounds, failed_wall_time, failed_tick_limit.",
    )
    simulator: SimulatorId
    perception_source: SimPerceptionSource
    scenario: ScenarioId
    seed: int = Field(ge=0, le=SEED_MAX)
    start_node: str
    destination_node: str
    route: RunRoute
    created_utc: UtcDatetime
    started_utc: UtcDatetime | None = None
    finished_utc: UtcDatetime | None = None
    error_code: str | None = None
    error_message: str | None = None
    live: RunLive | None = Field(default=None, description="Latest live state; null before the first frame.")


class RunList(ResponseModel):
    items: list[Run]
    next_cursor: str | None = None


class RunControl(RequestModel):
    action: RunAction


class ObstacleInject(RequestModel):
    """Synthetic scenario injection of a visual obstacle (simulation only)."""

    distance_ahead_m: float = Field(ge=8, le=40)
    lateral_offset_m: float = Field(ge=-1.5, le=1.5)


class ObstacleInjectResult(ResponseModel):
    run_id: str
    source: Literal["synthetic_scenario_injection"] = "synthetic_scenario_injection"
    distance_ahead_m: float
    lateral_offset_m: float
    injected_total: int = Field(ge=1, description="Injections accepted for this run including this one.")
    max_per_run: int = Field(ge=1)


class RunReportResponse(RunReport):
    """contracts/run-report.schema.json plus run-specific ``details`` (command summary, decision counts, ...)."""

    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)

    details: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------------------------------------------------
# Evaluations
# ---------------------------------------------------------------------------------------------------------------------
class Evaluation(ResponseModel):
    id: str
    kind: EvaluationKind
    state: EvaluationState
    created_utc: UtcDatetime
    finished_utc: UtcDatetime | None = None
    source: str = Field(description="What produced the samples, e.g. literoad_synthetic_rgb_cv.")
    sample_count: int = Field(ge=0)
    metrics: dict[str, Metric]
    limitations: list[str]
    code_commit: str | None = Field(default=None, pattern=COMMIT_PATTERN, description="Null when not built from git.")
    model_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    seeds: list[int] | None = None
    error_code: str | None = None


class EvaluationList(ResponseModel):
    items: list[Evaluation]


class EvaluationCreate(RequestModel):
    """Only the synthetic scenario suite can be started from the API; ML benchmarks are produced offline."""

    kind: Literal["synthetic_scenario_suite"] = "synthetic_scenario_suite"
    seeds: list[Annotated[int, Field(ge=0, le=SEED_MAX)]] = Field(
        default_factory=lambda: [1, 2, 3], min_length=1, max_length=MAX_EVALUATION_SEEDS
    )

    @field_validator("seeds")
    @classmethod
    def _unique(cls, seeds: list[int]) -> list[int]:
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        return seeds


# ---------------------------------------------------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------------------------------------------------
class BuildInfo(ResponseModel):
    commit: str | None = Field(default=None, description="40-64 hex git commit; null when not built from git.")
    version: str
    dirty: bool


class VideoAnalysisCapability(ResponseModel):
    available: bool
    reason: str | None = Field(default=None, description="Why video analysis is unavailable (e.g. model_missing).")
    model: ModelInfo | None = None
    device: str | None = None
    precision: str | None = None
    supported_classes: list[str] = Field(description="Classes the verified checkpoint can report.")
    unsupported_examples: list[str] = Field(default_factory=lambda: list(UNSUPPORTED_CLASS_EXAMPLES))


class SimulationCapability(ResponseModel):
    simulators: list[Simulator]


class QueueStatus(ResponseModel):
    """Measured queue occupancy; counts are null (with reason) when the queue cannot be read."""

    video_queued: int | None = Field(default=None, ge=0)
    video_running: int | None = Field(default=None, ge=0)
    capacity_total: int = Field(ge=1)
    per_session: int = Field(ge=1)
    sim_active: int | None = Field(default=None, ge=0)
    sim_capacity: int = Field(default=MAX_ACTIVE_SIMULATIONS, ge=1)
    reason: str | None = None


class MonitoringInfo(ResponseModel):
    metrics: bool = Field(description="Prometheus metrics are exported (internal network only).")


class Capabilities(ResponseModel):
    build: BuildInfo
    video_analysis: VideoAnalysisCapability
    simulation: SimulationCapability
    queue: QueueStatus
    limits: Limits
    monitoring: MonitoringInfo


# ---------------------------------------------------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------------------------------------------------
class CleanupResult(ResponseModel):
    removed_jobs: int = Field(ge=0)
    removed_runs: int = Field(ge=0)
    removed_evaluations: int = Field(ge=0)
    removed_bytes: int = Field(ge=0)
    retention_hours: float = Field(gt=0)
    started_utc: UtcDatetime
    finished_utc: UtcDatetime
