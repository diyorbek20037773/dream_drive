"""SimulationSession: the LiteRoad closed loop (docs/IMPLEMENTATION_SPEC.md section 4.1).

Per tick: acquire frame -> validate (id monotonic, sim age, wall age) -> perception on pixels -> world state ->
FSM -> Pure Pursuit (route) + pixel lane correction -> PID speed -> safety supervisor override -> command clamp ->
apply -> step(dt) -> evaluator measurement (after the step, for reporting only).

Isolation: perception receives only ``frame.image`` (+ static calibration at construction); the FSM receives
perception outputs, ego telemetry and route geometry; the controller receives ego telemetry, the static route and
the perceived lane curve. The evaluator object is used only after ``step`` and never feeds back into decisions.

Explainability (Run.live): ``decision_basis`` lists, for the command's input frame, the inputs the safety supervisor,
FSM and controller actually evaluated (taken from their own records, never recomputed and never evaluator truth);
``maneuver`` describes what the vehicle is doing, classified from the applied command and ego telemetry
(``app.control.maneuver``). Maneuver changes are kept as a compact timeline for the run report.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, cast

import cv2
import numpy as np

from app.agents.fsm import DrivingFsm, FsmDecision, FsmInputs
from app.agents.runtime_view import RuntimeView
from app.agents.safety import SafetyLimits, SafetySupervisor
from app.config import get_settings
from app.control.basis import BasisItem
from app.control.command import brake_override, command_to_wire, finalize_command
from app.control.controller import Controller
from app.control.maneuver import AppliedCommand, Maneuver, ManeuverClassifier
from app.navigation.astar import RouteUnreachable
from app.navigation.route import RoadSurface, Route, cached_route
from app.perception.synthetic_rgb_cv import SyntheticPerceptionResult, SyntheticRgbPerception, unavailable_result
from app.reporting.sim_report import build_simulation_report
from app.schemas.reason_codes import ReasonCode
from app.schemas.report import Metric
from app.schemas.world_state import (
    Command,
    Decision,
    DecisionState,
    EgoTelemetry,
    FrameInfo,
    QualityStatus,
    WorldState,
)
from app.simulators.base import ControlCommand, EgoState, RgbFrame, RoadGraph
from app.simulators.lite.adapter import LiteRoadAdapter
from app.simulators.lite.graph import JUNCTION_HALF_SIZE_M, build_town_graph, junction_nodes
from app.simulators.lite.scenarios import SCENARIOS, point_on_route, resolve_setup

__all__ = ["RouteUnreachable", "SessionConfig", "SimulationSession", "TickResult", "UnsupportedMode"]

R = ReasonCode
PERCEPTION_SOURCES = ("synthetic_rgb_cv", "learned_model")
ARRIVAL_RADIUS_M = 2.0
ARRIVAL_SPEED_MPS = 0.5
STANDSTILL_MPS = 0.05
STOP_HOLD_S = 5.0
MAX_INJECTED_OBSTACLES = 5
JPEG_QUALITY = 85
MANEUVER_TIMELINE_MAX = 2000  # changes kept for the report (a run has at most max_ticks ticks)

_STOP_OUTCOMES = (
    (R.OPERATOR_STOP.value, "stopped_operator"),
    (R.CAMERA_BLACK.value, "stopped_camera_invalid"),
    (R.CAMERA_INVALID.value, "stopped_camera_invalid"),
    (R.FRAME_STALE.value, "stopped_frame_stale"),
    (R.PERCEPTION_TIMEOUT.value, "stopped_perception_timeout"),
    (R.INFERENCE_FAILED.value, "stopped_perception_failed"),
    (R.OBSTACLE_AHEAD_STOP.value, "stopped_before_obstacle"),
)
_FAILURE_OUTCOMES = {
    R.EGO_OUT_OF_BOUNDS.value: "failed_out_of_bounds",
    R.TICK_LIMIT.value: "failed_tick_limit",
    R.WALL_TIME_LIMIT.value: "failed_wall_time",
    R.COLLISION_DETECTED_BY_EVALUATOR.value: "failed_collision",
}


class UnsupportedMode(ValueError):
    """Requested perception mode / scenario / timing is not supported (API maps to 422)."""


class _Perception(Protocol):
    source: str

    def process(self, image: np.ndarray, frame_id: int = 0) -> SyntheticPerceptionResult: ...


@dataclass
class SessionConfig:
    run_id: str
    start_node: str
    destination_node: str
    scenario: str = "normal"
    seed: int = 42
    perception_source: str = "synthetic_rgb_cv"
    dt: float = 0.05
    max_ticks: int = 2400
    encode_jpeg: bool = True  # headless evaluation may skip JPEG encoding (TickResult.jpeg is then b"")


@dataclass
class TickResult:
    frame_id: int
    sim_time_s: float
    jpeg: bytes
    world_state: WorldState
    live: dict[str, Any]
    run_state: str
    outcome: str | None
    terminal: bool


class _SyntheticAdapterPerception:
    """Adapter so the synthetic pipeline matches the ``process(image, frame_id)`` protocol."""

    source = "synthetic_rgb_cv"

    def __init__(self, inner: SyntheticRgbPerception) -> None:
        self.inner = inner

    def process(self, image: np.ndarray, frame_id: int = 0) -> SyntheticPerceptionResult:
        return self.inner.process(image)


class SimulationSession:
    # ------------------------------------------------------------------------------------------ static helpers
    @staticmethod
    def road_graph() -> RoadGraph:
        return build_town_graph()

    @staticmethod
    def scenarios() -> list[dict[str, Any]]:
        return [{"id": s.id, "available": s.available, "reason": s.reason} for s in SCENARIOS.values()]

    @staticmethod
    def plan_route(start: str, dest: str) -> dict[str, Any]:
        return cached_route(build_town_graph(), start, dest).to_dict()

    # ------------------------------------------------------------------------------------------ construction
    def __init__(self, cfg: SessionConfig, inference_client: Any = None, *, adapter: Any = None) -> None:
        if cfg.scenario not in SCENARIOS:
            raise UnsupportedMode(f"unknown scenario {cfg.scenario!r}")
        if cfg.perception_source not in PERCEPTION_SOURCES:
            raise UnsupportedMode(f"unknown perception source {cfg.perception_source!r}")
        if not (0 < cfg.dt <= 0.2) or cfg.max_ticks <= 0:
            raise UnsupportedMode("dt must be in (0, 0.2] s and max_ticks positive")
        settings = get_settings()
        if cfg.perception_source == "learned_model":
            if not settings.learned_sim_mode_enabled:
                raise UnsupportedMode(
                    "learned_model simulator perception is disabled: it has not passed the simulator validation "
                    "scenarios (LEARNED_SIM_MODE_ENABLED=false)"
                )
            if inference_client is None:
                raise UnsupportedMode("learned_model simulator perception needs an inference client")
        self.cfg = cfg
        self._graph = build_town_graph()
        self._route: Route = cached_route(self._graph, cfg.start_node, cfg.destination_node)  # RouteUnreachable
        self._adapter = adapter if adapter is not None else LiteRoadAdapter()
        self._view = RuntimeView(self._adapter)
        self._calib = self._view.get_camera_calibration()
        self._perception: _Perception
        if cfg.perception_source == "learned_model":
            from app.perception.learned_sim import LearnedModelSimPerception

            self._perception = LearnedModelSimPerception(inference_client, self._calib)
        else:
            self._perception = _SyntheticAdapterPerception(SyntheticRgbPerception(self._calib))
        self._source = self._perception.source
        self._wall_limit_s = float(settings.max_simulation_wall_seconds)
        perception_timeout = (
            float(settings.sim_inference_timeout_seconds) if cfg.perception_source == "learned_model" else 5.0
        )
        road = RoadSurface(self._graph, JUNCTION_HALF_SIZE_M, junction_nodes(self._graph))
        self._safety = SafetySupervisor(SafetyLimits(perception_timeout_s=perception_timeout), road)
        self._fsm = DrivingFsm(cfg.dt)
        self._controller = Controller(cfg.dt)
        self._maneuver = ManeuverClassifier(cfg.dt)
        self._maneuver_timeline: list[dict[str, Any]] = []
        self._maneuver_counts: Counter[str] = Counter()
        self._maneuver_truncated = False

        self._run_state = "initializing"
        self._outcome: str | None = None
        self._terminal = False
        self._paused = False
        self._stop_requested = False
        self._wall_limit_hit = False
        self._steps = 0
        self._ticks = 0
        self._last_frame_id: int | None = None
        self._pending: tuple[RgbFrame, SyntheticPerceptionResult, tuple[str, ...], tuple[BasisItem, ...]] | None = None
        self._last: TickResult | None = None
        self._last_frame: RgbFrame | None = None
        self._last_perception: SyntheticPerceptionResult | None = None
        self._hint: int | None = None
        self._stop_hold_s = 0.0
        self._wall_start = 0.0
        self._collisions_seen = 0
        self._injected = 0
        self._setup_obstacle_s: tuple[float, ...] = ()
        # statistics for the report
        self._decision_counts: Counter[str] = Counter()
        self._reason_counts: Counter[str] = Counter()
        self._quality_counts: Counter[str] = Counter()
        self._lane_counts: Counter[str] = Counter()
        self._deviations: list[float] = []
        self._tick_ms: list[float] = []
        self._perception_ms: list[float] = []
        self._overrides = 0
        self._clamped = 0
        self._nan_rejected = 0
        self._steer_abs: list[float] = []
        self._throttle: list[float] = []
        self._brake: list[float] = []
        self._min_gap: float | None = None
        self._final_gap: float | None = None
        self._first_safety_tick: dict[str, Any] | None = None
        self._commands: list[ControlCommand] = []

    # ------------------------------------------------------------------------------------------ properties
    @property
    def route(self) -> Route:
        return self._route

    @property
    def run_state(self) -> str:
        return self._run_state

    @property
    def outcome(self) -> str | None:
        return self._outcome

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def commands(self) -> list[ControlCommand]:
        """Every command applied so far (tests compare these for determinism / hidden-truth isolation)."""
        return list(self._commands)

    @property
    def perception_source(self) -> str:
        return self._source

    @property
    def last_frame_image(self) -> np.ndarray | None:
        """RGB pixels of the most recent frame perception consumed (read-only array)."""
        return self._last_frame.image if self._last_frame is not None else None

    @property
    def tick_wall_ms_samples(self) -> list[float]:
        return list(self._tick_ms)

    @property
    def lateral_deviation_samples(self) -> list[float]:
        """Evaluator measurements after each physics step (reporting only)."""
        return list(self._deviations)

    # ------------------------------------------------------------------------------------------ helpers
    def _perceive(self, frame: RgbFrame) -> tuple[SyntheticPerceptionResult, tuple[str, ...], tuple[BasisItem, ...]]:
        t0 = time.perf_counter()
        res = self._perception.process(frame.image, frame.frame_id)
        elapsed = time.perf_counter() - t0
        self._perception_ms.append(elapsed * 1000.0)
        codes = self._safety.check_perception(
            res.quality_status,
            res.quality_codes,
            res.lane_status,
            res.image_lane_offset_normalized,
            res.image_obstacle_cue,
            elapsed,
        )
        return res, codes, self._safety.last_perception_basis

    def _record_maneuver(self, m: Maneuver, sim_time_s: float, reason_codes: list[str] | tuple[str, ...]) -> None:
        """Counts ticks per maneuver and keeps the changes (compact timeline for the report)."""
        self._maneuver_counts[m.primary] += 1
        last = self._maneuver_timeline[-1]["primary"] if self._maneuver_timeline else None
        if m.primary == last:
            return
        if len(self._maneuver_timeline) >= MANEUVER_TIMELINE_MAX:
            self._maneuver_truncated = True
            return
        self._maneuver_timeline.append(
            {
                "sim_time_s": round(sim_time_s, 4),
                "since_sim_time_s": round(m.since_sim_time_s, 4),
                "frame_id": int(m.input_frame_id),
                "primary": m.primary,
                "longitudinal": m.longitudinal,
                "lateral": m.lateral,
                "speed_mps": round(m.speed_mps, 4),
                "reason_codes": list(dict.fromkeys(reason_codes)),
            }
        )

    def _frame_fault_result(self, quality: str, codes: tuple[str, ...]) -> SyntheticPerceptionResult:
        """Perception record for a frame that was never handed to perception (missing / stale)."""
        res = unavailable_result(self._calib, quality, codes, codes, source=self._source)
        if self._source != "learned_model":
            return res
        model_id = getattr(self._perception, "model_id", None)
        model_sha = getattr(self._perception, "model_sha256", None)
        if model_id and model_sha:
            return dataclasses.replace(res, model_id=model_id, model_sha256=model_sha)
        # a learned_model record must carry the checkpoint identity; none is known before the first inference
        return dataclasses.replace(res, source="unavailable")

    def _jpeg(self, image: np.ndarray) -> bytes:
        if not self.cfg.encode_jpeg:
            return b""
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return buf.tobytes()

    def _wall(self) -> float:
        return time.monotonic() - self._wall_start

    def _build(
        self,
        frame: RgbFrame | None,
        frame_id: int,
        sim_time: float,
        quality: str,
        pres: SyntheticPerceptionResult,
        ego: EgoState,
        decision: FsmDecision,
        command: ControlCommand | None,
        route_progress: float,
        acquisition_age: float | None,
        jpeg: bytes,
        maneuver: Maneuver | None = None,
        basis: tuple[BasisItem, ...] = (),
    ) -> TickResult:
        now_utc = datetime.now(UTC)
        acquired_utc = now_utc
        if frame is not None:
            acquired_utc = now_utc - timedelta(seconds=max(0.0, time.monotonic() - frame.acquired_wall_monotonic))
        perception = pres.to_perception()
        cmd_wire = command_to_wire(command) if command is not None else None
        ws = WorldState(
            run_id=self.cfg.run_id,
            mode="simulation",
            frame=FrameInfo(
                id=frame_id,
                timestamp_seconds=sim_time,
                clock="simulation_time",
                acquired_utc=acquired_utc,
                width=self._calib.width,
                height=self._calib.height,
                quality_status=cast(QualityStatus, quality),
            ),
            perception=perception,
            ego=EgoTelemetry(x_m=ego.x_m, y_m=ego.y_m, yaw_rad=ego.yaw_rad, speed_mps=max(0.0, ego.speed_mps)),
            decision=Decision(
                state=cast(DecisionState, decision.state),
                reason_codes=list(decision.reason_codes),
                action_kind="simulator_control",
            ),
            command=Command(**cmd_wire) if cmd_wire is not None else None,  # type: ignore[arg-type]
        )
        wall = self._wall()
        overlay = pres.overlay()
        live = {
            "frame_id": frame_id,
            "sim_time_s": round(sim_time, 6),
            "wall_time_s": round(wall, 6),
            "realtime_factor": round(sim_time / wall, 6) if wall > 0 else None,
            "ego": {
                "source": "ego_telemetry",
                "x_m": round(ego.x_m, 6),
                "y_m": round(ego.y_m, 6),
                "yaw_rad": round(ego.yaw_rad, 6),
                "speed_mps": round(max(0.0, ego.speed_mps), 6),
            },
            "decision": {
                "state": decision.state,
                "reason_codes": list(decision.reason_codes),
                "action_kind": "simulator_control",
            },
            "command": cmd_wire,
            "perception": {
                "source": pres.source,
                "quality_status": quality,
                "lane_status": pres.lane_status,
                "image_lane_offset_normalized": pres.image_lane_offset_normalized,
                "image_obstacle_cue": pres.image_obstacle_cue,
                "obstacle_boxes": overlay["obstacle_boxes"],
                "lane_points": overlay["lane_points"],
                "frame_width": overlay["frame_width"],
                "frame_height": overlay["frame_height"],
            },
            "route_progress": round(min(max(route_progress, 0.0), 1.0), 6),
            "acquisition_age_s": round(acquisition_age, 6) if acquisition_age is not None else None,
            "maneuver": maneuver.to_wire() if maneuver is not None else None,
            # every item is tagged with the input frame id of this tick's command (or of the shown frame)
            "decision_basis": [item.to_wire(frame_id) for item in basis],
        }
        result = TickResult(
            frame_id=frame_id,
            sim_time_s=sim_time,
            jpeg=jpeg,
            world_state=ws,
            live=live,
            run_state=self._run_state,
            outcome=self._outcome,
            terminal=self._terminal,
        )
        self._last = result
        return result

    def _finish(self, run_state: str, outcome: str) -> None:
        self._run_state = run_state
        self._outcome = outcome
        self._terminal = True

    # ------------------------------------------------------------------------------------------ lifecycle
    def start(self) -> TickResult:
        """reset(seed, scenario), place the scenario scene, capture and perceive the initial frame. No stepping."""
        if self._last is not None:
            return self._last
        cfg = self.cfg
        self._adapter.reset(cfg.seed, cfg.scenario)
        setup = resolve_setup(SCENARIOS[cfg.scenario], cfg.seed, self._route)
        if hasattr(self._adapter, "spawn_ego"):
            self._adapter.spawn_ego(*setup.start_pose)
        if hasattr(self._adapter, "add_obstacle"):
            for x, y, yaw in setup.obstacles:
                self._adapter.add_obstacle(x, y, yaw, visible=True, source="scenario")
        self._setup_obstacle_s = setup.obstacle_s_m
        evaluator = self._adapter.evaluator()
        if hasattr(evaluator, "set_route"):
            evaluator.set_route(self._route)
        self._wall_start = time.monotonic()
        frame = self._view.get_rgb_frame()
        pres, codes, pbasis = self._perceive(frame)
        self._pending = (frame, pres, codes, pbasis)
        self._last_frame_id = frame.frame_id
        self._last_frame, self._last_perception = frame, pres
        ego = self._view.get_ego_telemetry()
        proj = self._route.project(ego.x_m, ego.y_m)
        self._hint = proj.index
        self._run_state = "running"
        decision = FsmDecision("observe", (R.INITIALIZING.value,))
        quality = pres.quality_status
        maneuver = self._maneuver.update(0.0, ego.speed_mps, ego.yaw_rad, None, None, frame.frame_id)
        self._record_maneuver(maneuver, 0.0, decision.reason_codes)
        return self._build(
            frame,
            frame.frame_id,
            frame.sim_time_s,
            quality,
            pres,
            ego,
            decision,
            None,
            self._route.progress(proj.s_m),
            time.monotonic() - frame.acquired_wall_monotonic,
            self._jpeg(frame.image),
            maneuver,
            pbasis,
        )

    def pause(self) -> None:
        if not self._terminal and not self._stop_requested:
            self._paused = True
            if self._run_state == "running":
                self._run_state = "paused"

    def resume(self) -> None:
        if not self._terminal and self._paused:
            self._paused = False
            self._run_state = "stopping" if self._stop_requested else "running"

    def request_stop(self) -> None:
        if self._terminal:
            return
        self._stop_requested = True
        self._paused = False
        self._run_state = "stopping"

    def note_wall_time(self, wall_elapsed_s: float) -> None:
        if wall_elapsed_s > self._wall_limit_s:
            self._wall_limit_hit = True

    def inject_obstacle(self, distance_ahead_m: float, lateral_offset_m: float) -> bool:
        """Synthetic scenario injection: a visible obstacle on the route ``distance_ahead_m`` ahead of the ego."""
        if self._terminal or self._last is None or self._injected >= MAX_INJECTED_OBSTACLES:
            return False
        if not (math.isfinite(distance_ahead_m) and math.isfinite(lateral_offset_m)):
            return False
        if not hasattr(self._adapter, "add_obstacle"):
            return False
        ego = self._view.get_ego_telemetry()
        s_ego = self._route.project(ego.x_m, ego.y_m, self._hint).s_m
        s = s_ego + distance_ahead_m
        if s > self._route.length_m:
            return False
        x, y, yaw = point_on_route(self._route, s, lateral_offset_m)
        self._adapter.add_obstacle(x, y, yaw, visible=True, source="injected")
        self._injected += 1
        return True

    # ------------------------------------------------------------------------------------------ tick
    def _paused_result(self) -> TickResult:
        assert self._last is not None and self._last_frame is not None and self._last_perception is not None
        t_sim = self._steps * self.cfg.dt
        decision = self._fsm.step(FsmInputs(sim_time_s=t_sim, paused=True, perception_source=self._source))
        self._decision_counts[decision.state] += 1
        for c in decision.reason_codes:
            self._reason_counts[c] += 1
        ego = self._view.get_ego_telemetry()
        prev = self._last
        # no command is applied while paused (physics frozen): the maneuver is HOLD
        maneuver = self._maneuver.update(t_sim, ego.speed_mps, ego.yaw_rad, None, None, prev.frame_id)
        self._record_maneuver(maneuver, t_sim, decision.reason_codes)
        return self._build(
            self._last_frame,
            prev.frame_id,
            prev.sim_time_s,
            prev.world_state.frame.quality_status,
            self._last_perception,
            ego,
            decision,
            None,
            prev.live["route_progress"],
            None,
            prev.jpeg,
            maneuver,
            self._fsm.last_basis,
        )

    def tick(self) -> TickResult:
        if self._last is None:
            raise RuntimeError("start() must be called before tick()")
        if self._terminal:
            return self._last
        if self._wall() > self._wall_limit_s:
            self._wall_limit_hit = True
        if self._wall_limit_hit:
            self._paused = False  # the wall watchdog ends paused runs too
        if self._paused:
            return self._paused_result()
        t_tick = time.perf_counter()
        cfg = self.cfg
        expected_sim = self._steps * cfg.dt

        # 1. frame + validation
        frame: RgbFrame | None
        basis: list[BasisItem] = []
        if self._pending is not None:
            # Initial frame captured by start(); the simulation has not advanced, so it has not aged.
            frame, pres, safety_codes, pbasis = self._pending
            basis.extend(pbasis)
            self._pending = None
            quality = pres.quality_status
        else:
            try:
                frame = self._view.get_rgb_frame()
            except Exception:  # noqa: BLE001 - any camera failure is a missing frame, never a crash
                frame = None
            check = self._safety.check_frame(frame, self._last_frame_id, expected_sim, cfg.dt, time.monotonic())
            basis.extend(check.basis)
            if check.ok and frame is not None:
                pres, safety_codes, pbasis = self._perceive(frame)
                basis.extend(pbasis)
                quality = pres.quality_status
                self._last_frame_id = frame.frame_id
            else:
                q = check.quality_status or "missing"
                pres = self._frame_fault_result(q, check.codes)
                safety_codes = check.codes
                quality = q
        frame_id = frame.frame_id if frame is not None else (self._last_frame_id or 0)
        frame_time = frame.sim_time_s if frame is not None else expected_sim
        acquisition_age = time.monotonic() - frame.acquired_wall_monotonic if frame is not None else None

        # 2. localization telemetry + static route (allowed runtime inputs)
        ego = self._view.get_ego_telemetry()
        proj = self._route.project(ego.x_m, ego.y_m, self._hint)
        self._hint = proj.index
        failure = self._safety.check_ego(ego)
        basis.extend(self._safety.last_ego_basis)
        tick_limit = self._ticks >= cfg.max_ticks
        basis.append(
            BasisItem("tick_count", self._ticks, "safety_supervisor", None, cfg.max_ticks, ">=", tick_limit, "fail")
        )
        if tick_limit:
            failure = R.TICK_LIMIT.value
        basis.append(
            BasisItem(
                "wall_time_s",
                self._wall(),
                "safety_supervisor",
                "s",
                self._wall_limit_s,
                ">",
                self._wall_limit_hit,
                "fail",
            )
        )
        if self._wall_limit_hit:
            failure = R.WALL_TIME_LIMIT.value
        dx, dy = self._route.destination
        dest_dist = math.hypot(ego.x_m - dx, ego.y_m - dy)
        near = dest_dist <= ARRIVAL_RADIUS_M
        arrived = near and ego.speed_mps < ARRIVAL_SPEED_MPS
        basis.append(
            BasisItem("destination_distance_m", dest_dist, "route_graph", "m", ARRIVAL_RADIUS_M, "<=", near, "arrive")
        )
        if near:
            basis.append(
                BasisItem(
                    "arrival_speed_mps",
                    ego.speed_mps,
                    "ego_telemetry",
                    "m/s",
                    ARRIVAL_SPEED_MPS,
                    "<",
                    arrived,
                    "arrive",
                )
            )

        # 3. decision + control
        decision = self._fsm.step(
            FsmInputs(
                sim_time_s=expected_sim,
                operator_stop=self._stop_requested,
                failure_code=failure,
                safety_codes=safety_codes,
                paused=False,
                arrived=arrived,
                obstacle_cue=pres.image_obstacle_cue,
                lane_status=pres.lane_status,
                perception_source=self._source,
            )
        )
        basis.extend(self._fsm.last_basis)
        out = self._controller.compute(
            decision.state, ego, self._route, proj.s_m, pres.lane_status, pres.lane_centre, self._source
        )
        basis.extend(out.basis)
        reasons = [*decision.reason_codes, *out.reason_codes]
        cmd = finalize_command(out.steer, out.throttle, out.brake, frame_id, reasons)
        if decision.state in ("safety_stop", "failed"):
            cmd = brake_override(cmd, decision.reason_codes)
        basis.append(
            BasisItem(
                "safety_override",
                cmd.safety_override,
                "safety_supervisor",
                triggered=cmd.safety_override,
                effect="safety_stop",
            )
        )
        if R.CONTROL_NAN_REJECTED.value in cmd.reason_codes:
            basis.append(
                BasisItem("command_nan_rejected", True, "safety_supervisor", triggered=True, effect="safety_stop")
            )
        elif R.CONTROL_CLAMPED.value in cmd.reason_codes:
            basis.append(BasisItem("command_clamped", True, "safety_supervisor", triggered=True))
        if cmd.safety_override:
            self._overrides += 1
            if self._first_safety_tick is None:
                self._first_safety_tick = {
                    "frame_id": frame_id,
                    "sim_time_s": round(frame_time, 4),
                    "reason_codes": list(cmd.reason_codes),
                }
        if R.CONTROL_CLAMPED.value in cmd.reason_codes:
            self._clamped += 1
        if R.CONTROL_NAN_REJECTED.value in cmd.reason_codes:
            self._nan_rejected += 1
        decision_out = FsmDecision(
            decision.state,
            tuple(
                dict.fromkeys(
                    [
                        *decision.reason_codes,
                        *(c for c in out.reason_codes if decision.state in ("follow_route", "slow")),
                    ]
                )
            ),
        )

        # 4. actuation + physics (the maneuver describes the command actually applied, before the physics step)
        self._adapter.apply_control(cmd)
        self._commands.append(cmd)
        maneuver = self._maneuver.update(
            expected_sim,
            ego.speed_mps,
            ego.yaw_rad,
            AppliedCommand(cmd.steer, cmd.throttle, cmd.brake, cmd.safety_override),
            out.target_speed_mps,
            frame_id,
        )
        self._record_maneuver(maneuver, expected_sim, cmd.reason_codes)
        self._ticks += 1
        self._decision_counts[decision.state] += 1
        for c in decision_out.reason_codes:
            self._reason_counts[c] += 1
        self._quality_counts[quality] += 1
        self._lane_counts[pres.lane_status] += 1
        self._steer_abs.append(abs(cmd.steer))
        self._throttle.append(cmd.throttle)
        self._brake.append(cmd.brake)
        self._last_frame, self._last_perception = frame or self._last_frame, pres

        if decision.state == "failed":
            self._finish("failed", _FAILURE_OUTCOMES.get(decision.reason_codes[0], "failed_internal"))
        elif decision.state == "arrived":
            self._finish("arrived", "arrived")
        else:
            self._adapter.step(cfg.dt)
            self._steps += 1
            ev = self._adapter.evaluator()
            self._deviations.append(float(ev.lateral_deviation_m()))
            gap = ev.min_obstacle_gap_m() if hasattr(ev, "min_obstacle_gap_m") else None
            if gap is not None:
                self._min_gap = gap if self._min_gap is None else min(self._min_gap, gap)
                self._final_gap = gap
            collisions = int(ev.collision_count())
            speed_after = self._view.get_ego_telemetry().speed_mps
            if collisions > self._collisions_seen:
                self._collisions_seen = collisions
                self._finish("failed", "failed_collision")
                decision_out = FsmDecision("failed", (R.COLLISION_DETECTED_BY_EVALUATOR.value,))
            elif decision.state in ("safety_stop", "controlled_stop"):
                if self._stop_requested and speed_after < STANDSTILL_MPS:
                    self._finish("stopped", "stopped_operator")
                else:
                    self._stop_hold_s = self._stop_hold_s + cfg.dt if speed_after < STANDSTILL_MPS else 0.0
                    if self._stop_hold_s >= STOP_HOLD_S - 1e-9:
                        outcome = next(
                            (o for code, o in _STOP_OUTCOMES if code in decision.reason_codes), "stopped_safety"
                        )
                        self._finish("stopped", outcome)
            else:
                self._stop_hold_s = 0.0
        if not self._terminal:
            self._run_state = "stopping" if self._stop_requested else "running"
        jpeg = self._jpeg(frame.image) if frame is not None else (self._last.jpeg if self._last else b"")
        result = self._build(
            frame,
            frame_id,
            frame_time,
            quality,
            pres,
            ego,
            decision_out,
            cmd,
            self._route.progress(proj.s_m),
            acquisition_age,
            jpeg,
            maneuver,
            tuple(basis),
        )
        self._tick_ms.append((time.perf_counter() - t_tick) * 1000.0)
        return result

    # ------------------------------------------------------------------------------------------ reporting
    def summary(self, wall_time_s: float | None = None) -> dict[str, Any]:
        """Measured run statistics (evaluator measurements are labelled as such)."""
        wall = self._wall() if wall_time_s is None else wall_time_s
        sim_time = self._steps * self.cfg.dt
        ev = self._adapter.evaluator()
        dev = np.asarray(self._deviations) if self._deviations else None
        tick_ms = np.asarray(self._tick_ms) if self._tick_ms else None
        perc_ms = np.asarray(self._perception_ms) if self._perception_ms else None
        try:
            progress = float(ev.route_progress())
        except RuntimeError:
            progress = None
        return {
            "run_id": self.cfg.run_id,
            "scenario": self.cfg.scenario,
            "seed": self.cfg.seed,
            "start_node": self.cfg.start_node,
            "destination_node": self.cfg.destination_node,
            "perception_source": self._source,
            "route": self._route.to_dict(),
            "run_state": self._run_state,
            "outcome": self._outcome,
            "ticks": self._ticks,
            "physics_steps": self._steps,
            "sim_time_s": round(sim_time, 4),
            "wall_time_s": round(wall, 4),
            "realtime_factor": round(sim_time / wall, 4) if wall > 0 else None,
            "decision_counts": dict(self._decision_counts),
            "reason_counts": dict(self._reason_counts),
            "perception_quality_counts": dict(self._quality_counts),
            "lane_status_counts": dict(self._lane_counts),
            # what the vehicle did: ticks per maneuver and the sequence of maneuver changes (runtime-only inputs)
            "maneuver_counts": dict(self._maneuver_counts),
            "maneuver_timeline": [dict(item) for item in self._maneuver_timeline],
            "maneuver_timeline_truncated": self._maneuver_truncated,
            "commands": {
                "count": len(self._commands),
                "safety_overrides": self._overrides,
                "clamped": self._clamped,
                "nan_rejected": self._nan_rejected,
                "mean_abs_steer": round(float(np.mean(self._steer_abs)), 5) if self._steer_abs else None,
                "max_abs_steer": round(float(np.max(self._steer_abs)), 5) if self._steer_abs else None,
                "mean_throttle": round(float(np.mean(self._throttle)), 5) if self._throttle else None,
                "mean_brake": round(float(np.mean(self._brake)), 5) if self._brake else None,
                "first_safety_override": self._first_safety_tick,
            },
            "evaluator": {
                "collisions": int(ev.collision_count()),
                "route_progress": round(progress, 5) if progress is not None else None,
                "lateral_deviation_mean_m": round(float(dev.mean()), 4) if dev is not None else None,
                "lateral_deviation_max_m": round(float(dev.max()), 4) if dev is not None else None,
                "lateral_deviation_samples": int(dev.size) if dev is not None else 0,
                "min_obstacle_gap_m": round(self._min_gap, 4) if self._min_gap is not None else None,
                "final_obstacle_gap_m": round(self._final_gap, 4) if self._final_gap is not None else None,
                "scenario_obstacle_route_s_m": [round(s, 3) for s in self._setup_obstacle_s],
                "injected_obstacles": self._injected,
            },
            "timing_ms": {
                "tick_p50": round(float(np.percentile(tick_ms, 50)), 3) if tick_ms is not None else None,
                "tick_p95": round(float(np.percentile(tick_ms, 95)), 3) if tick_ms is not None else None,
                "tick_max": round(float(tick_ms.max()), 3) if tick_ms is not None else None,
                "perception_p50": round(float(np.percentile(perc_ms, 50)), 3) if perc_ms is not None else None,
                "perception_p95": round(float(np.percentile(perc_ms, 95)), 3) if perc_ms is not None else None,
            },
        }

    def report(self, code_commit: str, wall_time_s: float) -> dict[str, Any]:
        """RunReport (kind simulation) + ``details``. Only available once the run is terminal."""
        if not self._terminal or self._outcome is None:
            raise RuntimeError("report_unavailable: run is not terminal")
        s = self.summary(wall_time_s)
        ev, timing = s["evaluator"], s["timing_ms"]
        n_steps = max(1, s["physics_steps"])
        metrics: dict[str, Metric] = {
            "ticks": Metric.measured(s["ticks"], "count", 1),
            "sim_time_s": Metric.measured(s["sim_time_s"], "s", 1),
            "wall_time_s": Metric.measured(round(wall_time_s, 4), "s", 1),
            "collisions": Metric.measured(ev["collisions"], "count", 1, "evaluator truth, measured after each step"),
            "safety_overrides": Metric.measured(s["commands"]["safety_overrides"], "count", max(1, s["ticks"])),
        }
        if s["realtime_factor"] is not None:
            metrics["realtime_factor"] = Metric.measured(s["realtime_factor"], "ratio", 1, "sim time / wall time")
        if ev["route_progress"] is not None:
            metrics["route_progress"] = Metric.measured(ev["route_progress"], "ratio", 1, "evaluator projection")
        if ev["lateral_deviation_samples"]:
            metrics["lateral_deviation_mean_m"] = Metric.measured(
                ev["lateral_deviation_mean_m"], "m", n_steps, "rear-axle distance to truth lane-centre path"
            )
            metrics["lateral_deviation_max_m"] = Metric.measured(ev["lateral_deviation_max_m"], "m", n_steps)
        else:
            metrics["lateral_deviation_mean_m"] = Metric.missing("m", "no physics step was executed")
        if ev["min_obstacle_gap_m"] is not None:
            metrics["min_obstacle_gap_m"] = Metric.measured(
                ev["min_obstacle_gap_m"], "m", n_steps, "ego footprint to nearest obstacle footprint"
            )
        else:
            metrics["min_obstacle_gap_m"] = Metric.missing("m", "no obstacle in this run")
        if timing["tick_p50"] is not None:
            metrics["tick_wall_ms_p50"] = Metric.measured(timing["tick_p50"], "ms", len(self._tick_ms))
            metrics["tick_wall_ms_p95"] = Metric.measured(timing["tick_p95"], "ms", len(self._tick_ms))
        metrics["ttc_seconds"] = Metric.missing(
            "s", "time-to-collision is not computed in synthetic mode", status="unsupported"
        )
        model_sha = getattr(self._perception, "model_sha256", None) if self._source == "learned_model" else None
        return build_simulation_report(
            run_id=self.cfg.run_id,
            outcome=self._outcome,
            perception_source=self._source,
            code_commit=code_commit,
            metrics=metrics,
            details=s,
            model_sha256=model_sha,
        )
