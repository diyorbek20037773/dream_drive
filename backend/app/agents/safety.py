"""Safety supervisor: frame/perception/ego/control checks and brake override (docs/SIMULATION.md "Safety checks").

Checks, in the order the session runs them each tick:
- frame acquisition: missing frame (exception / None) -> CAMERA_INVALID; frame id not newer than the last consumed
  one, simulated frame age > 1.5 dt, or wall-clock acquisition age above the limit -> FRAME_STALE;
- perception: unusable frame quality (CAMERA_BLACK / CAMERA_INVALID from the pixel validator), perception wall time
  above the timeout -> PERCEPTION_TIMEOUT, out-of-range / non-finite perception outputs -> CAMERA_INVALID
  (reported with ``perception_output_invalid``);
- ego: non-finite or out-of-range telemetry, outside the world bounds or more than 6 m from any road centreline
  (static topology) -> EGO_OUT_OF_BOUNDS (run failure);
- controls: NaN/inf rejection and clamping happen in ``app.control.command.finalize_command``;
- operator stop -> OPERATOR_STOP.
Any active code turns the command into a brake override (throttle 0, brake 1, ``safety_override=True``).

Decision basis: each check records the values it compared and the limit it used (``FrameCheck.basis``,
``last_perception_basis``, ``last_ego_basis``), in evaluation order and only for checks that actually ran.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.control.basis import BasisItem
from app.navigation.route import RoadSurface
from app.schemas.reason_codes import ReasonCode
from app.simulators.base import EgoState, RgbFrame

R = ReasonCode
ALLOWED_LANE_STATUS = frozenset({"valid", "uncertain", "unsupported", "unavailable"})
ALLOWED_QUALITY = frozenset({"valid", "degraded", "missing", "invalid", "stale"})


@dataclass(frozen=True)
class SafetyLimits:
    max_frame_wall_age_s: float = 1.0
    max_frame_sim_age_factor: float = 1.5  # x dt
    perception_timeout_s: float = 5.0
    max_road_distance_m: float = 6.0
    max_speed_mps: float = 5.5
    world_bounds: tuple[float, float, float, float] = (-50.0, 250.0, -50.0, 195.0)  # x_min, x_max, y_min, y_max


@dataclass(frozen=True)
class FrameCheck:
    ok: bool
    quality_status: str | None  # set when the frame is unusable: missing | stale
    codes: tuple[str, ...]
    basis: tuple[BasisItem, ...] = field(default=(), compare=False)


class SafetySupervisor:
    def __init__(self, limits: SafetyLimits, road: RoadSurface) -> None:
        self.limits = limits
        self.road = road
        self.last_perception_basis: tuple[BasisItem, ...] = ()
        self.last_ego_basis: tuple[BasisItem, ...] = ()

    def check_frame(
        self, frame: RgbFrame | None, last_frame_id: int | None, expected_sim_time_s: float, dt: float, wall_now: float
    ) -> FrameCheck:
        sup, stop = "safety_supervisor", "safety_stop"
        b: list[BasisItem] = [BasisItem("frame_received", frame is not None, sup, triggered=frame is None, effect=stop)]
        if frame is None:
            return FrameCheck(False, "missing", (R.CAMERA_INVALID.value,), tuple(b))
        if last_frame_id is not None:
            not_newer = frame.frame_id <= last_frame_id
            b.append(BasisItem("frame_id", frame.frame_id, sup, None, last_frame_id, "<=", not_newer, stop))
            if not_newer:
                return FrameCheck(False, "stale", (R.FRAME_STALE.value,), tuple(b))
        sim_age = expected_sim_time_s - frame.sim_time_s
        sim_limit = self.limits.max_frame_sim_age_factor * dt
        sim_stale = sim_age > sim_limit + 1e-9
        b.append(BasisItem("frame_sim_age_s", sim_age, sup, "s", sim_limit, ">", sim_stale, stop))
        if sim_stale:
            return FrameCheck(False, "stale", (R.FRAME_STALE.value,), tuple(b))
        wall_age = wall_now - frame.acquired_wall_monotonic
        wall_stale = wall_age > self.limits.max_frame_wall_age_s
        b.append(
            BasisItem("frame_wall_age_s", wall_age, sup, "s", self.limits.max_frame_wall_age_s, ">", wall_stale, stop)
        )
        if wall_stale:
            return FrameCheck(False, "stale", (R.FRAME_STALE.value,), tuple(b))
        return FrameCheck(True, None, (), tuple(b))

    def check_perception(
        self,
        quality_status: str,
        quality_codes: tuple[str, ...],
        lane_status: str,
        image_lane_offset: float | None,
        obstacle_cue: float | None,
        elapsed_s: float,
    ) -> tuple[str, ...]:
        codes: list[str] = []
        timeout = elapsed_s > self.limits.perception_timeout_s
        if timeout:
            codes.append(R.PERCEPTION_TIMEOUT.value)
        unusable = quality_status in ("missing", "invalid")
        if unusable:
            codes.extend(quality_codes or (R.CAMERA_INVALID.value,))
        invalid = quality_status not in ALLOWED_QUALITY or lane_status not in ALLOWED_LANE_STATUS
        if obstacle_cue is not None and (not math.isfinite(obstacle_cue) or not 0.0 <= obstacle_cue <= 1.0):
            invalid = True
        if image_lane_offset is not None and (not math.isfinite(image_lane_offset) or abs(image_lane_offset) > 1.0):
            invalid = True
        if invalid:
            codes.append(R.CAMERA_INVALID.value)
        sup, stop = "safety_supervisor", "safety_stop"
        limit = self.limits.perception_timeout_s
        self.last_perception_basis = (
            BasisItem("perception_wall_s", elapsed_s, sup, "s", limit, ">", timeout, stop),
            BasisItem("frame_quality", quality_status, sup, triggered=unusable, effect=stop),
            BasisItem("perception_output_valid", not invalid, sup, triggered=invalid, effect=stop),
        )
        return tuple(dict.fromkeys(codes))

    def check_ego(self, ego: EgoState) -> str | None:
        sup, fail = "safety_supervisor", "fail"
        vals = (ego.x_m, ego.y_m, ego.yaw_rad, ego.speed_mps)
        valid = all(math.isfinite(v) for v in vals) and 0.0 <= ego.speed_mps <= self.limits.max_speed_mps
        b = [BasisItem("ego_telemetry_valid", valid, sup, triggered=not valid, effect=fail)]
        self.last_ego_basis = tuple(b)
        if not valid:
            return R.EGO_OUT_OF_BOUNDS.value
        x0, x1, y0, y1 = self.limits.world_bounds
        inside = x0 <= ego.x_m <= x1 and y0 <= ego.y_m <= y1
        b.append(BasisItem("ego_inside_world_bounds", inside, sup, triggered=not inside, effect=fail))
        self.last_ego_basis = tuple(b)
        if not inside:
            return R.EGO_OUT_OF_BOUNDS.value
        dist = self.road.distance_to_centreline(ego.x_m, ego.y_m)
        off_road = dist > self.limits.max_road_distance_m
        b.append(BasisItem("ego_road_distance_m", dist, sup, "m", self.limits.max_road_distance_m, ">", off_road, fail))
        self.last_ego_basis = tuple(b)
        if off_road:
            return R.EGO_OUT_OF_BOUNDS.value
        return None
