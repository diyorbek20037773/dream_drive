"""Deterministic route + pixel-lane controller (no learned or language model in the loop).

Lateral: Pure Pursuit towards the lane-centre route point ``lookahead`` metres ahead (ego telemetry + static route).
The pixel lane estimate then pulls the target laterally towards the perceived lane centre at the same forward
distance (weight 0.5 when the lane is ``valid``, 0.25 when ``uncertain``, 0 when unavailable), but only on gently
curved route sections where the visible lane is the lane the route follows. A pixel ablation therefore changes the
steering command, and lane quality changes speed through the FSM.
Longitudinal: the FSM state selects a target speed (cruise / slow / stop), further limited by route curvature
(lateral acceleration budget) and by the remaining distance to the destination; a PID turns it into pedals.
Decision basis: ``ControlOutput.basis`` lists the values ``compute`` used (pixel blend weight and gates, curve and
destination speed limits, target speed) - see ``app.control.basis``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.control.basis import BasisItem
from app.control.pid import SpeedPID, accel_to_pedals
from app.control.pure_pursuit import pure_pursuit_steer, target_point
from app.navigation.route import Route
from app.perception.synthetic_rgb_cv import CurveFit
from app.schemas.reason_codes import ReasonCode
from app.simulators.base import EgoState

CRUISE_SPEED_MPS = 4.8
SLOW_SPEED_MPS = 2.5
LAT_ACCEL_BUDGET_MPS2 = 0.6
DESTINATION_DECEL_MPS2 = 0.8
PIXEL_WEIGHT = {"valid": 0.5, "uncertain": 0.25}
PIXEL_MAX_ROUTE_CURVATURE = 0.06  # 1/m: blend only where the route radius exceeds ~17 m
PIXEL_MAX_DISAGREEMENT_M = 1.0
STANDSTILL_MPS = 0.05
HOLD_BRAKE = 0.4

STOP_STATES = frozenset({"controlled_stop", "safety_stop", "arrived", "paused", "failed", "observe"})


@dataclass
class ControlOutput:
    steer: float
    throttle: float
    brake: float
    target_speed_mps: float
    reason_codes: list[str]
    pixel_weight: float = 0.0
    pixel_lateral_shift_m: float = 0.0
    curve_limited: bool = False
    debug: dict[str, float] = field(default_factory=dict)
    basis: tuple[BasisItem, ...] = ()


class Controller:
    def __init__(self, dt: float) -> None:
        self.dt = dt
        self.pid = SpeedPID()

    def target_speed(
        self, state: str, route: Route, s_ego: float, speed: float, basis: list[BasisItem] | None = None
    ) -> tuple[float, bool]:
        """Target speed (m/s) and whether the route curvature limits it. When ``basis`` is given, the evaluated
        limits are appended to it."""
        if state in STOP_STATES:
            return 0.0, False
        base = SLOW_SPEED_MPS if state == "slow" else CRUISE_SPEED_MPS
        horizon = max(8.0, speed * speed / (2 * DESTINATION_DECEL_MPS2) + 6.0)
        kappa = route.max_abs_curvature(s_ego, s_ego + horizon)
        v_curve = math.sqrt(LAT_ACCEL_BUDGET_MPS2 / kappa) if kappa > 1e-6 else math.inf
        remaining = max(0.0, route.length_m - s_ego)
        v_dest = math.sqrt(2 * DESTINATION_DECEL_MPS2 * max(0.0, remaining - 0.8))
        target = min(base, v_curve, v_dest)
        curve_limited = v_curve < base and v_curve <= v_dest
        if basis is not None:
            dest_limited = v_dest < base and v_dest < v_curve
            basis.append(BasisItem("route_curvature_ahead_1pm", kappa, "route_graph", "1/m"))
            basis.append(
                BasisItem(
                    "curve_speed_limit_mps", v_curve, "route_graph", "m/s", base, "<", curve_limited, "speed_limit"
                )
            )
            basis.append(BasisItem("route_remaining_m", remaining, "route_graph", "m"))
            basis.append(
                BasisItem(
                    "destination_speed_limit_mps", v_dest, "route_graph", "m/s", base, "<", dest_limited, "speed_limit"
                )
            )
        return target, curve_limited

    def compute(
        self,
        state: str,
        ego: EgoState,
        route: Route,
        s_ego: float,
        lane_status: str,
        lane_centre: CurveFit | None,
        pixel_source: str = "synthetic_rgb_cv",
    ) -> ControlOutput:
        """``pixel_source`` only labels the provenance of the pixel lane inputs in ``basis``."""
        reasons: list[str] = []
        basis: list[BasisItem] = []
        src = pixel_source
        tgt = target_point(route, ego.x_m, ego.y_m, ego.yaw_rad, s_ego, ego.speed_mps)
        y_target = tgt.y_left
        weight = PIXEL_WEIGHT.get(lane_status, 0.0) if lane_centre is not None else 0.0
        shift = 0.0
        kappa_ahead = route.max_abs_curvature(s_ego, tgt.s_m + 2.0)
        gate_curve = kappa_ahead <= PIXEL_MAX_ROUTE_CURVATURE
        if weight > 0:
            basis.append(
                BasisItem(
                    "pixel_blend_route_curvature_1pm",
                    kappa_ahead,
                    "route_graph",
                    "1/m",
                    PIXEL_MAX_ROUTE_CURVATURE,
                    "<=",
                    gate_curve,
                    "steer_correction",
                )
            )
        if weight > 0 and gate_curve and 2.5 <= tgt.x_fwd <= 24.0:
            y_pix = lane_centre.y_at(tgt.x_fwd)  # type: ignore[union-attr]
            disagreement = abs(y_pix - tgt.y_left)
            agree = disagreement <= PIXEL_MAX_DISAGREEMENT_M
            basis.append(
                BasisItem(
                    "pixel_lane_disagreement_m",
                    disagreement,
                    src,
                    "m",
                    PIXEL_MAX_DISAGREEMENT_M,
                    "<=",
                    agree,
                    "steer_correction",
                )
            )
            if agree:
                shift = weight * (y_pix - tgt.y_left)
                y_target = tgt.y_left + shift
            else:
                weight = 0.0
        else:
            weight = 0.0
        basis.append(BasisItem("pixel_blend_weight", weight, src, triggered=weight > 0, effect="steer_correction"))
        basis.append(
            BasisItem("pixel_lateral_shift_m", shift, src, "m", triggered=weight > 0, effect="steer_correction")
        )
        steer = pure_pursuit_steer(tgt.x_fwd, y_target)

        basis.append(BasisItem("ego_speed_mps", ego.speed_mps, "ego_telemetry", "m/s"))
        v_target, curve_limited = self.target_speed(state, route, s_ego, ego.speed_mps, basis)
        basis.append(BasisItem("target_speed_mps", v_target, "controller", "m/s"))
        if curve_limited:
            reasons.append(ReasonCode.CURVE_SPEED_LIMIT.value)
        if v_target <= 0.0 and ego.speed_mps <= STANDSTILL_MPS:
            self.pid.reset()
            throttle, brake = 0.0, HOLD_BRAKE
        else:
            accel = self.pid.update(v_target, ego.speed_mps, self.dt)
            throttle, brake = accel_to_pedals(accel, ego.speed_mps)
        return ControlOutput(
            steer=steer,
            throttle=throttle,
            brake=brake,
            target_speed_mps=v_target,
            reason_codes=reasons,
            pixel_weight=weight,
            pixel_lateral_shift_m=shift,
            curve_limited=curve_limited,
            basis=tuple(basis),
            debug={
                "lookahead_m": tgt.lookahead_m,
                "target_x": tgt.x_fwd,
                "target_y_route": tgt.y_left,
                "kappa_ahead": kappa_ahead,
            },
        )
