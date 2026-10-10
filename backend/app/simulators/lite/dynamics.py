"""Kinematic bicycle ego model (docs/IMPLEMENTATION_SPEC.md section 4).

Reference point: rear-axle centre (the "ego origin"). ENU world frame, yaw CCW from +x, metres / seconds.
steer +1 = full left = +0.5 rad front-wheel angle = positive yaw rate when moving forward.
Longitudinal: a = throttle * 2.5 - brake * 6.0 - rolling drag 0.3 (only while moving); speed clamped to [0, 5].
Integration per fixed step: position uses the midpoint heading of the step (second-order accurate in yaw).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

WHEELBASE_M = 2.6
MAX_STEER_RAD = 0.5
MAX_SPEED_MPS = 5.0
MAX_ACCEL_MPS2 = 2.5
MAX_BRAKE_DECEL_MPS2 = 6.0
ROLLING_DRAG_MPS2 = 0.3
# Footprint relative to the rear axle: 0.9 m rear overhang, 2.6 m wheelbase, 1.0 m front overhang.
EGO_LENGTH_M = 4.5
EGO_WIDTH_M = 1.8
EGO_REAR_OVERHANG_M = 0.9
EGO_FRONT_FROM_ORIGIN_M = EGO_LENGTH_M - EGO_REAR_OVERHANG_M  # 3.6 m


@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float
    speed: float


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def step_bicycle(state: VehicleState, steer: float, throttle: float, brake: float, dt: float) -> VehicleState:
    """Advance one fixed step. Inputs are clamped to their physical ranges (the controller clamps earlier)."""
    if not all(math.isfinite(v) for v in (steer, throttle, brake, dt)):
        raise ValueError("non-finite actuator input reached the vehicle model")
    delta = _clamp(steer, -1.0, 1.0) * MAX_STEER_RAD
    throttle = _clamp(throttle, 0.0, 1.0)
    brake = _clamp(brake, 0.0, 1.0)
    v = state.speed
    yaw_rate = v / WHEELBASE_M * math.tan(delta)
    yaw_mid = state.yaw + 0.5 * yaw_rate * dt
    x = state.x + v * math.cos(yaw_mid) * dt
    y = state.y + v * math.sin(yaw_mid) * dt
    yaw = math.remainder(state.yaw + yaw_rate * dt, math.tau)
    accel = throttle * MAX_ACCEL_MPS2 - brake * MAX_BRAKE_DECEL_MPS2 - (ROLLING_DRAG_MPS2 if v > 0 else 0.0)
    speed = _clamp(v + accel * dt, 0.0, MAX_SPEED_MPS)
    return VehicleState(x=x, y=y, yaw=yaw, speed=speed)


def footprint(state: VehicleState) -> list[tuple[float, float]]:
    """Ego rectangle corners (CCW from above) in the world frame."""
    c, s = math.cos(state.yaw), math.sin(state.yaw)
    hw = EGO_WIDTH_M / 2
    local = [
        (EGO_FRONT_FROM_ORIGIN_M, hw),
        (-EGO_REAR_OVERHANG_M, hw),
        (-EGO_REAR_OVERHANG_M, -hw),
        (EGO_FRONT_FROM_ORIGIN_M, -hw),
    ]
    return [(state.x + lx * c - ly * s, state.y + lx * s + ly * c) for lx, ly in local]
