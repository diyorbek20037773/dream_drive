"""LiteRoad kinematic bicycle: units, ENU coordinate and yaw/steer sign conventions (docs/IMPLEMENTATION_SPEC.md 4)."""

from __future__ import annotations

import math

import pytest

from app.simulators.lite.dynamics import (
    EGO_FRONT_FROM_ORIGIN_M,
    MAX_SPEED_MPS,
    VehicleState,
    footprint,
    step_bicycle,
)

DT = 0.05


def drive(state: VehicleState, steer: float, throttle: float, brake: float, seconds: float) -> VehicleState:
    for _ in range(int(round(seconds / DT))):
        state = step_bicycle(state, steer, throttle, brake, DT)
    return state


def test_heading_east_moves_plus_x() -> None:
    s = drive(VehicleState(0.0, 0.0, 0.0, 3.0), 0.0, 0.3 / 2.5, 0.0, 1.0)  # throttle exactly cancels drag
    assert s.x == pytest.approx(3.0, abs=1e-6)
    assert s.y == pytest.approx(0.0, abs=1e-9)
    assert s.speed == pytest.approx(3.0, abs=1e-9)


def test_heading_north_moves_plus_y() -> None:
    s = drive(VehicleState(0.0, 0.0, math.pi / 2, 2.0), 0.0, 0.0, 0.0, 0.5)
    assert s.y > 0.9 and abs(s.x) < 1e-9


def test_positive_steer_is_left_turn_ccw() -> None:
    s = drive(VehicleState(0.0, 0.0, 0.0, 3.0), 1.0, 0.12, 0.0, 1.0)
    assert s.yaw > 0.0, "steer +1 must give a positive (CCW) yaw rate"
    assert s.y > 0.0, "a left turn while heading east moves north (+y)"
    expected_rate = 3.0 / 2.6 * math.tan(0.5)
    assert s.yaw == pytest.approx(expected_rate * 1.0, rel=0.02)


def test_negative_steer_is_right_turn() -> None:
    s = drive(VehicleState(0.0, 0.0, 0.0, 3.0), -1.0, 0.12, 0.0, 1.0)
    assert s.yaw < 0.0 and s.y < 0.0


def test_longitudinal_limits_and_units() -> None:
    s = drive(VehicleState(0.0, 0.0, 0.0, 0.0), 0.0, 1.0, 0.0, 1.0)
    # 2.5 m/s^2 minus 0.3 m/s^2 rolling drag once moving (the first step starts from rest)
    assert s.speed == pytest.approx(2.5 - 0.3 * (1.0 - DT), abs=1e-6)
    s = drive(s, 0.0, 1.0, 0.0, 10.0)
    assert s.speed == pytest.approx(MAX_SPEED_MPS)
    s2 = drive(VehicleState(0.0, 0.0, 0.0, 5.0), 0.0, 0.0, 1.0, 0.5)
    assert s2.speed == pytest.approx(5.0 - (6.0 + 0.3) * 0.5, abs=1e-6)
    s3 = drive(VehicleState(0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 1.0, 2.0)
    assert s3.speed == 0.0, "speed never becomes negative"


def test_inputs_are_clamped_and_nan_rejected() -> None:
    a = step_bicycle(VehicleState(0, 0, 0, 3.0), 5.0, 9.0, 0.0, DT)
    b = step_bicycle(VehicleState(0, 0, 0, 3.0), 1.0, 1.0, 0.0, DT)
    assert (a.x, a.y, a.yaw, a.speed) == (b.x, b.y, b.yaw, b.speed)
    with pytest.raises(ValueError):
        step_bicycle(VehicleState(0, 0, 0, 3.0), float("nan"), 0.0, 0.0, DT)


def test_footprint_front_is_ahead_of_origin() -> None:
    fp = footprint(VehicleState(10.0, 5.0, math.pi / 2, 0.0))
    ys = [p[1] for p in fp]
    xs = [p[0] for p in fp]
    assert max(ys) == pytest.approx(5.0 + EGO_FRONT_FROM_ORIGIN_M)
    assert min(ys) == pytest.approx(5.0 - 0.9)
    assert max(xs) - min(xs) == pytest.approx(1.8)
