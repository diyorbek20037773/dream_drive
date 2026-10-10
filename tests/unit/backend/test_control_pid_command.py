"""Speed PID (anti-windup, clamp), pedal mapping, Pure Pursuit geometry and final command shaping."""

from __future__ import annotations

import math

import pytest

from app.control.command import brake_override, command_to_wire, finalize_command
from app.control.pid import SpeedPID, accel_to_pedals
from app.control.pure_pursuit import lookahead_distance, pure_pursuit_steer, target_point
from app.navigation.route import plan_route
from app.simulators.lite.graph import build_town_graph

DT = 0.05


def test_pid_output_is_clamped() -> None:
    pid = SpeedPID()
    assert pid.update(100.0, 0.0, DT) == pid.out_max
    pid.reset()
    assert pid.update(-100.0, 0.0, DT) == pid.out_min


def test_pid_anti_windup_recovers_quickly() -> None:
    """After a long saturated phase the integral must stay bounded, so the controller reacts as soon as the error
    changes sign (a wound-up integrator would keep accelerating for seconds)."""
    pid = SpeedPID()
    for _ in range(400):  # 20 s at full saturation: target far above the measured speed
        out = pid.update(5.0, 0.0, DT)
    assert out == pid.out_max
    assert abs(pid.integral) <= pid.integral_limit
    # target now below the measured speed: output must turn to braking immediately
    out = pid.update(2.0, 3.0, DT)
    assert out < 0.0
    naive_integral = 400 * 5.0 * DT
    assert pid.integral < naive_integral / 10


def test_pid_integral_removes_steady_state_error() -> None:
    pid = SpeedPID()
    v = 0.0
    for _ in range(600):
        a = pid.update(3.0, v, DT)
        thr, brk = accel_to_pedals(a, v)
        v = max(0.0, min(5.0, v + (thr * 2.5 - brk * 6.0 - (0.3 if v > 0 else 0.0)) * DT))
    assert v == pytest.approx(3.0, abs=0.05)


@pytest.mark.parametrize("accel", [-6.0, -2.0, -0.1, 0.0, 0.5, 2.5])
def test_pedals_are_mutually_exclusive(accel: float) -> None:
    thr, brk = accel_to_pedals(accel, 2.0)
    assert 0.0 <= thr <= 1.0 and 0.0 <= brk <= 1.0
    assert thr == 0.0 or brk == 0.0


def test_lookahead_clamp() -> None:
    assert lookahead_distance(0.0) == 3.0
    assert lookahead_distance(2.0) == pytest.approx(3.2)
    assert lookahead_distance(5.0) == pytest.approx(5.0)
    assert lookahead_distance(20.0) == 8.0


def test_pure_pursuit_steer_sign_and_magnitude() -> None:
    assert pure_pursuit_steer(5.0, 0.0) == 0.0
    left = pure_pursuit_steer(5.0, 1.0)
    assert left > 0.0, "target to the left (+y) gives positive (left) steering"
    assert pure_pursuit_steer(5.0, -1.0) == pytest.approx(-left)
    # circle through the target: curvature 2y/L^2, delta = atan(L_wb * k)
    assert left == pytest.approx(math.atan(2.6 * 2 * 1.0 / 26.0) / 0.5)


def test_target_point_in_vehicle_frame() -> None:
    route = plan_route(build_town_graph(), "A", "C")
    tgt = target_point(route, 10.0, -1.75, 0.0, 10.0, 0.0)
    assert tgt.x_fwd == pytest.approx(3.0, abs=1e-6) and tgt.y_left == pytest.approx(0.0, abs=1e-6)
    tgt = target_point(route, 10.0, -1.25, 0.0, 10.0, 0.0)  # ego 0.5 m left of the lane centre
    assert tgt.y_left == pytest.approx(-0.5, abs=1e-6)
    end = target_point(route, route.destination[0], route.destination[1], 0.0, route.length_m, 5.0)
    assert end.x_fwd == pytest.approx(5.0, abs=1e-6), "beyond the route end the final heading is extended"


def test_finalize_in_bounds_has_no_clamp_reason() -> None:
    c = finalize_command(0.2, 0.5, 0.0, 7, ["ROUTE_FOLLOWING"])
    assert (c.steer, c.throttle, c.brake, c.input_frame_id) == (0.2, 0.5, 0.0, 7)
    assert c.reason_codes == ("ROUTE_FOLLOWING",) and not c.safety_override


@pytest.mark.parametrize(
    ("steer", "thr", "brk"), [(1.7, 0.2, 0.0), (-3.0, 0.0, 0.0), (0.0, 1.4, 0.0), (0.0, 0.0, -0.2), (0.0, 0.3, 0.4)]
)
def test_finalize_clamps_and_marks(steer: float, thr: float, brk: float) -> None:
    c = finalize_command(steer, thr, brk, 3, ["ROUTE_FOLLOWING"])
    assert -1.0 <= c.steer <= 1.0 and 0.0 <= c.throttle <= 1.0 and 0.0 <= c.brake <= 1.0
    assert not (c.throttle > 0 and c.brake > 0)
    assert "CONTROL_CLAMPED" in c.reason_codes


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_finalize_rejects_non_finite(bad: float) -> None:
    for args in ((bad, 0.1, 0.0), (0.0, bad, 0.0), (0.0, 0.0, bad)):
        c = finalize_command(*args, 11, ["ROUTE_FOLLOWING"])
        assert (c.steer, c.throttle, c.brake) == (0.0, 0.0, 1.0)
        assert c.safety_override and "CONTROL_NAN_REJECTED" in c.reason_codes
        assert c.input_frame_id == 11


def test_brake_override_and_wire_shape() -> None:
    base = finalize_command(0.3, 0.6, 0.0, 5, ["ROUTE_FOLLOWING"])
    o = brake_override(base, ["CAMERA_BLACK"])
    assert (o.throttle, o.brake, o.safety_override) == (0.0, 1.0, True)
    assert o.reason_codes[0] == "CAMERA_BLACK" and o.steer == pytest.approx(0.3)
    wire = command_to_wire(o)
    assert set(wire) == {"steer", "throttle", "brake", "input_frame_id", "source", "safety_override", "reason_codes"}
    assert wire["source"] == "dreamdrive_controller"
