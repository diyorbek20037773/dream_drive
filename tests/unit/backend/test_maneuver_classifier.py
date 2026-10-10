"""Maneuver classifier (app.control.maneuver): signs, hysteresis, confirmation and paused hold.

The classifier only describes the applied command + ego telemetry; these tests drive it with synthetic telemetry
sequences at the LiteRoad tick (dt = 0.05 s).
"""

from __future__ import annotations

import math
from typing import get_args

import pytest

from app.control import maneuver as m
from app.control.maneuver import AppliedCommand, ManeuverClassifier, primary_for
from app.schemas import api as api_schemas
from app.schemas.events import DecisionManeuver

DT = 0.05


def drive(
    clf: ManeuverClassifier,
    t0: float,
    n: int,
    v0: float,
    accel: float,
    yaw0: float = 0.0,
    yaw_rate: float = 0.0,
    cmd: AppliedCommand | None = AppliedCommand(0.0, 0.0, 0.0),
    target: float | None = 4.8,
) -> tuple[list[m.Maneuver], float, float, float]:
    """Feed n ticks of constant acceleration / yaw rate; returns (maneuvers, t, v, yaw) after the last tick."""
    out = []
    t, v, yaw = t0, v0, yaw0
    for i in range(n):
        out.append(clf.update(t, v, yaw, cmd, target, i))
        t += DT
        v = max(0.0, v + accel * DT)
        yaw = math.remainder(yaw + yaw_rate * DT, math.tau)
    return out, t, v, yaw


def test_codes_match_api_literals() -> None:
    assert set(get_args(api_schemas.ManeuverPrimary.__value__)) == set(m.PRIMARY_CODES)
    assert set(get_args(api_schemas.ManeuverLongitudinal.__value__)) == set(m.LONGITUDINAL_CODES)
    assert set(get_args(api_schemas.ManeuverLateral.__value__)) == set(m.LATERAL_CODES)
    assert set(get_args(api_schemas.ManeuverSignalSource.__value__)) == set(m.SIGNAL_SOURCES)


def test_wire_shape_validates_against_api_and_event_models() -> None:
    clf = ManeuverClassifier(DT)
    wire = clf.update(0.0, 0.0, 0.0, None, None, 3).to_wire()
    model = api_schemas.LiveManeuver.model_validate(wire)
    assert model.primary == "HOLD" and model.input_frame_id == 3 and model.signal_source == "none"
    DecisionManeuver.model_validate({k: wire[k] for k in ("primary", "longitudinal", "lateral", "since_sim_time_s")})


def test_precedence() -> None:
    assert primary_for("holding", "turning_left") == "HOLD"
    assert primary_for("stopped", "turning_left") == "STOP"
    assert primary_for("braking_to_stop", "turning_right") == "BRAKE"
    assert primary_for("decelerating", "turning_left") == "TURN_LEFT"
    assert primary_for("accelerating", "turning_right") == "TURN_RIGHT"
    assert primary_for("decelerating", "straight") == "SLOW_DOWN"
    assert primary_for("accelerating", "straight") == "ACCELERATE"
    assert primary_for("cruising", "straight") == "CRUISE"


def test_paused_or_no_command_is_hold_immediately() -> None:
    clf = ManeuverClassifier(DT)
    out, t, v, yaw = drive(clf, 0.0, 30, 2.0, 1.0, cmd=AppliedCommand(0.0, 0.6, 0.0))
    assert out[-1].primary == "ACCELERATE"
    # operator pause: no command applied, simulated time does not advance, the vehicle keeps its frozen speed
    held = clf.update(t, v, yaw, None, None, 99)
    assert (held.primary, held.longitudinal, held.signal_source) == ("HOLD", "holding", "none")
    assert held.accel_mps2 is None and held.yaw_rate_radps is None
    assert held.since_sim_time_s == pytest.approx(t)
    again = clf.update(t, v, yaw, None, None, 99)
    assert again.primary == "HOLD" and again.since_sim_time_s == pytest.approx(t)


def test_accelerate_from_standstill_uses_command_model_then_telemetry() -> None:
    clf = ManeuverClassifier(DT)
    first = clf.update(0.0, 0.0, 0.0, AppliedCommand(0.0, 1.0, 0.0), 4.8, 0)
    assert first.signal_source == "command_model" and first.accel_mps2 == pytest.approx(2.5)
    assert first.longitudinal == "accelerating"
    assert first.primary == "ACCELERATE"  # first label of a run is shown at once
    out, *_ = drive(clf, DT, 10, 0.125, 2.5, cmd=AppliedCommand(0.0, 1.0, 0.0))
    assert out[-1].signal_source == "ego_telemetry_window"
    assert out[-1].accel_mps2 == pytest.approx(2.5, abs=1e-6)
    assert {x.primary for x in out} == {"ACCELERATE"}


def test_cruise_and_decelerate_hysteresis() -> None:
    clf = ManeuverClassifier(DT)
    out, t, v, _ = drive(clf, 0.0, 40, 4.8, 0.0, cmd=AppliedCommand(0.0, 0.12, 0.0))
    assert out[-1].primary == "CRUISE" and out[-1].longitudinal == "cruising"
    # coasting at -0.3 m/s^2 (rolling drag, no pedal) is decelerating once the measured window shows it
    out, t, v, _ = drive(clf, t, 20, v, -0.3, cmd=AppliedCommand(0.0, 0.0, 0.0))
    assert out[-1].longitudinal == "decelerating" and out[-1].primary == "SLOW_DOWN"
    # inside the hysteresis band (-0.10 .. -0.25 m/s^2) the label is kept ...
    out, t, v, _ = drive(clf, t, 20, v, -0.15, cmd=AppliedCommand(0.0, 0.0, 0.0))
    assert out[-1].longitudinal == "decelerating"
    # ... and released only once the deceleration is weaker than the exit threshold
    out, t, v, _ = drive(clf, t, 20, v, 0.0, cmd=AppliedCommand(0.0, 0.12, 0.0))
    assert out[-1].longitudinal == "cruising" and out[-1].primary == "CRUISE"


def test_brake_pedal_with_positive_target_is_slow_down() -> None:
    clf = ManeuverClassifier(DT)
    drive(clf, 0.0, 20, 4.8, 0.0, cmd=AppliedCommand(0.0, 0.12, 0.0))
    out, *_ = drive(clf, 1.0, 6, 4.8, -1.0, cmd=AppliedCommand(0.0, 0.0, 0.2), target=2.5)
    assert all(x.longitudinal == "decelerating" for x in out)
    assert out[-1].primary == "SLOW_DOWN"


def test_braking_to_stop_then_stopped_are_immediate() -> None:
    clf = ManeuverClassifier(DT)
    out, t, v, yaw = drive(clf, 0.0, 30, 4.8, 0.0, cmd=AppliedCommand(0.0, 0.12, 0.0))
    assert out[-1].primary == "CRUISE"
    brake = clf.update(t, v, yaw, AppliedCommand(0.0, 0.0, 0.8), 0.0, 31)
    assert (brake.primary, brake.longitudinal) == ("BRAKE", "braking_to_stop")
    assert brake.since_sim_time_s == pytest.approx(t)
    out, t, v, yaw = drive(clf, t + DT, 20, v, -5.0, cmd=AppliedCommand(0.0, 0.0, 0.8), target=0.0)
    assert v == 0.0
    stop = clf.update(t, 0.0, yaw, AppliedCommand(0.0, 0.0, 0.4), 0.0, 60)
    assert (stop.primary, stop.longitudinal) == ("STOP", "stopped")
    # a safety override brake counts as braking to a stop even when the controller target was positive
    clf2 = ManeuverClassifier(DT)
    drive(clf2, 0.0, 10, 3.0, 0.0, cmd=AppliedCommand(0.0, 0.1, 0.0))
    ov = clf2.update(0.5, 3.0, 0.0, AppliedCommand(0.0, 0.0, 1.0, safety_override=True), 4.8, 11)
    assert ov.primary == "BRAKE"


def test_turn_signs_follow_steer_convention() -> None:
    """+steer = left = counter-clockwise (positive) yaw rate; - = right."""
    left = ManeuverClassifier(DT)
    out, *_ = drive(left, 0.0, 20, 4.0, 0.0, yaw_rate=0.2, cmd=AppliedCommand(0.3, 0.1, 0.0))
    assert out[-1].lateral == "turning_left" and out[-1].primary == "TURN_LEFT"
    assert out[-1].yaw_rate_radps == pytest.approx(0.2, abs=1e-6)
    right = ManeuverClassifier(DT)
    out, *_ = drive(right, 0.0, 20, 4.0, 0.0, yaw_rate=-0.2, cmd=AppliedCommand(-0.3, 0.1, 0.0))
    assert out[-1].lateral == "turning_right" and out[-1].primary == "TURN_RIGHT"
    # yaw wrap-around at +-pi does not produce a spurious turn
    wrap = ManeuverClassifier(DT)
    out, *_ = drive(wrap, 0.0, 20, 4.0, 0.0, yaw0=math.pi - 0.01, yaw_rate=-0.2, cmd=AppliedCommand(-0.3, 0.1, 0.0))
    assert out[-1].lateral == "turning_right"


def test_turn_requires_agreeing_steer_and_motion() -> None:
    clf = ManeuverClassifier(DT)
    out, *_ = drive(clf, 0.0, 20, 4.0, 0.0, yaw_rate=0.2, cmd=AppliedCommand(-0.05, 0.1, 0.0))
    assert out[-1].lateral == "straight", "yaw left while steering right does not enter a left turn"
    still = ManeuverClassifier(DT)
    out, *_ = drive(still, 0.0, 20, 0.0, 0.0, cmd=AppliedCommand(0.8, 0.0, 0.4), target=0.0)
    assert out[-1].lateral == "straight" and out[-1].primary == "STOP", "steering at standstill is not a turn"


def test_turn_hysteresis_band() -> None:
    clf = ManeuverClassifier(DT)
    # 0.06 rad/s is inside the band: not enough to enter a turn from straight
    out, t, v, yaw = drive(clf, 0.0, 20, 4.0, 0.0, yaw_rate=0.06, cmd=AppliedCommand(0.1, 0.1, 0.0))
    assert out[-1].lateral == "straight"
    out, t, v, yaw = drive(clf, t, 20, v, 0.0, yaw0=yaw, yaw_rate=0.15, cmd=AppliedCommand(0.3, 0.1, 0.0))
    assert out[-1].lateral == "turning_left"
    # ... but enough to stay in a turn that was entered
    out, t, v, yaw = drive(clf, t, 20, v, 0.0, yaw0=yaw, yaw_rate=0.06, cmd=AppliedCommand(0.1, 0.1, 0.0))
    assert out[-1].lateral == "turning_left"
    out, t, v, yaw = drive(clf, t, 20, v, 0.0, yaw0=yaw, yaw_rate=0.0, cmd=AppliedCommand(0.0, 0.1, 0.0))
    assert out[-1].lateral == "straight"


def test_short_blips_do_not_flicker_the_primary() -> None:
    clf = ManeuverClassifier(DT)
    out, t, v, yaw = drive(clf, 0.0, 30, 4.8, 0.0, cmd=AppliedCommand(0.0, 0.12, 0.0))
    assert out[-1].primary == "CRUISE"
    # one tick of brake pedal with a positive target: candidate SLOW_DOWN is not confirmed
    blip = clf.update(t, v, yaw, AppliedCommand(0.0, 0.0, 0.05), 4.8, 31)
    assert blip.longitudinal == "decelerating" and blip.primary == "CRUISE"
    back = clf.update(t + DT, v, yaw, AppliedCommand(0.0, 0.12, 0.0), 4.8, 32)
    assert back.primary == "CRUISE"
    # held for CONFIRM_S it switches, with the start time of the candidate
    t1 = t + 2 * DT
    labels = [clf.update(t1 + i * DT, v, yaw, AppliedCommand(0.0, 0.0, 0.05), 4.8, 33 + i) for i in range(3)]
    assert [x.primary for x in labels] == ["CRUISE", "CRUISE", "SLOW_DOWN"]
    assert labels[-1].since_sim_time_s == pytest.approx(t1)


def test_invalid_dt_rejected() -> None:
    with pytest.raises(ValueError):
        ManeuverClassifier(0.0)
