"""Decision-basis records of the FSM, safety supervisor and controller come from their own evaluation (thresholds,
hysteresis state, gates) and stop at the precedence level that decided."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.agents.fsm import SLOW_ENTER, SLOW_EXIT, STOP_ENTER, STOP_EXIT, DrivingFsm, FsmInputs
from app.agents.safety import SafetyLimits, SafetySupervisor
from app.control.basis import BasisItem
from app.control.controller import Controller
from app.navigation.route import RoadSurface, cached_route
from app.simulators.base import EgoState, RgbFrame
from app.simulators.lite.graph import JUNCTION_HALF_SIZE_M, build_town_graph, junction_nodes

DT = 0.05


def by_key(items: tuple[BasisItem, ...], key: str) -> BasisItem:
    found = [i for i in items if i.key == key]
    assert found, f"{key} missing from {[i.key for i in items]}"
    return found[0]


def test_fsm_obstacle_cue_enter_hold_and_exit_dwell() -> None:
    fsm = DrivingFsm(DT)
    d = fsm.step(FsmInputs(sim_time_s=0.0, obstacle_cue=0.1))
    cue = by_key(fsm.last_basis, "image_obstacle_cue")
    assert d.state == "follow_route"
    assert (cue.value, cue.threshold, cue.comparator) == (0.1, SLOW_ENTER, ">=")
    assert (cue.triggered, cue.effect) == (False, "slow")

    d = fsm.step(FsmInputs(sim_time_s=0.05, obstacle_cue=0.52))
    cue = by_key(fsm.last_basis, "image_obstacle_cue")
    assert d.state == "slow" and (cue.threshold, cue.comparator, cue.triggered) == (SLOW_ENTER, ">=", True)

    d = fsm.step(FsmInputs(sim_time_s=0.10, obstacle_cue=0.75))
    cue = by_key(fsm.last_basis, "image_obstacle_cue")
    assert d.state == "controlled_stop"
    assert (cue.threshold, cue.triggered, cue.effect) == (STOP_ENTER, True, "controlled_stop")

    # held by hysteresis: the shown threshold is the stop exit threshold
    d = fsm.step(FsmInputs(sim_time_s=0.15, obstacle_cue=0.6))
    cue = by_key(fsm.last_basis, "image_obstacle_cue")
    assert d.state == "controlled_stop" and (cue.threshold, cue.comparator, cue.triggered) == (STOP_EXIT, ">", True)

    # below the exit threshold: the dwell timer holds the stop until 1.0 s has elapsed
    t = 0.20
    d = fsm.step(FsmInputs(sim_time_s=t, obstacle_cue=0.1))
    dwell = by_key(fsm.last_basis, "obstacle_exit_dwell_s")
    assert d.state == "controlled_stop" and dwell.triggered and dwell.threshold == 1.0 and dwell.unit == "s"
    assert dwell.value == pytest.approx(DT)
    for _ in range(19):
        t += DT
        d = fsm.step(FsmInputs(sim_time_s=t, obstacle_cue=0.1))
    dwell = by_key(fsm.last_basis, "obstacle_exit_dwell_s")
    assert d.state == "follow_route" and "OBSTACLE_CLEARED" in d.reason_codes
    assert dwell.value == pytest.approx(1.0) and not dwell.triggered
    cue = by_key(fsm.last_basis, "image_obstacle_cue")
    assert not cue.triggered and cue.threshold in (SLOW_EXIT, SLOW_ENTER, STOP_EXIT)


def test_fsm_basis_stops_at_the_deciding_precedence_level() -> None:
    fsm = DrivingFsm(DT)
    fsm.step(FsmInputs(sim_time_s=0.0, safety_codes=("CAMERA_BLACK",), obstacle_cue=0.9))
    keys = [i.key for i in fsm.last_basis]
    assert keys == ["failure_code", "operator_stop", "safety_codes"], "obstacle cue was not evaluated"
    latched = by_key(fsm.last_basis, "safety_codes")
    assert latched.value == "CAMERA_BLACK" and latched.triggered and latched.effect == "safety_stop"

    fsm2 = DrivingFsm(DT)
    fsm2.step(FsmInputs(sim_time_s=0.0, paused=True))
    assert [i.key for i in fsm2.last_basis][-1] == "operator_pause"
    assert by_key(fsm2.last_basis, "operator_pause").triggered


def test_fsm_lane_status_and_recovery_dwell() -> None:
    fsm = DrivingFsm(DT)
    d = fsm.step(FsmInputs(sim_time_s=0.0, lane_status="uncertain", perception_source="learned_model"))
    lane = by_key(fsm.last_basis, "lane_status")
    assert d.state == "slow" and lane.triggered and lane.effect == "slow" and lane.source == "learned_model"
    d = fsm.step(FsmInputs(sim_time_s=0.05, lane_status="valid"))
    dwell = by_key(fsm.last_basis, "lane_valid_dwell_s")
    assert d.state == "slow" and dwell.triggered and dwell.threshold == 0.5


def test_safety_frame_checks_record_compared_values() -> None:
    graph = build_town_graph()
    sup = SafetySupervisor(SafetyLimits(), RoadSurface(graph, JUNCTION_HALF_SIZE_M, junction_nodes(graph)))
    img = np.zeros((4, 4, 3), np.uint8)
    ok = sup.check_frame(RgbFrame(5, 0.25, img, 100.0), 4, 0.25, DT, 100.2)
    assert ok.ok and [i.key for i in ok.basis] == ["frame_received", "frame_id", "frame_sim_age_s", "frame_wall_age_s"]
    wall = by_key(ok.basis, "frame_wall_age_s")
    assert wall.value == pytest.approx(0.2) and wall.threshold == 1.0 and not wall.triggered
    stale = sup.check_frame(RgbFrame(5, 0.25, img, 100.0), 4, 0.25, DT, 102.0)
    assert not stale.ok and by_key(stale.basis, "frame_wall_age_s").triggered
    missing = sup.check_frame(None, 4, 0.25, DT, 100.0)
    assert [i.key for i in missing.basis] == ["frame_received"] and missing.basis[0].triggered
    sup.check_perception("valid", (), "valid", 0.0, 0.2, 7.0)
    timeout = by_key(sup.last_perception_basis, "perception_wall_s")
    assert timeout.triggered and timeout.threshold == 5.0 and timeout.effect == "safety_stop"


def test_controller_basis_curve_limit_and_pixel_gate() -> None:
    graph = build_town_graph()
    route = cached_route(graph, "A", "C")
    ctl = Controller(DT)
    x, y = route.point_at(0.0)
    yaw = route.heading_at(0.0)
    out = ctl.compute("follow_route", EgoState(x, y, yaw, 4.0), route, 0.0, "unavailable", None)
    keys = [i.key for i in out.basis]
    assert "pixel_blend_weight" in keys and by_key(out.basis, "pixel_blend_weight").value == 0.0
    target = by_key(out.basis, "target_speed_mps")
    assert target.value == pytest.approx(out.target_speed_mps) and target.source == "controller"
    curve = by_key(out.basis, "curve_speed_limit_mps")
    assert curve.triggered == out.curve_limited and curve.effect == "speed_limit"
    if curve.value is not None and math.isfinite(float(curve.value)):
        assert curve.threshold == 4.8
    stop = ctl.compute("controlled_stop", EgoState(x, y, yaw, 4.0), route, 0.0, "valid", None)
    assert by_key(stop.basis, "target_speed_mps").value == 0.0
    assert "curve_speed_limit_mps" not in [i.key for i in stop.basis], "stop states do not evaluate speed limits"


def test_basis_item_validates_vocabulary() -> None:
    with pytest.raises(ValueError):
        BasisItem("x", 1.0, "evaluator")
    with pytest.raises(ValueError):
        BasisItem("x", 1.0, "controller", effect="teleport")
    wire = BasisItem("x", math.inf, "controller", "m/s", math.nan, ">").to_wire(9)
    assert wire["value"] is None and wire["threshold"] is None and wire["input_frame_id"] == 9
