"""Driving FSM: precedence, obstacle hysteresis with dwell, lane-quality slow, latching."""

from __future__ import annotations

from app.agents.fsm import DrivingFsm, FsmInputs

DT = 0.05


def test_precedence_order() -> None:
    full = dict(
        operator_stop=True,
        failure_code="EGO_OUT_OF_BOUNDS",
        safety_codes=("CAMERA_BLACK",),
        paused=True,
        arrived=True,
        obstacle_cue=0.9,
        lane_status="unavailable",
    )
    assert DrivingFsm().step(FsmInputs(0.0, **full)).state == "failed"
    full["failure_code"] = None
    d = DrivingFsm().step(FsmInputs(0.0, **full))
    assert d.state == "safety_stop" and d.reason_codes == ("OPERATOR_STOP",)
    full["operator_stop"] = False
    d = DrivingFsm().step(FsmInputs(0.0, **full))
    assert d.state == "safety_stop" and d.reason_codes == ("CAMERA_BLACK",)
    full["safety_codes"] = ()
    assert DrivingFsm().step(FsmInputs(0.0, **full)).state == "paused"
    full["paused"] = False
    assert DrivingFsm().step(FsmInputs(0.0, **full)).state == "arrived"
    full["arrived"] = False
    assert DrivingFsm().step(FsmInputs(0.0, **full)).state == "controlled_stop"
    full["obstacle_cue"] = 0.4
    d = DrivingFsm().step(FsmInputs(0.0, **full))
    assert d.state == "slow" and "OBSTACLE_AHEAD_SLOW" in d.reason_codes and "LANE_UNCERTAIN_SLOW" in d.reason_codes
    full["obstacle_cue"] = 0.0
    d = DrivingFsm().step(FsmInputs(0.0, **full))
    assert d.state == "slow" and d.reason_codes == ("LANE_UNCERTAIN_SLOW",)
    full["lane_status"] = "valid"
    d = DrivingFsm().step(FsmInputs(0.0, **full))
    assert d.state == "follow_route" and d.reason_codes == ("ROUTE_FOLLOWING",)


def test_obstacle_stop_hysteresis_and_dwell() -> None:
    fsm = DrivingFsm()
    t = 0.0

    def step(cue: float) -> str:
        nonlocal t
        t += DT
        return fsm.step(FsmInputs(t, obstacle_cue=cue)).state

    assert step(0.69) == "slow"  # >= 0.35 enters slow, < 0.70 does not stop
    assert step(0.70) == "controlled_stop"
    assert step(0.50) == "controlled_stop", "0.50 is above the 0.45 exit threshold"
    # cue at the exit threshold must hold for 1.0 s before leaving the stop
    states = [step(0.45) for _ in range(19)]
    assert set(states) == {"controlled_stop"}
    assert step(0.45) == "slow", "after 1.0 s <= 0.45 the stop is released to slow (cue still > 0.25)"
    # a cue spike re-enters the stop immediately
    assert step(0.75) == "controlled_stop"


def test_stop_dwell_resets_when_cue_rises_again() -> None:
    fsm = DrivingFsm()
    t = 0.0
    fsm.step(FsmInputs(t, obstacle_cue=0.8))
    for _ in range(15):
        t += DT
        fsm.step(FsmInputs(t, obstacle_cue=0.3))
    t += DT
    assert fsm.step(FsmInputs(t, obstacle_cue=0.5)).state == "controlled_stop"
    for _ in range(19):
        t += DT
        assert fsm.step(FsmInputs(t, obstacle_cue=0.2)).state == "controlled_stop"
    t += DT
    d = fsm.step(FsmInputs(t, obstacle_cue=0.2))
    assert d.state == "follow_route" and "OBSTACLE_CLEARED" in d.reason_codes


def test_slow_hysteresis() -> None:
    fsm = DrivingFsm()
    t = 0.0
    assert fsm.step(FsmInputs(t, obstacle_cue=0.34)).state == "follow_route"
    t += DT
    assert fsm.step(FsmInputs(t, obstacle_cue=0.35)).state == "slow"
    t += DT
    assert fsm.step(FsmInputs(t, obstacle_cue=0.30)).state == "slow", "between exit 0.25 and enter 0.35 stays slow"
    for _ in range(20):
        t += DT
        d = fsm.step(FsmInputs(t, obstacle_cue=0.25))
    assert d.state == "follow_route" and "OBSTACLE_CLEARED" in d.reason_codes


def test_lane_quality_slow_and_recovery_dwell() -> None:
    fsm = DrivingFsm()
    t = 0.0
    assert fsm.step(FsmInputs(t, obstacle_cue=0.0, lane_status="uncertain")).state == "slow"
    for _ in range(9):
        t += DT
        assert fsm.step(FsmInputs(t, obstacle_cue=0.0, lane_status="valid")).state == "slow"
    t += DT
    assert fsm.step(FsmInputs(t, obstacle_cue=0.0, lane_status="valid")).state == "follow_route"


def test_safety_stop_and_arrival_latch() -> None:
    fsm = DrivingFsm()
    assert fsm.step(FsmInputs(0.0, safety_codes=("FRAME_STALE",))).state == "safety_stop"
    d = fsm.step(FsmInputs(0.05, obstacle_cue=0.0))
    assert d.state == "safety_stop" and d.reason_codes == ("FRAME_STALE",), "safety stop latches"
    fsm2 = DrivingFsm()
    assert fsm2.step(FsmInputs(0.0, arrived=True)).state == "arrived"
    assert fsm2.step(FsmInputs(0.05, arrived=False, obstacle_cue=0.0)).state == "arrived"
    assert fsm2.step(FsmInputs(0.1, operator_stop=True)).state == "safety_stop", "operator stop outranks arrival"
