"""Run.live explainability: maneuver + decision basis in the real LiteRoad closed loop.

- every decision-basis item and the maneuver carry the frame id of the command they explain;
- the basis shows the very values perception/FSM used for that frame (e.g. the obstacle cue);
- evaluator truth never appears in the basis, and an evaluator-only obstacle changes neither basis nor maneuver;
- the obstacle_stop scenario reports SLOW_DOWN/BRAKE before STOP in the maneuver timeline of the run report.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from app.agents.runner import SessionConfig, SimulationSession
from app.schemas.api import RunLive, RunReportResponse
from app.simulators.lite.adapter import LiteRoadAdapter

pytestmark = pytest.mark.timeout(600)

# Inputs measured on the wall clock (watchdogs) legitimately differ between two otherwise identical runs.
WALL_CLOCK_KEYS = {"frame_wall_age_s", "perception_wall_s", "wall_time_s"}
TRUTH_WORDS = ("evaluator", "collision", "truth", "lateral_deviation", "obstacle_gap", "hidden", "route_progress")


def session(adapter: LiteRoadAdapter | None = None, scenario: str = "normal", seed: int = 7) -> SimulationSession:
    return SimulationSession(
        SessionConfig(
            run_id="explain", start_node="A", destination_node="C", scenario=scenario, seed=seed, encode_jpeg=False
        ),
        adapter=adapter,
    )


def comparable_basis(live: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (b["key"], b["value"], b["threshold"], b["comparator"], b["triggered"], b["effect"], b["input_frame_id"])
        for b in live["decision_basis"]
        if b["key"] not in WALL_CLOCK_KEYS
    ]


def test_basis_and_maneuver_match_the_command_frame(run_session: Callable[..., tuple[Any, list[Any]]]) -> None:
    _s, results = run_session("obstacle_stop", seed=1)
    assert results[0].live["maneuver"]["primary"] == "HOLD"  # start(): no command yet
    checked = 0
    for r in results[1:]:
        live = r.live
        RunLive.model_validate(live)  # wire shape is the documented API model
        cmd = live["command"]
        assert cmd is not None
        assert live["maneuver"]["input_frame_id"] == cmd["input_frame_id"] == live["frame_id"]
        assert live["decision_basis"], "every closed-loop tick records its decision basis"
        for item in live["decision_basis"]:
            assert item["input_frame_id"] == cmd["input_frame_id"]
        # the basis shows the same obstacle cue the perception reported for this frame
        cues = [b for b in live["decision_basis"] if b["key"] == "image_obstacle_cue"]
        if cues:
            assert cues[0]["value"] == pytest.approx(live["perception"]["image_obstacle_cue"], abs=1e-6)
            checked += 1
        # safety override on the command <-> safety_override item
        override = [b for b in live["decision_basis"] if b["key"] == "safety_override"]
        assert override and override[0]["value"] is cmd["safety_override"]
    assert checked > 50


def test_triggered_obstacle_cue_explains_slow_and_stop(run_session: Callable[..., tuple[Any, list[Any]]]) -> None:
    _s, results = run_session("obstacle_stop", seed=2)
    seen = set()
    for r in results[1:]:
        state = r.live["decision"]["state"]
        if state not in ("slow", "controlled_stop"):
            continue
        reasons = r.live["decision"]["reason_codes"]
        if "OBSTACLE_AHEAD_SLOW" not in reasons and "OBSTACLE_AHEAD_STOP" not in reasons:
            continue
        triggered = [
            b
            for b in r.live["decision_basis"]
            if b["key"] in ("image_obstacle_cue", "obstacle_exit_dwell_s") and b["triggered"]
        ]
        assert triggered, f"no triggered obstacle input explains {state} at frame {r.frame_id}"
        expected = "slow" if state == "slow" else "controlled_stop"
        assert {b["effect"] for b in triggered} == {expected}
        cue = next((b for b in triggered if b["key"] == "image_obstacle_cue"), None)
        if cue is not None and cue["comparator"] == ">=":
            assert cue["value"] >= cue["threshold"]
        seen.add(state)
    assert seen == {"slow", "controlled_stop"}


def test_basis_never_contains_evaluator_truth_and_ignores_hidden_obstacles() -> None:
    plain, hidden_adapter = LiteRoadAdapter(), LiteRoadAdapter()
    a, b = session(plain), session(hidden_adapter)
    a.start()
    b.start()
    hidden_adapter.add_obstacle(25.0, -1.75, 0.0, visible=False, source="hidden_test")
    allowed_sources = {
        "synthetic_rgb_cv",
        "learned_model",
        "ego_telemetry",
        "route_graph",
        "operator",
        "safety_supervisor",
        "controller",
    }
    n = 0
    while True:
        ra, rb = a.tick(), b.tick()
        n += 1
        for live in (ra.live, rb.live):
            for item in live["decision_basis"]:
                assert item["source"] in allowed_sources
                assert not any(word in item["key"] for word in TRUTH_WORDS), item["key"]
        if ra.terminal or rb.terminal:
            break
        assert comparable_basis(ra.live) == comparable_basis(rb.live), f"basis differs at tick {n}"
        assert ra.live["maneuver"] == rb.live["maneuver"], f"maneuver differs at tick {n}"
    # the evaluator saw the hidden obstacle; the runtime explanation of that last tick did not
    assert rb.outcome == "failed_collision"
    assert not any("COLLISION" in str(item["value"]) for item in rb.live["decision_basis"])
    assert comparable_basis(ra.live) == comparable_basis(rb.live)
    assert ra.live["maneuver"] == rb.live["maneuver"]
    assert n > 50


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_obstacle_stop_timeline_slows_then_stops(run_session: Callable[..., tuple[Any, list[Any]]], seed: int) -> None:
    s, results = run_session("obstacle_stop", seed=seed)
    assert results[-1].outcome == "stopped_before_obstacle"
    timeline = s.summary()["maneuver_timeline"]
    primaries = [e["primary"] for e in timeline]
    assert primaries[0] == "HOLD" and primaries[-1] == "STOP"
    first_stop = primaries.index("STOP")
    assert {"SLOW_DOWN", "BRAKE"} & set(primaries[:first_stop]), primaries
    # changes only: no two consecutive entries share a primary; times and frame ids increase
    assert all(x != y for x, y in zip(primaries, primaries[1:], strict=False))
    times = [e["sim_time_s"] for e in timeline]
    frames = [e["frame_id"] for e in timeline]
    assert times == sorted(times) and frames == sorted(frames)
    stop = timeline[first_stop]
    assert "OBSTACLE_AHEAD_STOP" in stop["reason_codes"] and stop["speed_mps"] <= 0.1
    assert results[-1].live["maneuver"]["primary"] == "STOP"
    report = RunReportResponse.model_validate(s.report("a" * 40, 1.0))
    assert report.details["maneuver_timeline"] == timeline
    assert sum(report.details["maneuver_counts"].values()) == len(results)
    assert any("maneuver_timeline" in text for text in report.limitations)


def test_pause_is_hold_with_operator_basis() -> None:
    s = session()
    s.start()
    for _ in range(40):
        r = s.tick()
    assert r.live["maneuver"]["primary"] in ("ACCELERATE", "CRUISE", "SLOW_DOWN")
    s.pause()
    p = s.tick()
    assert p.live["decision"]["state"] == "paused" and p.live["command"] is None
    assert p.live["maneuver"]["primary"] == "HOLD" and p.live["maneuver"]["signal_source"] == "none"
    pause_items = [b for b in p.live["decision_basis"] if b["key"] == "operator_pause"]
    assert pause_items and pause_items[0]["triggered"] and pause_items[0]["effect"] == "pause"
    assert all(b["input_frame_id"] == p.frame_id for b in p.live["decision_basis"])
    s.resume()
    for _ in range(5):
        r = s.tick()
    assert r.live["maneuver"]["primary"] != "HOLD"


def test_camera_blackout_basis_shows_safety_stop(run_session: Callable[..., tuple[Any, list[Any]]]) -> None:
    _s, results = run_session("camera_blackout", seed=1)
    stops = [r for r in results[1:] if r.live["decision"]["state"] == "safety_stop"]
    assert stops
    first = stops[0].live
    triggered = [b for b in first["decision_basis"] if b["triggered"] and b["effect"] == "safety_stop"]
    assert {b["key"] for b in triggered} >= {"safety_codes", "safety_override"}
    assert results[-1].live["maneuver"]["primary"] == "STOP"
    primaries = [e["primary"] for e in _s.summary()["maneuver_timeline"]]
    assert "BRAKE" in primaries and primaries[-1] == "STOP"


def test_bend_reports_turns(run_session: Callable[..., tuple[Any, list[Any]]]) -> None:
    s, results = run_session("bend", seed=1)
    primaries = {e["primary"] for e in s.summary()["maneuver_timeline"]}
    assert {"TURN_LEFT", "TURN_RIGHT"} & primaries, primaries
    for r in results[1:]:
        man = r.live["maneuver"]
        if man["lateral"] == "turning_left":
            assert man["yaw_rate_radps"] > 0 and r.live["command"]["steer"] > 0
        if man["lateral"] == "turning_right":
            assert man["yaw_rate_radps"] < 0 and r.live["command"]["steer"] < 0
