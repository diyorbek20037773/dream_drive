"""Mandatory LiteRoad closed-loop scenarios (docs/SIMULATION.md, docs/QA.md "Simulation" gate).

Tolerances come from measured runs on the developer CPU (docs/SIMULATION_EVIDENCE.md): normal A->C arrives in
~720-750 ticks, bend D->I in ~740 ticks, mean lateral deviation ~0.03 m, obstacle stops leave ~8 m of free space.
The bounds below leave wide margins so the tests check behaviour, not noise.
"""

from __future__ import annotations

import pytest

NORMAL_TICK_BUDGET = 1000
BEND_TICK_BUDGET = 1000

pytestmark = pytest.mark.timeout(600)


def _summary_ok(session, results) -> dict:  # type: ignore[no-untyped-def]
    s = session.summary()
    assert results[-1].terminal and s["outcome"] == results[-1].outcome
    return s


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_normal_route_arrival(run_session, seed: int) -> None:  # type: ignore[no-untyped-def]
    session, results = run_session("normal", seed)
    s = _summary_ok(session, results)
    assert s["outcome"] == "arrived" and results[-1].run_state == "arrived"
    assert s["ticks"] <= NORMAL_TICK_BUDGET
    ev = s["evaluator"]
    assert ev["collisions"] == 0
    assert ev["route_progress"] >= 0.98
    assert ev["lateral_deviation_mean_m"] < 0.12
    assert ev["lateral_deviation_max_m"] < 0.5
    assert s["commands"]["safety_overrides"] == 0 and s["commands"]["nan_rejected"] == 0
    # every tick carried a pixel perception source and a command tied to its input frame
    for r in results[1:]:
        assert r.world_state.perception.source == "synthetic_rgb_cv"
        assert r.world_state.command is not None
        assert r.world_state.command.input_frame_id == r.frame_id
        assert r.world_state.mode == "simulation" and r.world_state.decision.action_kind == "simulator_control"
    final = results[-1].live
    assert final["decision"]["state"] == "arrived" and "ARRIVED_AT_DESTINATION" in final["decision"]["reason_codes"]
    assert final["ego"]["speed_mps"] < 0.5


def test_bend_route_through_curved_edges(run_session) -> None:  # type: ignore[no-untyped-def]
    session, results = run_session("bend", 1)
    s = _summary_ok(session, results)
    assert s["route"]["node_ids"] == ["D", "H", "I"], "bend scenario uses the curved D-H and H-I edges"
    assert s["outcome"] == "arrived" and s["ticks"] <= BEND_TICK_BUDGET
    ev = s["evaluator"]
    assert ev["collisions"] == 0 and ev["lateral_deviation_mean_m"] < 0.12 and ev["lateral_deviation_max_m"] < 0.5
    assert s["reason_counts"].get("CURVE_SPEED_LIMIT", 0) > 0, "the curve speed limit must actually engage"
    steers = [r.world_state.command.steer for r in results[1:] if r.world_state.command is not None]
    assert max(steers) > 0.15 and min(steers) < -0.15, "S-bend needs both left and right steering"
    assert s["lane_status_counts"].get("valid", 0) > 0.6 * s["ticks"]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_obstacle_controlled_stop(run_session, seed: int) -> None:  # type: ignore[no-untyped-def]
    session, results = run_session("obstacle_stop", seed)
    s = _summary_ok(session, results)
    assert s["outcome"] == "stopped_before_obstacle" and results[-1].run_state == "stopped"
    assert all(r.run_state != "arrived" for r in results)
    ev = s["evaluator"]
    assert ev["collisions"] == 0
    assert ev["final_obstacle_gap_m"] >= 2.0, "stopped with free space in front of the bumper"
    assert ev["route_progress"] < 0.5
    states = [r.live["decision"]["state"] for r in results]
    assert "slow" in states and "controlled_stop" in states
    stop_ticks = [r for r in results if r.live["decision"]["state"] == "controlled_stop"]
    assert all("OBSTACLE_AHEAD_STOP" in r.live["decision"]["reason_codes"] for r in stop_ticks)
    assert all(r.live["perception"]["image_obstacle_cue"] >= 0.45 for r in stop_ticks)
    assert all(r.live["command"]["throttle"] == 0.0 for r in stop_ticks)
    assert results[-1].live["ego"]["speed_mps"] < 0.05
    assert any(r.live["perception"]["obstacle_boxes"] for r in results)


def test_camera_blackout_safety_stop_before_unsafe_stepping(run_session) -> None:  # type: ignore[no-untyped-def]
    session, results = run_session("camera_blackout", 1)
    s = _summary_ok(session, results)
    assert s["outcome"] == "stopped_camera_invalid"
    black = [i for i, r in enumerate(results) if r.live["perception"]["quality_status"] == "missing"]
    assert black, "camera blackout must produce missing-quality frames"
    first = results[black[0]]
    assert first.sim_time_s == pytest.approx(4.0)
    # the very tick that consumed the first black frame already brakes (before the next physics step)
    assert first.live["decision"]["state"] == "safety_stop"
    assert first.live["command"]["brake"] == 1.0 and first.live["command"]["throttle"] == 0.0
    assert first.live["command"]["safety_override"] is True
    assert "CAMERA_BLACK" in first.live["command"]["reason_codes"]
    assert all(r.live["command"]["safety_override"] is False for r in results[1 : black[0]])
    for r in results[black[0] :]:
        assert r.live["command"]["throttle"] == 0.0 and r.live["command"]["brake"] == 1.0
    assert s["evaluator"]["collisions"] == 0


def test_stale_frame_safety_stop(run_session) -> None:  # type: ignore[no-untyped-def]
    session, results = run_session("stale_frame", 1)
    s = _summary_ok(session, results)
    assert s["outcome"] == "stopped_frame_stale"
    stale = [i for i, r in enumerate(results) if r.live["perception"]["quality_status"] == "stale"]
    assert stale
    first = results[stale[0]]
    assert first.frame_id == results[stale[0] - 1].frame_id, "the camera repeated the previous frame id"
    assert first.live["decision"]["state"] == "safety_stop"
    assert "FRAME_STALE" in first.live["decision"]["reason_codes"]
    assert first.live["command"]["brake"] == 1.0 and first.live["command"]["throttle"] == 0.0
    for r in results[stale[0] :]:
        assert r.live["command"]["brake"] == 1.0
