"""SimulationSession lifecycle: route rejection, modes, pause/resume, operator stop, injections, limits, NaN."""

from __future__ import annotations

import math

import pytest

from app.agents.runner import RouteUnreachable, SessionConfig, SimulationSession, UnsupportedMode
from app.control.controller import ControlOutput

pytestmark = pytest.mark.timeout(300)


def make(scenario: str = "normal", seed: int = 1, start: str = "A", dest: str = "C", **kw) -> SimulationSession:  # type: ignore[no-untyped-def]
    return SimulationSession(
        SessionConfig(run_id="t", start_node=start, destination_node=dest, scenario=scenario, seed=seed, **kw)
    )


def test_unreachable_destination_rejected() -> None:
    with pytest.raises(RouteUnreachable) as exc:
        make(dest="Q")
    assert exc.value.code == "destination_unreachable"
    with pytest.raises(RouteUnreachable):
        SimulationSession.plan_route("A", "Q")
    with pytest.raises(RouteUnreachable) as exc2:
        make(dest="NOPE")
    assert exc2.value.code == "unknown_node"
    route = SimulationSession.plan_route("A", "C")
    assert route["node_ids"] == ["A", "B", "C"] and route["length_m"] == pytest.approx(130.0, abs=0.01)


def test_unsupported_modes() -> None:
    with pytest.raises(UnsupportedMode):
        make(scenario="flying")
    with pytest.raises(UnsupportedMode, match="LEARNED_SIM_MODE_ENABLED"):
        SimulationSession(
            SessionConfig(run_id="t", start_node="A", destination_node="C", perception_source="learned_model"),
            inference_client=object(),
        )
    with pytest.raises(UnsupportedMode):
        make(dt=0.0)
    ids = {s["id"]: s for s in SimulationSession.scenarios()}
    assert set(ids) == {"normal", "bend", "obstacle_stop", "camera_blackout", "stale_frame", "pause_resume"}
    assert ids["pause_resume"]["available"] is False and ids["pause_resume"]["reason"]


def test_start_captures_initial_frame_without_stepping() -> None:
    s = make(encode_jpeg=True)
    r = s.start()
    assert r.frame_id == 0 and r.sim_time_s == 0.0 and not r.terminal
    assert r.run_state == "running" and r.live["decision"]["state"] == "observe"
    assert r.live["decision"]["reason_codes"] == ["INITIALIZING"] and r.live["command"] is None
    assert r.jpeg[:2] == b"\xff\xd8", "TickResult.jpeg is the JPEG of the consumed frame"
    assert r.world_state.frame.quality_status == "valid"
    first = s.tick()
    assert first.frame_id == 0, "the first control step uses the initial frame captured by start()"
    assert first.live["command"]["input_frame_id"] == 0
    assert s.tick().frame_id == 1


def test_pause_resume_freezes_physics() -> None:
    s = make(scenario="pause_resume")
    s.start()
    for _ in range(40):
        r = s.tick()
    assert r.live["ego"]["speed_mps"] > 1.0
    s.pause()
    frozen = [s.tick() for _ in range(15)]
    assert all(f.run_state == "paused" for f in frozen)
    assert all(f.live["decision"]["state"] == "paused" for f in frozen)
    assert all(f.live["decision"]["reason_codes"] == ["OPERATOR_PAUSE"] for f in frozen)
    assert all(f.live["command"] is None for f in frozen)
    poses = {
        (f.live["ego"]["x_m"], f.live["ego"]["y_m"], f.live["ego"]["yaw_rad"], f.sim_time_s, f.frame_id) for f in frozen
    }
    assert len(poses) == 1, "no physics step and no new frame while paused"
    s.resume()
    after = s.tick()
    assert after.run_state == "running" and after.frame_id == frozen[0].frame_id + 1
    assert after.live["perception"]["quality_status"] == "valid", "a synchronous pause must not age the frame"
    later = [s.tick() for _ in range(5)]
    assert later[-1].sim_time_s > after.sim_time_s and later[-1].live["ego"]["x_m"] > after.live["ego"]["x_m"]


def test_operator_stop_brakes_to_standstill() -> None:
    s = make()
    s.start()
    for _ in range(60):
        r = s.tick()
    assert r.live["ego"]["speed_mps"] > 2.0
    s.request_stop()
    r = s.tick()
    assert r.run_state == "stopping"
    assert r.live["decision"]["state"] == "safety_stop" and r.live["decision"]["reason_codes"] == ["OPERATOR_STOP"]
    assert r.live["command"]["brake"] == 1.0 and r.live["command"]["safety_override"] is True
    n = 0
    while not r.terminal and n < 100:
        r = s.tick()
        n += 1
    assert r.terminal and r.run_state == "stopped" and r.outcome == "stopped_operator"
    assert n < 40, "standstill within a bounded number of ticks (5 m/s / 6 m/s^2 < 1 s)"
    again = s.tick()
    assert again is r, "stop/tick after terminal is idempotent"
    s.request_stop()
    assert s.run_state == "stopped"


def test_operator_stop_while_paused() -> None:
    s = make()
    s.start()
    for _ in range(30):
        s.tick()
    s.pause()
    s.tick()
    s.request_stop()
    r = s.tick()
    assert r.run_state in ("stopping", "stopped") and r.live["decision"]["state"] == "safety_stop"


def test_injected_obstacle_causes_controlled_stop() -> None:
    s = make()
    r = s.start()
    for _ in range(20):
        r = s.tick()
    assert s.inject_obstacle(20.0, 0.0) is True
    assert s.inject_obstacle(500.0, 0.0) is False, "beyond the route end"
    while not r.terminal:
        r = s.tick()
    summ = s.summary()
    assert r.outcome == "stopped_before_obstacle" and summ["evaluator"]["collisions"] == 0
    assert summ["evaluator"]["injected_obstacles"] == 1
    assert s.inject_obstacle(20.0, 0.0) is False, "no injection after the run ended"


def test_injection_limit() -> None:
    s = make()
    s.start()
    assert [s.inject_obstacle(30.0 + 5 * i, 1.4) for i in range(6)] == [True] * 5 + [False]


def test_tick_and_wall_limits() -> None:
    s = make(max_ticks=10)
    r = s.start()
    while not r.terminal:
        r = s.tick()
    assert r.outcome == "failed_tick_limit" and r.run_state == "failed"
    assert r.live["decision"]["reason_codes"] == ["TICK_LIMIT"] and r.live["command"]["brake"] == 1.0

    s2 = make()
    s2.start()
    for _ in range(5):
        s2.tick()
    pose_before = s2.tick().live["ego"]
    s2.note_wall_time(1e6)
    r2 = s2.tick()
    assert r2.terminal and r2.outcome == "failed_wall_time"
    assert r2.live["command"]["brake"] == 1.0 and r2.live["command"]["throttle"] == 0.0
    assert s2.summary()["physics_steps"] == 6, "the watchdog stops stepping"
    assert pose_before is not None


def test_ego_out_of_bounds_fails_run() -> None:
    s = make()
    s.start()
    s._adapter.spawn_ego(30.0, -15.0, 0.0, 2.0)  # noqa: SLF001 - scene manipulation for the test
    r = s.tick()
    assert r.terminal and r.outcome == "failed_out_of_bounds"
    assert r.live["decision"]["reason_codes"] == ["EGO_OUT_OF_BOUNDS"] and r.live["command"]["brake"] == 1.0


def test_nan_control_is_rejected_in_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    s = make()
    s.start()
    for _ in range(30):
        s.tick()
    real = s._controller.compute  # noqa: SLF001

    def broken(*args, **kwargs):  # type: ignore[no-untyped-def]
        out = real(*args, **kwargs)
        return ControlOutput(
            steer=math.nan, throttle=out.throttle, brake=out.brake, target_speed_mps=0.0, reason_codes=[]
        )

    monkeypatch.setattr(s._controller, "compute", broken)  # noqa: SLF001
    r = s.tick()
    cmd = r.live["command"]
    assert "CONTROL_NAN_REJECTED" in cmd["reason_codes"]
    assert (cmd["steer"], cmd["throttle"], cmd["brake"], cmd["safety_override"]) == (0.0, 0.0, 1.0, True)
    assert s.summary()["commands"]["nan_rejected"] == 1


def test_report_requires_terminal_and_valid_commit() -> None:
    s = make(max_ticks=5)
    r = s.start()
    with pytest.raises(RuntimeError, match="report_unavailable"):
        s.report("a" * 40, 1.0)
    while not r.terminal:
        r = s.tick()
    with pytest.raises(ValueError):
        s.report("unknown", 1.0)
