# ruff: noqa: F811, S105, S106 - pytest fixtures are imported from the shared harness; test-only passwords
"""Simulation job loop (app.jobs.sim_job) driving a fake SimulationSession: pacing, controls, injections, live
frame publication, events, watchdog and persisted final state/report."""

from __future__ import annotations

import json
import time
from typing import Any

import pytest
from test_api_harness import FakeSession, Harness, harness, run_in_thread  # noqa: F401

from app.jobs import sim_job
from app.storage import repo

RUN = {"start_node": "A", "destination_node": "C", "scenario": "normal", "seed": 7}


def _create(h: Harness) -> tuple[Any, dict[str, Any], str]:
    client, info = h.login()
    response = client.post("/api/v1/runs", json=RUN, headers=h.csrf(info))
    assert response.status_code == 202, response.text
    return client, info, response.json()["id"]


def _state(h: Harness, run_id: str) -> str:
    row = repo.get_run_any(h.services.db, run_id)
    assert row is not None
    return row.state


@pytest.fixture(autouse=True)
def fake_session_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    def create(row: repo.RunRow, ctx: Any) -> tuple[Any, Any]:
        cfg = type("Cfg", (), {"run_id": row.id, "dt": ctx.settings.sim_fixed_dt_seconds, "scenario": row.scenario})()
        return FakeSession(cfg), None

    monkeypatch.setattr(sim_job, "create_session", create)


def test_run_to_arrival_persists_report_and_live(harness: Harness) -> None:
    client, _info, run_id = _create(harness)
    started = time.monotonic()
    assert sim_job.run_simulation_job(run_id) == "arrived"
    elapsed = time.monotonic() - started
    # Pacing: 8 ticks of 0.01 s simulated time cannot finish faster than 0.08 s of wall time.
    assert elapsed >= 0.07
    run = client.get(f"/api/v1/runs/{run_id}").json()
    assert run["state"] == "arrived" and run["outcome"] == "arrived" and run["finished_utc"] and run["started_utc"]
    assert run["live"]["frame_id"] == 8 and run["live"]["decision"]["reason_codes"] == ["ARRIVED_AT_DESTINATION"]
    assert run["live"]["perception"]["source"] == "synthetic_rgb_cv"
    report = client.get(f"/api/v1/runs/{run_id}/report")
    assert report.status_code == 200, report.text
    body = report.json()
    assert body["run_id"] == run_id and body["kind"] == "simulation" and body["status"] == "arrived"
    assert body["runtime_used_dynamic_ground_truth"] is False and body["details"]["wall_time_s"] > 0
    frame = client.get(f"/api/v1/runs/{run_id}/frame")
    assert frame.status_code == 200 and frame.headers["content-type"] == "image/jpeg"
    assert frame.headers["x-frame-id"] == "8" and frame.headers["x-perception-source"] == "synthetic_rgb_cv"
    assert frame.headers["cache-control"] == "no-store" and float(frame.headers["x-sim-time"]) == pytest.approx(0.08)
    assert frame.content.endswith((8).to_bytes(4, "big"))  # the exact image of that frame id

    owner = repo.get_run_any(harness.services.db, run_id).owner_id  # type: ignore[union-attr]
    events = harness.events(owner)
    states = [e["payload"]["state"] for e in events if e["type"] == "run_state"]
    assert states[0] == "initializing" and "running" in states and states[-1] == "arrived"
    assert any(e["type"] == "frame_ready" for e in events)
    decisions = [e["payload"] for e in events if e["type"] == "decision"]
    assert decisions and decisions[-1]["state"] == "arrived" and decisions[-1]["command"]["brake"] == 1.0
    # Run slot is free again; stop after arrival returns the terminal state.
    stop = client.post(f"/api/v1/runs/{run_id}/control", json={"action": "stop"}, headers=harness.csrf(_info))
    assert stop.json()["state"] == "arrived"


def test_pause_resume_stop_reach_the_worker(harness: Harness) -> None:
    FakeSession.ticks_to_arrive = 100_000
    client, info, run_id = _create(harness)
    thread = run_in_thread(lambda: sim_job.run_simulation_job(run_id))
    harness.wait_for(lambda: _state(harness, run_id) == "running")

    def control(action: str) -> dict[str, Any]:
        response = client.post(f"/api/v1/runs/{run_id}/control", json={"action": action}, headers=harness.csrf(info))
        assert response.status_code == 200, response.text
        return response.json()

    control("pause")
    harness.wait_for(lambda: _state(harness, run_id) == "paused")
    paused_frame = client.get(f"/api/v1/runs/{run_id}").json()["live"]["frame_id"]
    time.sleep(0.1)
    assert client.get(f"/api/v1/runs/{run_id}").json()["live"]["frame_id"] == paused_frame  # no stepping
    assert control("pause")["state"] == "paused"  # idempotent
    control("resume")
    harness.wait_for(lambda: _state(harness, run_id) == "running")
    harness.wait_for(lambda: client.get(f"/api/v1/runs/{run_id}").json()["live"]["frame_id"] > paused_frame + 3)
    stopping = control("stop")
    assert stopping["state"] in ("stopping", "stopped")
    thread.join(10)
    assert not thread.is_alive()
    final = client.get(f"/api/v1/runs/{run_id}").json()
    assert final["state"] == "stopped" and final["outcome"] == "stopped_operator"
    session = FakeSession.instances[-1]
    assert session.pause_calls >= 1 and session.resume_calls == 1 and session.stop_requested
    resume_after = client.post(f"/api/v1/runs/{run_id}/control", json={"action": "resume"}, headers=harness.csrf(info))
    assert resume_after.status_code == 409


def test_obstacle_injection_reaches_session_and_is_bounded(harness: Harness) -> None:
    FakeSession.ticks_to_arrive = 100_000
    client, info, run_id = _create(harness)
    thread = run_in_thread(lambda: sim_job.run_simulation_job(run_id))
    harness.wait_for(lambda: _state(harness, run_id) == "running")
    codes = []
    for i in range(6):
        response = client.post(
            f"/api/v1/runs/{run_id}/obstacles",
            json={"distance_ahead_m": 10 + i, "lateral_offset_m": 0.5},
            headers=harness.csrf(info),
        )
        codes.append(response.status_code)
        if response.status_code == 202:
            body = response.json()
            assert body["injected_total"] == i + 1 and body["source"] == "synthetic_scenario_injection"
        else:
            assert response.json()["code"] == "obstacle_limit_reached"
    assert codes == [202] * 5 + [409]
    harness.wait_for(lambda: len(FakeSession.instances[-1].injected) == 5)
    assert FakeSession.instances[-1].injected[0] == (10.0, 0.5)
    client.post(f"/api/v1/runs/{run_id}/control", json={"action": "stop"}, headers=harness.csrf(info))
    thread.join(10)


def test_wall_time_watchdog_through_session(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSession.ticks_to_arrive = 100_000
    FakeSession.default_wall_limit_s = 0.2
    _client, _info, run_id = _create(harness)
    assert sim_job.run_simulation_job(run_id) == "failed"
    row = repo.get_run_any(harness.services.db, run_id)
    assert row is not None and row.outcome == "failed_wall_time" and row.state == "failed"


def test_hard_watchdog_when_session_ignores_wall_limit(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSession.ticks_to_arrive = 100_000
    monkeypatch.setattr(sim_job, "HARD_WATCHDOG_GRACE_S", 0.1)
    monkeypatch.setattr(harness.settings, "max_simulation_wall_seconds", 0.2)
    original = FakeSession.__init__

    def init(self: FakeSession, cfg: Any, inference_client: Any = None) -> None:
        original(self, cfg, inference_client)
        self.ignore_wall_limit = True

    monkeypatch.setattr(FakeSession, "__init__", init)
    _client, _info, run_id = _create(harness)
    assert sim_job.run_simulation_job(run_id) == "failed"
    row = repo.get_run_any(harness.services.db, run_id)
    assert row is not None and row.outcome == "failed_wall_time" and row.error_code == "wall_time_limit"


def test_session_crash_marks_failed_without_internals(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_tick(self: FakeSession) -> Any:
        raise RuntimeError("secret /data/path detail")

    monkeypatch.setattr(FakeSession, "tick", broken_tick)
    client, _info, run_id = _create(harness)
    assert sim_job.run_simulation_job(run_id) == "failed"
    run = client.get(f"/api/v1/runs/{run_id}")
    assert run.json()["error_code"] == "internal_error" and "secret" not in run.text and "/data" not in run.text


def test_job_skips_runs_that_are_not_initializing(harness: Harness) -> None:
    client, info, run_id = _create(harness)
    client.post(f"/api/v1/runs/{run_id}/control", json={"action": "stop"}, headers=harness.csrf(info))
    assert sim_job.run_simulation_job(run_id) == "stopped"
    assert FakeSession.instances == []


def test_stopping_before_start_finishes_as_stopped(harness: Harness) -> None:
    client, _info, run_id = _create(harness)
    repo.transition_run(harness.services.db, run_id, ("initializing",), "stopping")
    assert sim_job.run_simulation_job(run_id) == "stopped"
    row = repo.get_run_any(harness.services.db, run_id)
    assert row is not None and row.outcome == "stopped_operator" and FakeSession.instances == []


def test_live_json_and_frame_written_atomically(harness: Harness) -> None:
    _client, _info, run_id = _create(harness)
    sim_job.run_simulation_job(run_id)
    frame, live = harness.redis.mget([f"dd:run:{run_id}:frame", f"dd:run:{run_id}:live"])
    assert frame is not None and live is not None
    assert int.from_bytes(frame[-4:], "big") == json.loads(live)["frame_id"]
    assert 0 < harness.redis.ttl(f"dd:run:{run_id}:frame") <= 60
