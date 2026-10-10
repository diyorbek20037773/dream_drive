# ruff: noqa: F811, S105, S106 - pytest fixtures are imported from the shared harness; test-only passwords
"""Simulators catalogue and run validation/admission/control endpoints (runner replaced by a fake session)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from rq import Queue
from test_api_harness import Harness, build_harness, harness  # noqa: F401

from app.storage import repo

RUN = {"start_node": "A", "destination_node": "C", "scenario": "normal", "seed": 7}


def _create(h: Harness, client: Any, info: dict[str, Any], **overrides: Any) -> Any:
    return client.post("/api/v1/runs", json={**RUN, **overrides}, headers=h.csrf(info))


def test_simulator_catalogue(harness: Harness) -> None:
    client, _ = harness.login()
    items = {s["id"]: s for s in client.get("/api/v1/simulators").json()["items"]}
    lite = items["lite"]
    assert lite["available"] is True and lite["default_start"] == "A"
    modes = {m["id"]: m for m in lite["perception_modes"]}
    assert modes["synthetic_rgb_cv"]["available"] is True
    assert modes["learned_model"] == {"id": "learned_model", "available": False, "reason": "learned_mode_not_validated"}
    nodes = {n["id"]: n for n in lite["road_graph"]["nodes"]}
    assert nodes["C"]["reachable_from_default"] is True and nodes["Q"]["reachable_from_default"] is False
    assert lite["road_graph"]["edges"][0]["from"] == "A" and lite["road_graph"]["units"] == "m"
    scenarios = {s["id"]: s for s in lite["scenarios"]}
    assert scenarios["pause_resume"]["available"] is False and scenarios["pause_resume"]["reason"]
    carla = items["carla"]
    assert carla["available"] is False and carla["reason"] == "carla_host_not_configured"
    assert carla["road_graph"] is None
    assert harness.new_client().get("/api/v1/simulators").status_code == 401


@pytest.fixture
def learned_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    yield from build_harness(tmp_path, monkeypatch, learned_sim_mode_enabled=True, carla_host="carla.internal")


def test_learned_mode_requires_flag_and_ready_model(learned_harness: Harness) -> None:
    h = learned_harness
    client, info = h.login()
    lite = client.get("/api/v1/simulators").json()["items"][0]
    assert {m["id"]: m["available"] for m in lite["perception_modes"]}["learned_model"] is True
    carla = client.get("/api/v1/simulators").json()["items"][1]
    assert carla["reason"] == "carla_adapter_not_available"
    h.inference.ready = False
    h.services.model_status.invalidate()
    response = _create(h, client, info, perception_source="learned_model")
    assert response.status_code == 422
    assert response.json()["code"] == "unsupported_mode" and response.json()["details"]["reason"] == "model_not_ready"


def test_run_validation_errors(harness: Harness) -> None:
    client, info = harness.login()
    unknown = _create(harness, client, info, destination_node="ZZ")
    assert unknown.status_code == 422 and unknown.json()["code"] == "unknown_node"
    unreachable = _create(harness, client, info, destination_node="Q")
    assert unreachable.status_code == 422 and unreachable.json()["code"] == "destination_unreachable"
    learned = _create(harness, client, info, perception_source="learned_model")
    assert learned.status_code == 422 and learned.json()["code"] == "unsupported_mode"
    assert learned.json()["details"]["reason"] == "learned_mode_not_validated"
    scenario = _create(harness, client, info, scenario="pause_resume")
    assert scenario.status_code == 422 and scenario.json()["code"] == "unsupported_scenario"
    missing = _create(harness, client, info, scenario="camera_blackout")  # not offered by this (fake) catalogue
    assert missing.status_code == 422 and missing.json()["code"] == "unsupported_scenario"
    carla = _create(harness, client, info, simulator="carla")
    assert carla.status_code == 503 and carla.json()["code"] == "simulator_unavailable"
    truth = _create(harness, client, info, perception_source="simulator_truth")
    assert truth.status_code == 422 and truth.json()["code"] == "validation_error"
    assert repo.count_runs(harness.services.db, repo.RUN_ACTIVE_STATES) == 0
    assert Queue("sim", connection=harness.redis).count == 0


def test_create_run_enqueues_and_enforces_single_active(harness: Harness) -> None:
    client, info = harness.login()
    response = _create(harness, client, info)
    assert response.status_code == 202, response.text
    run = response.json()
    assert run["state"] == "initializing" and run["live"] is None and run["simulator"] == "lite"
    assert run["route"]["node_ids"] == ["A", "C"] and run["route"]["length_m"] == 130.0
    assert run["perception_source"] == "synthetic_rgb_cv" and run["seed"] == 7
    queue = Queue("sim", connection=harness.redis)
    assert queue.job_ids == [f"sim-{run['id']}"]
    assert queue.fetch_job(f"sim-{run['id']}").func_name == "app.jobs.sim_job.run_simulation_job"  # type: ignore[union-attr]
    other, other_info = harness.login()
    busy = _create(harness, other, other_info)
    assert busy.status_code == 429 and busy.json()["code"] == "simulation_busy"
    listed = client.get("/api/v1/runs").json()
    assert [r["id"] for r in listed["items"]] == [run["id"]]
    assert other.get("/api/v1/runs").json()["items"] == []


def test_stop_before_worker_start_and_terminal_controls(harness: Harness) -> None:
    client, info = harness.login()
    run_id = _create(harness, client, info).json()["id"]
    stopped = client.post(f"/api/v1/runs/{run_id}/control", json={"action": "stop"}, headers=harness.csrf(info))
    assert stopped.status_code == 200
    assert stopped.json()["state"] == "stopped" and stopped.json()["outcome"] == "stopped_operator"
    assert Queue("sim", connection=harness.redis).count == 0
    again = client.post(f"/api/v1/runs/{run_id}/control", json={"action": "stop"}, headers=harness.csrf(info))
    assert again.status_code == 200 and again.json()["state"] == "stopped"
    pause = client.post(f"/api/v1/runs/{run_id}/control", json={"action": "pause"}, headers=harness.csrf(info))
    assert pause.status_code == 409 and pause.json()["code"] == "invalid_transition"
    bad = client.post(f"/api/v1/runs/{run_id}/control", json={"action": "reverse"}, headers=harness.csrf(info))
    assert bad.status_code == 422
    # The slot is free again.
    assert _create(harness, client, info).status_code == 202


def test_frame_report_and_obstacles_before_running(harness: Harness) -> None:
    client, info = harness.login()
    run_id = _create(harness, client, info).json()["id"]
    frame = client.get(f"/api/v1/runs/{run_id}/frame")
    assert frame.status_code == 204 and frame.headers["cache-control"] == "no-store"
    report = client.get(f"/api/v1/runs/{run_id}/report")
    assert report.status_code == 409 and report.json()["code"] == "report_unavailable"
    obstacle = client.post(
        f"/api/v1/runs/{run_id}/obstacles",
        json={"distance_ahead_m": 20, "lateral_offset_m": 0},
        headers=harness.csrf(info),
    )
    assert obstacle.status_code == 409 and obstacle.json()["code"] == "run_not_active"
    out_of_bounds = client.post(
        f"/api/v1/runs/{run_id}/obstacles",
        json={"distance_ahead_m": 100, "lateral_offset_m": 0},
        headers=harness.csrf(info),
    )
    assert out_of_bounds.status_code == 422


def test_run_creation_is_rate_limited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import functools

    gen = build_harness(tmp_path, monkeypatch, max_requests_per_minute=2)
    h = next(gen)
    try:
        h.services.limiter.hit = functools.partial(h.services.limiter.hit, now=1_000_020.0)  # type: ignore[method-assign]
        client, info = h.login()
        assert _create(h, client, info, destination_node="Q").status_code == 422  # counts as an expensive request
        assert _create(h, client, info, destination_node="Q").status_code == 422
        limited = _create(h, client, info)
        assert limited.status_code == 429 and limited.json()["code"] == "rate_limited"
    finally:
        gen.close()
