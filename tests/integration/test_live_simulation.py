"""Live stack: LiteRoad closed-loop runs through the public API (normal arrival, obstacle stop, unreachable)."""

from __future__ import annotations

import os
import time
from typing import Any

import pytest
from live_client import Api

RUN_TIMEOUT_S = float(os.environ.get("DD_RUN_TIMEOUT_S", "200"))
TERMINAL = ("stopped", "arrived", "failed")


def _lite(api: Api) -> dict[str, Any]:
    items = api.get("/api/v1/simulators").json()["items"]
    lite = next(s for s in items if s["id"] == "lite")
    assert lite["available"] is True
    return lite


def _start(api: Api, scenario: str, seed: int) -> dict[str, Any]:
    deadline = time.monotonic() + 180
    while True:
        response = api.post(
            "/api/v1/runs",
            json={"simulator": "lite", "start_node": "A", "destination_node": "C", "scenario": scenario, "seed": seed},
        )
        if response.status_code == 429 and time.monotonic() < deadline:
            # Another run is still active (simulation_busy) or the per-minute budget is used up (rate_limited).
            time.sleep(max(3.0, float(response.headers.get("retry-after", "3"))))
            continue
        assert response.status_code == 202, response.text
        return response.json()


def _wait_terminal(api: Api, run_id: str) -> tuple[dict[str, Any], int]:
    frames = 0
    deadline = time.monotonic() + RUN_TIMEOUT_S
    run: dict[str, Any] = {}
    while time.monotonic() < deadline:
        run = api.get(f"/api/v1/runs/{run_id}").json()
        frame = api.get(f"/api/v1/runs/{run_id}/frame")
        if frame.status_code == 200:
            assert frame.headers["content-type"] == "image/jpeg" and frame.content[:2] == b"\xff\xd8"
            assert frame.headers["x-perception-source"] == "synthetic_rgb_cv"
            assert int(frame.headers["x-frame-id"]) >= 0 and frame.headers["cache-control"] == "no-store"
            frames += 1
        if run["state"] in TERMINAL:
            return run, frames
        time.sleep(0.5)
    pytest.fail(f"run {run_id} did not finish within {RUN_TIMEOUT_S} s: {run.get('state')}")


def test_simulator_catalogue_and_unreachable(api: Api) -> None:
    lite = _lite(api)
    nodes = {n["id"]: n for n in lite["road_graph"]["nodes"]}
    unreachable = [n for n, v in nodes.items() if not v["reachable_from_default"]]
    assert unreachable, "the road graph must contain an unreachable node"
    response = api.post("/api/v1/runs", json={"start_node": lite["default_start"], "destination_node": unreachable[0]})
    assert response.status_code == 422 and response.json()["code"] == "destination_unreachable"


def test_normal_run_arrives(api: Api) -> None:
    _lite(api)
    run = _start(api, "normal", 1)
    assert run["state"] == "initializing" and run["route"]["length_m"] > 0
    final, frames = _wait_terminal(api, run["id"])
    assert final["state"] == "arrived" and final["outcome"] == "arrived", final
    assert frames > 0
    assert final["live"]["perception"]["source"] == "synthetic_rgb_cv"
    report = api.get(f"/api/v1/runs/{run['id']}/report")
    assert report.status_code == 200, report.text
    body = report.json()
    assert body["kind"] == "simulation" and body["perception_source"] == "synthetic_rgb_cv"
    assert body["runtime_used_dynamic_ground_truth"] is False
    stop = api.post(f"/api/v1/runs/{run['id']}/control", json={"action": "stop"})
    assert stop.status_code == 200 and stop.json()["state"] == "arrived"


def test_obstacle_scenario_stops_before_obstacle(api: Api) -> None:
    run = _start(api, "obstacle_stop", 2)
    final, _frames = _wait_terminal(api, run["id"])
    # The vehicle must stop (or fail visibly) instead of falsely arriving through the obstacle.
    assert final["state"] != "arrived", final
    assert final["outcome"] in ("stopped_before_obstacle",) or final["state"] == "failed", final
    report = api.get(f"/api/v1/runs/{run['id']}/report")
    assert report.status_code == 200


def test_operator_stop_reaches_worker(api: Api) -> None:
    run = _start(api, "normal", 3)
    deadline = time.monotonic() + 60
    while api.get(f"/api/v1/runs/{run['id']}").json()["state"] == "initializing" and time.monotonic() < deadline:
        time.sleep(0.3)
    response = api.post(f"/api/v1/runs/{run['id']}/control", json={"action": "stop"})
    assert response.status_code == 200 and response.json()["state"] in ("stopping", "stopped", "arrived")
    final, _ = _wait_terminal(api, run["id"])
    assert final["state"] in ("stopped", "arrived")
    if final["state"] == "stopped":
        assert final["outcome"] == "stopped_operator"
