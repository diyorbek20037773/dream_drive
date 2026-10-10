# ruff: noqa: F811, S105, S106 - pytest fixtures are imported from the shared harness; test-only passwords
"""Capabilities, health/readiness, Prometheus metrics, JSON logs and evaluations endpoints."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest
from test_api_harness import TEST_COMMIT, Harness, harness  # noqa: F401

from app.api import evaluations as evaluations_api
from app.jobs import eval_job
from app.logging_setup import JsonFormatter, log_context, redact
from app.storage import repo


def test_capabilities_with_ready_model(harness: Harness) -> None:
    body = harness.new_client().get("/api/v1/capabilities").json()
    video = body["video_analysis"]
    assert video["available"] is True and video["reason"] is None
    assert video["supported_classes"] == ["vehicle"]
    assert video["unsupported_examples"] == ["person", "traffic_light", "traffic_sign"]
    assert video["device"] == "cpu" and video["model"]["model_id"] == "yolopv2"
    assert body["build"] == {"commit": TEST_COMMIT, "version": "dev", "dirty": False}
    assert body["queue"] == {
        "video_queued": 0,
        "video_running": 0,
        "capacity_total": harness.settings.max_queued_total,
        "per_session": harness.settings.max_queued_per_session,
        "sim_active": 0,
        "sim_capacity": 1,
        "reason": None,
    }
    assert [s["id"] for s in body["simulation"]["simulators"]] == ["lite", "carla"]
    assert body["monitoring"] == {"metrics": True}


def test_capabilities_without_model_reports_reason(harness: Harness) -> None:
    harness.inference.ready = False
    harness.inference.reason = "model_missing"
    harness.services.model_status.invalidate()
    video = harness.new_client().get("/api/v1/capabilities").json()["video_analysis"]
    assert video["available"] is False and video["reason"] == "model_missing" and video["supported_classes"] == []
    harness.inference.unreachable = True
    harness.services.model_status.invalidate()
    video = harness.new_client().get("/api/v1/capabilities").json()["video_analysis"]
    assert video["available"] is False and video["reason"] == "inference_unreachable" and video["model"] is None


def test_build_commit_null_when_unknown(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness.settings, "git_commit", "unknown")
    assert harness.new_client().get("/api/v1/capabilities").json()["build"]["commit"] is None


def test_capabilities_counts_queue(harness: Harness) -> None:
    client, info = harness.login()
    harness.upload(client, info)
    harness.upload(client, info)
    queue = client.get("/api/v1/capabilities").json()["queue"]
    assert queue["video_queued"] == 2 and queue["video_running"] == 0


def test_health_live_and_ready(harness: Harness) -> None:
    client = harness.new_client()
    assert client.get("/health/live").json()["status"] == "ok"
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["checks"] == {"database": {"ok": True, "reason": None}, "queue": {"ok": True, "reason": None}}
    # Model readiness never fails API readiness.
    harness.inference.ready = False
    harness.services.model_status.invalidate()
    assert client.get("/health/ready").status_code == 200
    harness.server.connected = False
    try:
        down = client.get("/health/ready")
    finally:
        harness.server.connected = True
    assert down.status_code == 503
    assert down.json()["status"] == "unavailable" and down.json()["checks"]["queue"]["reason"] == "queue_unavailable"
    assert client.get("/health/live").status_code == 200


def test_metrics_use_route_templates_not_ids(harness: Harness) -> None:
    client, _ = harness.login()
    client.get("/api/v1/jobs/" + "c" * 32)
    text = client.get("/metrics").text
    assert 'route="/api/v1/jobs/{job_id}"' in text
    assert "c" * 32 not in text
    assert "dreamdrive_http_request_duration_seconds_bucket" in text
    assert 'status_class="4xx"' in text
    assert "dreamdrive_model_ready" in text


def test_json_log_format_and_redaction() -> None:
    formatter = JsonFormatter("api")
    record = logging.LogRecord("dreamdrive.test", logging.INFO, __file__, 1, "cookie dd_session=abc.def; x", None, None)
    record.code = "demo"
    with log_context(job_id="j1", request_id="r1"):
        line = json.loads(formatter.format(record))
    assert line["component"] == "api" and line["level"] == "INFO" and line["code"] == "demo"
    assert line["job_id"] == "j1" and line["request_id"] == "r1" and line["ts"].endswith("Z")
    assert "abc.def" not in line["msg"]
    assert redact("X-CSRF-Token: secrettoken") == "X-CSRF-Token: [redacted]"
    assert "hunter2" not in redact('{"admin_password": "hunter2"}')
    assert "tok123" not in redact("Authorization: Bearer tok123")


EVAL_ARTIFACT = {
    "id": "eval-artifact-1",
    "kind": "synthetic_scenario_suite",
    "created_utc": "2026-10-10T10:00:00+00:00",
    "finished_utc": "2026-10-10T10:05:00+00:00",
    "source": "literoad_synthetic_rgb_cv",
    "sample_count": 3,
    "metrics": {"arrival_rate": {"value": 1.0, "unit": "ratio", "sample_count": 3, "status": "measured", "note": ""}},
    "limitations": ["synthetic"],
    "code_commit": TEST_COMMIT,
    "model_sha256": None,
    "seeds": [1, 2, 3],
}


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    store: dict[str, dict[str, Any]] = {}
    fake = SimpleNamespace(
        list_evaluations=lambda root: list(store.values()),
        load_evaluation=lambda root, eval_id: store.get(eval_id),
    )
    monkeypatch.setattr(evaluations_api, "_registry", lambda: fake)
    monkeypatch.setattr(evaluations_api.importlib.util, "find_spec", lambda name: object())
    return store


def test_evaluations_list_get_and_create(harness: Harness, registry: dict[str, Any], monkeypatch: Any) -> None:
    client, info = harness.login()
    assert client.get("/api/v1/evaluations").json() == {"items": []}
    registry[EVAL_ARTIFACT["id"]] = dict(EVAL_ARTIFACT)
    listed = client.get("/api/v1/evaluations").json()["items"]
    assert [e["id"] for e in listed] == ["eval-artifact-1"] and listed[0]["state"] == "succeeded"
    assert client.get("/api/v1/evaluations/eval-artifact-1").json()["sample_count"] == 3
    assert client.get("/api/v1/evaluations/does-not-exist").status_code == 404

    created = client.post("/api/v1/evaluations", json={"seeds": [4, 5]}, headers=harness.csrf(info))
    assert created.status_code == 202, created.text
    eval_id = created.json()["id"]
    assert created.json()["state"] == "queued" and created.json()["sample_count"] == 0
    busy = client.post("/api/v1/evaluations", json={"seeds": [6]}, headers=harness.csrf(info))
    assert busy.status_code == 429 and busy.json()["code"] == "evaluation_busy"
    other, _ = harness.login()
    assert other.get(f"/api/v1/evaluations/{eval_id}").status_code == 404  # queued request is owner-scoped

    def fake_suite(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["seeds"] == [4, 5] and kwargs["code_commit"] == TEST_COMMIT
        result = {**EVAL_ARTIFACT, "id": "eval-artifact-2", "seeds": [4, 5], "sample_count": 2}
        registry["eval-artifact-2"] = result
        return result

    monkeypatch.setattr(eval_job, "run_suite", fake_suite)
    assert eval_job.run_scenario_suite_job(eval_id) == "succeeded"
    row = repo.get_evaluation_any(harness.services.db, eval_id)
    assert row is not None and row.artifact_id == "eval-artifact-2"
    ids = [e["id"] for e in client.get("/api/v1/evaluations").json()["items"]]
    assert sorted(ids) == ["eval-artifact-1", "eval-artifact-2"]  # the request row is not duplicated
    resolved = client.get(f"/api/v1/evaluations/{eval_id}").json()
    assert resolved["id"] == "eval-artifact-2" and resolved["sample_count"] == 2


def test_evaluation_failure_is_visible(harness: Harness, registry: dict[str, Any], monkeypatch: Any) -> None:
    client, info = harness.login()
    eval_id = client.post("/api/v1/evaluations", json={"seeds": [1]}, headers=harness.csrf(info)).json()["id"]

    def boom(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("internal")

    monkeypatch.setattr(eval_job, "run_suite", boom)
    assert eval_job.run_scenario_suite_job(eval_id) == "failed"
    item = client.get(f"/api/v1/evaluations/{eval_id}").json()
    assert item["state"] == "failed" and item["error_code"] == "internal_error" and item["metrics"] == {}


def test_evaluation_blocked_while_simulation_active(harness: Harness, registry: dict[str, Any]) -> None:
    client, info = harness.login()
    run = client.post("/api/v1/runs", json={"start_node": "A", "destination_node": "C"}, headers=harness.csrf(info))
    assert run.status_code == 202
    response = client.post("/api/v1/evaluations", json={"seeds": [1]}, headers=harness.csrf(info))
    assert response.status_code == 429 and response.json()["code"] == "simulation_busy"
