# ruff: noqa: F811, S105, S106 - pytest fixtures are imported from the shared harness; test-only passwords
"""Two sessions cannot read or change each other's jobs, artifacts, runs or SSE events (404, never 403)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from test_api_harness import FakeSession, Harness, harness  # noqa: F401
from test_jobs_video import pipeline  # noqa: F401

from app.jobs import sim_job, video_job
from app.storage import repo

ARTIFACTS = ("input", "annotated.mp4", "timeline.json", "report.json", "report.html", "poster.jpg")


@pytest.fixture
def two_owners(harness: Harness, pipeline: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    def create(row: repo.RunRow, ctx: Any) -> tuple[Any, Any]:
        cfg = type("Cfg", (), {"run_id": row.id, "dt": ctx.settings.sim_fixed_dt_seconds})()
        return FakeSession(cfg), None

    monkeypatch.setattr(sim_job, "create_session", create)
    a, ainfo = harness.login()
    b, binfo = harness.login()
    job_id = harness.upload(a, ainfo).json()["id"]
    assert video_job.run_video_job(job_id) == "succeeded"
    run = a.post("/api/v1/runs", json={"start_node": "A", "destination_node": "C"}, headers=harness.csrf(ainfo))
    run_id = run.json()["id"]
    assert sim_job.run_simulation_job(run_id) == "arrived"
    return {"a": a, "ainfo": ainfo, "b": b, "binfo": binfo, "job": job_id, "run": run_id}


def _assert_not_found(response: Any) -> None:
    assert response.status_code == 404, (response.request.method, response.request.url, response.status_code)
    assert response.json()["code"] == "not_found"


def test_foreign_job_and_artifacts_are_not_found(harness: Harness, two_owners: dict[str, Any]) -> None:
    b, binfo, job_id = two_owners["b"], two_owners["binfo"], two_owners["job"]
    a = two_owners["a"]
    assert a.get(f"/api/v1/jobs/{job_id}").status_code == 200  # sanity: the owner sees it
    _assert_not_found(b.get(f"/api/v1/jobs/{job_id}"))
    _assert_not_found(b.get(f"/api/v1/jobs/{job_id}/results"))
    for name in ARTIFACTS:
        _assert_not_found(b.get(f"/api/v1/jobs/{job_id}/artifacts/{name}"))
        _assert_not_found(b.get(f"/api/v1/jobs/{job_id}/artifacts/{name}", headers={"Range": "bytes=0-10"}))
    _assert_not_found(b.post(f"/api/v1/jobs/{job_id}/cancel", headers=harness.csrf(binfo)))
    _assert_not_found(b.delete(f"/api/v1/jobs/{job_id}", headers=harness.csrf(binfo)))
    assert b.get("/api/v1/jobs").json()["items"] == []
    # Nothing B did changed A's job.
    job = a.get(f"/api/v1/jobs/{job_id}").json()
    assert job["state"] == "succeeded"
    assert a.get(f"/api/v1/jobs/{job_id}/artifacts/annotated.mp4").status_code == 200


def test_foreign_run_is_not_found(harness: Harness, two_owners: dict[str, Any]) -> None:
    b, binfo, run_id = two_owners["b"], two_owners["binfo"], two_owners["run"]
    _assert_not_found(b.get(f"/api/v1/runs/{run_id}"))
    _assert_not_found(b.get(f"/api/v1/runs/{run_id}/frame"))
    _assert_not_found(b.get(f"/api/v1/runs/{run_id}/report"))
    for action in ("pause", "resume", "stop"):
        _assert_not_found(
            b.post(f"/api/v1/runs/{run_id}/control", json={"action": action}, headers=harness.csrf(binfo))
        )
    _assert_not_found(
        b.post(
            f"/api/v1/runs/{run_id}/obstacles",
            json={"distance_ahead_m": 10, "lateral_offset_m": 0},
            headers=harness.csrf(binfo),
        )
    )
    assert b.get("/api/v1/runs").json()["items"] == []
    assert two_owners["a"].get(f"/api/v1/runs/{run_id}/report").status_code == 200


def test_sse_never_carries_foreign_ids(harness: Harness, two_owners: dict[str, Any]) -> None:
    a, b = two_owners["a"], two_owners["b"]
    a_text = a.get("/api/v1/events", params={"last_event_id": "0-0"}).text
    assert two_owners["job"] in a_text and two_owners["run"] in a_text
    b_text = b.get("/api/v1/events", params={"last_event_id": "0-0"}).text
    assert two_owners["job"] not in b_text and two_owners["run"] not in b_text
    for line in b_text.splitlines():
        if line.startswith("data:"):
            assert json.loads(line[5:])["resource_id"] not in (two_owners["job"], two_owners["run"])


def test_unknown_and_foreign_ids_are_indistinguishable(harness: Harness, two_owners: dict[str, Any]) -> None:
    b = two_owners["b"]
    foreign = b.get(f"/api/v1/jobs/{two_owners['job']}")
    unknown = b.get("/api/v1/jobs/" + "e" * 32)
    assert foreign.status_code == unknown.status_code == 404
    assert foreign.json()["message"] == unknown.json()["message"]
    assert foreign.json()["details"] == unknown.json()["details"] == {}
