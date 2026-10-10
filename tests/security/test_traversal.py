# ruff: noqa: F811, S105, S106 - pytest fixtures are imported from the shared harness; test-only passwords
"""Path traversal attempts against artifact and resource routes never reach files outside the allowlist."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from test_api_harness import Harness, harness  # noqa: F401
from test_jobs_video import pipeline  # noqa: F401

from app.jobs import video_job

NASTY_NAMES = [
    "%2e%2e%2f%2e%2e%2fdreamdrive.db",
    "..%2f..%2f..%2fdreamdrive.db",
    "%2e%2e%5c%2e%2e%5cdreamdrive.db",
    "input.mp4",
    "input%00.mp4",
    "annotated.mp4%00.txt",
    "ANNOTATED.MP4",
    "report.json%2f..%2f..%2fdreamdrive.db",
    "~",
    "%252e%252e%252fdreamdrive.db",
]


@pytest.fixture
def job(harness: Harness, pipeline: dict[str, Any]) -> tuple[Any, str]:
    client, info = harness.login()
    job_id = harness.upload(client, info).json()["id"]
    assert video_job.run_video_job(job_id) == "succeeded"
    (harness.settings.data_root / "secret.txt").write_text("TOP-SECRET", encoding="utf-8")
    return client, job_id


def test_encoded_traversal_in_artifact_name(harness: Harness, job: tuple[Any, str]) -> None:
    client, job_id = job
    for name in NASTY_NAMES:
        response = client.get(f"/api/v1/jobs/{job_id}/artifacts/{name}")
        assert response.status_code in (404, 422), (name, response.status_code)
        assert "SQLite format" not in response.text and "TOP-SECRET" not in response.text


def test_traversal_in_resource_ids(harness: Harness, job: tuple[Any, str]) -> None:
    client, _job_id = job
    for job_id in ("..%2f..%2fsecret.txt", "%2e%2e", "a" * 65, "x y", "..", "%00"):
        for url in (f"/api/v1/jobs/{job_id}", f"/api/v1/jobs/{job_id}/artifacts/input", f"/api/v1/runs/{job_id}/frame"):
            response = client.get(url)
            assert response.status_code in (404, 405, 422), (url, response.status_code)
            assert "TOP-SECRET" not in response.text and "SQLite format" not in response.text


def _raw_get(app: Any, raw_path: bytes, cookies: str) -> tuple[int, bytes]:
    """Send an unnormalized path straight to the ASGI app (HTTP clients would collapse ``..`` segments)."""
    status = 0
    body = b""

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        nonlocal status, body
        if message["type"] == "http.response.start":
            status = message["status"]
        elif message["type"] == "http.response.body":
            body += message.get("body", b"")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": raw_path.decode("latin-1"),
        "raw_path": raw_path,
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"testserver"), (b"cookie", cookies.encode())],
        "client": ("127.0.0.1", 5000),
        "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))
    return status, body


def test_unnormalized_dot_segments_never_serve_files(harness: Harness, job: tuple[Any, str]) -> None:
    client, job_id = job
    cookie = "; ".join(f"{k}={v}" for k, v in client.cookies.items())
    for raw in (
        f"/api/v1/jobs/{job_id}/artifacts/../../../dreamdrive.db",
        f"/api/v1/jobs/{job_id}/artifacts/../input",
        "/api/v1/jobs/../../secret.txt",
        f"/api/v1/jobs/{job_id}/artifacts/..\\..\\dreamdrive.db",
    ):
        status, body = _raw_get(client.app, raw.encode(), cookie)
        assert status in (404, 405, 422), (raw, status)
        assert b"SQLite format" not in body and b"TOP-SECRET" not in body
