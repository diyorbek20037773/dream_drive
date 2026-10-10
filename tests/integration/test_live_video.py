"""Live stack: session -> capabilities -> upload a real video -> real inference job -> results -> ranged download
of annotated.mp4 -> SSE replay of the job's events -> delete."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from live_client import REQUIRE_LIVE, Api

JOB_TIMEOUT_S = float(os.environ.get("DD_JOB_TIMEOUT_S", "1800"))


def _sse_events(api: Api, last_event_id: str = "0-0", until: str | None = None, timeout_s: float = 20.0) -> list[dict]:
    events: list[dict] = []
    deadline = time.monotonic() + timeout_s
    with api.http.stream(
        "GET", "/api/v1/events", params={"last_event_id": last_event_id}, timeout=httpx.Timeout(timeout_s)
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:]))
                if until and until in line:
                    break
            if time.monotonic() > deadline:
                break
    return events


def test_capabilities_report_real_model(api: Api) -> None:
    caps = api.get("/api/v1/capabilities").json()
    video = caps["video_analysis"]
    if not video["available"]:
        message = f"video analysis unavailable: {video['reason']}"
        if REQUIRE_LIVE:
            pytest.fail(message)
        pytest.skip(message)
    assert video["model"]["ready"] is True and len(video["model"]["sha256"]) == 64
    assert video["supported_classes"], "a ready model must list its verified classes"
    assert "person" in video["unsupported_examples"]
    lite = caps["simulation"]["simulators"][0]
    assert lite["id"] == "lite" and lite["available"] is True


def test_video_upload_to_results_and_download(api: Api, fixture_video: Path) -> None:
    caps = api.get("/api/v1/capabilities").json()
    if not caps["video_analysis"]["available"]:
        message = f"video analysis unavailable: {caps['video_analysis']['reason']}"
        if REQUIRE_LIVE:
            pytest.fail(message)
        pytest.skip(message)
    data = fixture_video.read_bytes()
    response = api.post("/api/v1/videos", files={"file": ("fixture.mp4", data, "video/mp4")})
    assert response.status_code == 202, response.text
    job = response.json()
    job_id = job["id"]
    assert job["state"] == "queued" and job["input"]["sha256"] == hashlib.sha256(data).hexdigest()

    deadline = time.monotonic() + JOB_TIMEOUT_S
    seen_states = set()
    while time.monotonic() < deadline:
        job = api.get(f"/api/v1/jobs/{job_id}").json()
        seen_states.add(job["state"])
        if job["state"] in ("succeeded", "failed", "cancelled", "interrupted"):
            break
        time.sleep(2.0)
    assert job["state"] == "succeeded", job
    assert job["model"] and len(job["model"]["sha256"]) == 64 and job["video"]["width"] > 0

    results = api.get(f"/api/v1/jobs/{job_id}/results").json()
    summary = results["summary"]
    assert results["perception_source"] == "learned_model"
    assert summary["analyzed_frames"] > 0 and summary["timing_ms"]["inference_p50"] is not None
    assert summary["audio"] == "removed"
    names = {a["name"]: a for a in results["artifacts"]}
    assert {"annotated.mp4", "timeline.json", "report.json", "report.html", "poster.jpg", "input"} <= set(names)

    url = names["annotated.mp4"]["url"]
    head = api.get(url, headers={"Range": "bytes=0-1023"})
    assert head.status_code == 206 and len(head.content) == 1024
    assert head.headers["content-range"].startswith("bytes 0-1023/")
    assert head.content[4:8] == b"ftyp"
    full = api.get(url)
    assert full.status_code == 200 and len(full.content) == names["annotated.mp4"]["bytes"]

    timeline = api.get(results["timeline_url"]).json()
    # Every sampled frame is in the timeline (frames whose inference failed are kept and labelled).
    assert timeline["job_id"] == job_id and len(timeline["frames"]) >= summary["analyzed_frames"]
    assert all(f["mode"] == "video_analysis" and f["command"] is None for f in timeline["frames"])

    events = _sse_events(api, until='"succeeded"')
    job_events = [e for e in events if e["resource_id"] == job_id]
    states = [e["payload"]["state"] for e in job_events if e["type"] == "job_state"]
    assert "queued" in states and "succeeded" in states
    assert any(e["type"] == "job_progress" for e in job_events)

    deleted = api.delete(f"/api/v1/jobs/{job_id}")
    assert deleted.status_code == 200 and deleted.json()["status"] == "deleted"
    assert api.get(f"/api/v1/jobs/{job_id}").status_code == 404
    assert api.get(url).status_code == 404


def test_invalid_upload_rejected(api: Api) -> None:
    response = api.post("/api/v1/videos", files={"file": ("x.mp4", b"not a video" * 10, "video/mp4")})
    assert response.status_code in (415, 503)
    body: dict[str, Any] = response.json()
    assert set(body) == {"code", "message", "request_id", "details"}
