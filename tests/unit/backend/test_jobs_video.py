# ruff: noqa: F811, S105, S106 - pytest fixtures are imported from the shared harness; test-only passwords
"""Video job lifecycle through the real worker entry point with the pipeline replaced at its module boundary.

The pipeline itself (app.video.pipeline) is tested by the video component; here we verify the job state machine,
progress/events, cancellation, deletion coordination and results/artifact exposure.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
from rq import Queue
from test_api_harness import Harness, harness, mp4_bytes, run_in_thread  # noqa: F401

from app.jobs import video_job
from app.jobs.context import SHUTDOWN
from app.storage import repo
from app.video.errors import JobCancelled, VideoError

PROBE = {"duration_s": 10.0, "width": 1280, "height": 720, "fps_avg": 30.0, "rotation": 0, "codec": "h264"}
SUMMARY = {
    "analyzed_frames": 3,
    "decoded_frames": 30,
    "coverage_ratio": 1.0,
    "duration_s": 10.0,
    "effective_analyzed_fps": 1.5,
    "timing_ms": {
        "decode": 10.0,
        "inference_p50": 400.0,
        "inference_p95": 420.0,
        "postprocess": 5.0,
        "encode": 3.0,
        "total_wall": 2000.0,
    },
    "decision_counts": {"observe": 3},
    "reason_counts": {"LANE_VALID": 3},
    "tracks": [{"track_id": 1, "class_name": "vehicle", "first_t": 0.0, "last_t": 0.4, "frames": 3, "max_score": 0.9}],
    "class_counts": {"vehicle": 3},
    "audio": "removed",
}
ANNOTATED = bytes(range(256)) * 40  # 10240 bytes


def write_outputs(out_dir: Path) -> None:
    (out_dir / "annotated.mp4").write_bytes(ANNOTATED)
    (out_dir / "poster.jpg").write_bytes(b"\xff\xd8poster\xff\xd9")
    (out_dir / "timeline.json").write_text(json.dumps({"schema_version": "1.0", "frames": []}), encoding="utf-8")
    (out_dir / "report.html").write_text("<html><body>report</body></html>", encoding="utf-8")
    (out_dir / "report.json").write_text(
        json.dumps({"limitations": ["Only class 3 (vehicle) is verified for this checkpoint."]}), encoding="utf-8"
    )


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace probe + pipeline; ``state['behavior']`` selects success / error / cancel-aware loops."""
    state: dict[str, Any] = {"behavior": "success", "calls": [], "hook": None}

    def fake_probe(input_path: Path, ctx: Any) -> dict[str, Any]:
        assert input_path.is_file()
        if state.get("probe_error"):
            raise VideoError(state["probe_error"], "Video exceeds the configured limits")
        return dict(PROBE)

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        state["calls"].append(kwargs)
        progress, cancel_check, out_dir = kwargs["progress_cb"], kwargs["cancel_check"], kwargs["out_dir"]
        progress(0, 0, "probing")
        behavior = state["behavior"]
        if behavior == "video_error":
            raise VideoError("video_decode_failed", "no frames could be decoded")
        if behavior == "crash":
            raise RuntimeError("secret internal detail at /data/artifacts/jobs")
        for i in range(1, 4):
            if state["hook"] is not None and i == 2:
                state["hook"]()
            if cancel_check():
                raise JobCancelled()
            progress(i, 3, "analyzing")
        if behavior == "wait_for_cancel":
            for _ in range(500):
                if cancel_check():
                    raise JobCancelled()
                threading.Event().wait(0.01)
        progress(3, 3, "encoding")
        write_outputs(Path(out_dir))
        progress(3, 3, "done")
        return dict(SUMMARY)

    monkeypatch.setattr(video_job, "probe_input", fake_probe)
    monkeypatch.setattr(video_job, "run_analysis", fake_run)
    monkeypatch.setattr(video_job, "CANCEL_DB_POLL_S", 0.0)
    return state


def _upload(h: Harness) -> tuple[Any, dict[str, Any], str]:
    client, info = h.login()
    response = h.upload(client, info, mp4_bytes(4096))
    assert response.status_code == 202, response.text
    return client, info, response.json()["id"]


def _owner(h: Harness, job_id: str) -> str:
    row = repo.get_job_any(h.services.db, job_id)
    assert row is not None
    return row.owner_id


def test_successful_job_lifecycle(harness: Harness, pipeline: dict[str, Any]) -> None:
    client, _info, job_id = _upload(harness)
    assert video_job.run_video_job(job_id) == "succeeded"
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["state"] == "succeeded" and job["progress"] == 1.0 and job["stage"] == "done"
    assert job["video"]["width"] == 1280 and job["model"]["model_id"] == "yolopv2"
    assert job["started_utc"] and job["finished_utc"] and job["frames_done"] == 3
    call = pipeline["calls"][0]
    assert call["job_id"] == job_id and call["input_path"].name == "input.mp4" and call["settings"] is harness.settings

    results = client.get(f"/api/v1/jobs/{job_id}/results").json()
    assert results["state"] == "succeeded" and results["mode"] == "video_analysis"
    assert results["perception_source"] == "learned_model" and results["summary"]["analyzed_frames"] == 3
    names = [a["name"] for a in results["artifacts"]]
    assert names == ["annotated.mp4", "poster.jpg", "timeline.json", "report.json", "report.html", "input"]
    annotated = results["artifacts"][0]
    assert annotated["url"] == f"/api/v1/jobs/{job_id}/artifacts/annotated.mp4"
    assert annotated["bytes"] == len(ANNOTATED) and annotated["content_type"] == "video/mp4"
    assert results["timeline_url"].endswith("/timeline.json")
    assert any("vehicle" in x for x in results["limitations"])
    assert any("observation recording" in x for x in results["limitations"])

    events = harness.events(_owner(harness, job_id))
    states = [e["payload"]["state"] for e in events if e["type"] == "job_state"]
    assert states == ["queued", "validating", "running", "succeeded"]
    progress = [e["payload"] for e in events if e["type"] == "job_progress"]
    assert progress and progress[-1]["stage"] == "done" and progress[-1]["progress"] == 1.0
    assert all(0.0 <= p["progress"] <= 1.0 for p in progress)


def test_video_error_marks_failed_with_code(harness: Harness, pipeline: dict[str, Any]) -> None:
    pipeline["behavior"] = "video_error"
    client, _info, job_id = _upload(harness)
    assert video_job.run_video_job(job_id) == "failed"
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["state"] == "failed" and job["error_code"] == "video_decode_failed"
    unavailable = client.get(f"/api/v1/jobs/{job_id}/results")
    assert unavailable.status_code == 409 and unavailable.json()["code"] == "results_unavailable"


def test_probe_rejection_fails_during_validation(harness: Harness, pipeline: dict[str, Any]) -> None:
    pipeline["probe_error"] = "video_too_large_dimensions"
    client, _info, job_id = _upload(harness)
    assert video_job.run_video_job(job_id) == "failed"
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["error_code"] == "video_too_large_dimensions" and pipeline["calls"] == []


def test_unexpected_crash_hides_internals(harness: Harness, pipeline: dict[str, Any]) -> None:
    pipeline["behavior"] = "crash"
    client, _info, job_id = _upload(harness)
    assert video_job.run_video_job(job_id) == "failed"
    text = client.get(f"/api/v1/jobs/{job_id}").text
    assert "internal_error" in text and "/data" not in text and "secret" not in text


def test_cancel_queued_job_is_idempotent(harness: Harness, pipeline: dict[str, Any]) -> None:
    client, info, job_id = _upload(harness)
    first = client.post(f"/api/v1/jobs/{job_id}/cancel", headers=harness.csrf(info))
    assert first.status_code == 200 and first.json()["state"] == "cancelled"
    assert Queue("video", connection=harness.redis).job_ids == []
    again = client.post(f"/api/v1/jobs/{job_id}/cancel", headers=harness.csrf(info))
    assert again.json()["state"] == "cancelled" and again.json()["finished_utc"] == first.json()["finished_utc"]
    # A worker that still receives the RQ entry leaves the cancelled job alone.
    assert video_job.run_video_job(job_id) == "cancelled"
    assert pipeline["calls"] == []


def test_cancel_running_job_stops_between_frames(harness: Harness, pipeline: dict[str, Any]) -> None:
    pipeline["behavior"] = "wait_for_cancel"
    client, info, job_id = _upload(harness)
    worker = run_in_thread(lambda: video_job.run_video_job(job_id))
    harness.wait_for(lambda: repo.get_job_any(harness.services.db, job_id).state == "running")  # type: ignore[union-attr]
    response = client.post(f"/api/v1/jobs/{job_id}/cancel", headers=harness.csrf(info))
    assert response.json()["state"] in ("cancelling", "cancelled")
    worker.join(10)
    assert not worker.is_alive()
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["state"] == "cancelled"
    assert not (harness.settings.artifact_root / "jobs" / job_id / "annotated.mp4").exists()
    assert harness.redis.get(f"dd:job:{job_id}:cancel") is None
    # Cancelling a terminal job returns it unchanged.
    assert client.post(f"/api/v1/jobs/{job_id}/cancel", headers=harness.csrf(info)).json()["state"] == "cancelled"


def test_worker_shutdown_interrupts_job(harness: Harness, pipeline: dict[str, Any]) -> None:
    pipeline["hook"] = SHUTDOWN.set
    client, _info, job_id = _upload(harness)
    assert video_job.run_video_job(job_id) == "interrupted"
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["state"] == "interrupted" and job["error_code"] == "worker_shutdown"


def test_delete_terminal_job_removes_bytes_and_is_idempotent(harness: Harness, pipeline: dict[str, Any]) -> None:
    client, info, job_id = _upload(harness)
    video_job.run_video_job(job_id)
    job_dir = harness.settings.artifact_root / "jobs" / job_id
    assert job_dir.is_dir()
    first = client.delete(f"/api/v1/jobs/{job_id}", headers=harness.csrf(info))
    assert first.status_code == 200 and first.json() == {"id": job_id, "status": "deleted"}
    assert not job_dir.exists()
    assert client.get(f"/api/v1/jobs/{job_id}").status_code == 404
    again = client.delete(f"/api/v1/jobs/{job_id}", headers=harness.csrf(info))
    assert again.status_code == 200 and again.json()["status"] == "deleted"
    # Another session cannot use the tombstone to learn that the id existed.
    other, other_info = harness.login()
    assert other.delete(f"/api/v1/jobs/{job_id}", headers=harness.csrf(other_info)).status_code == 404


def test_delete_queued_job_cancels_and_removes(harness: Harness, pipeline: dict[str, Any]) -> None:
    client, info, job_id = _upload(harness)
    response = client.delete(f"/api/v1/jobs/{job_id}", headers=harness.csrf(info))
    assert response.json()["status"] == "deleted"
    assert repo.get_job_any(harness.services.db, job_id) is None
    assert Queue("video", connection=harness.redis).job_ids == []
    assert video_job.run_video_job(job_id) == "missing"


def test_delete_in_progress_job_is_cleaned_up_by_worker(harness: Harness, pipeline: dict[str, Any]) -> None:
    pipeline["behavior"] = "wait_for_cancel"
    client, info, job_id = _upload(harness)
    worker = run_in_thread(lambda: video_job.run_video_job(job_id))
    harness.wait_for(lambda: repo.get_job_any(harness.services.db, job_id).state == "running")  # type: ignore[union-attr]
    response = client.delete(f"/api/v1/jobs/{job_id}", headers=harness.csrf(info))
    assert response.json() == {"id": job_id, "status": "deletion_scheduled"}
    assert client.get(f"/api/v1/jobs/{job_id}").status_code == 404  # hidden immediately
    assert job_id not in [j["id"] for j in client.get("/api/v1/jobs").json()["items"]]
    again = client.delete(f"/api/v1/jobs/{job_id}", headers=harness.csrf(info))
    assert again.json()["status"] in ("deletion_scheduled", "deleted")
    worker.join(10)
    assert repo.get_job_any(harness.services.db, job_id) is None
    assert not (harness.settings.artifact_root / "jobs" / job_id).exists()
    assert client.delete(f"/api/v1/jobs/{job_id}", headers=harness.csrf(info)).json()["status"] == "deleted"


def test_list_pagination_newest_first(harness: Harness, pipeline: dict[str, Any]) -> None:
    client, info = harness.login()
    ids = [harness.upload(client, info).json()["id"] for _ in range(3)]
    page1 = client.get("/api/v1/jobs", params={"limit": 2}).json()
    assert [j["id"] for j in page1["items"]] == ids[::-1][:2] and page1["next_cursor"]
    page2 = client.get("/api/v1/jobs", params={"limit": 2, "cursor": page1["next_cursor"]}).json()
    assert [j["id"] for j in page2["items"]] == [ids[0]] and page2["next_cursor"] is None
    bad = client.get("/api/v1/jobs", params={"cursor": "%%%not-base64"})
    assert bad.status_code == 422
    assert client.get("/api/v1/jobs", params={"limit": 101}).status_code == 422


def test_progress_updates_are_persisted(harness: Harness, pipeline: dict[str, Any]) -> None:
    seen: list[tuple[str, float]] = []

    def hook() -> None:
        row = repo.get_job_any(harness.services.db, job_id)
        assert row is not None
        seen.append((row.state, row.progress))

    pipeline["hook"] = hook
    _client, _info, job_id = _upload(harness)
    video_job.run_video_job(job_id)
    assert seen and seen[0][0] == "running" and 0.0 < seen[0][1] < 0.95
