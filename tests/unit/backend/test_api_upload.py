# ruff: noqa: F811, S105, S106 - pytest fixtures are imported from the shared harness; test-only passwords
"""POST /api/v1/videos: streaming validation, admission, readiness and storage (docs/API.md, spec §3.3)."""

from __future__ import annotations

import functools
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from rq import Queue
from test_api_harness import Harness, build_harness, harness, mp4_bytes, sha256  # noqa: F401

from app.api.videos import sniff_container


def _files_under(path: Path) -> list[Path]:
    return [p for p in path.rglob("*") if p.is_file()] if path.exists() else []


def _assert_no_leftovers(h: Harness) -> None:
    assert _files_under(h.settings.tmp_root) == []
    assert _files_under(h.settings.artifact_root / "jobs") == []


def test_upload_creates_queued_job_and_enqueues(harness: Harness) -> None:
    client, info = harness.login()
    data = mp4_bytes(50_000)
    response = harness.upload(client, info, data)
    assert response.status_code == 202, response.text
    job = response.json()
    assert job["state"] == "queued" and job["kind"] == "video_analysis"
    assert job["input"] == {"sha256": sha256(data), "bytes": len(data), "ext": "mp4"}
    assert job["analysis_fps"] == harness.settings.analysis_fps
    stored = harness.settings.artifact_root / "jobs" / job["id"] / "input.mp4"
    assert stored.read_bytes() == data
    assert _files_under(harness.settings.tmp_root) == []
    queue = Queue("video", connection=harness.redis)
    assert queue.job_ids == [f"video-{job['id']}"]
    rq_job = queue.fetch_job(f"video-{job['id']}")
    assert rq_job is not None and rq_job.func_name == "app.jobs.video_job.run_video_job" and rq_job.args == (job["id"],)
    assert rq_job.timeout and rq_job.timeout > 0
    events = harness.events(_owner(harness, job["id"]))
    assert events[-1]["type"] == "job_state" and events[-1]["payload"]["state"] == "queued"


def _owner(h: Harness, job_id: str) -> str:
    row = h.services.db.execute("SELECT owner_id FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return row[0]


def test_mov_brand_is_stored_as_mov(harness: Harness) -> None:
    client, info = harness.login()
    response = harness.upload(client, info, mp4_bytes(2048, brand=b"qt  "), filename="x.mp4")
    assert response.status_code == 202
    job = response.json()
    assert job["input"]["ext"] == "mov"
    assert (harness.settings.artifact_root / "jobs" / job["id"] / "input.mov").is_file()


def test_client_filename_never_used_or_logged(harness: Harness, caplog: pytest.LogCaptureFixture) -> None:
    client, info = harness.login()
    evil = "../../secret-name-ÆØÅ.mp4"
    with caplog.at_level(logging.DEBUG):
        response = harness.upload(client, info, mp4_bytes(), filename=evil)
    assert response.status_code == 202
    job_dir = harness.settings.artifact_root / "jobs" / response.json()["id"]
    assert [p.name for p in job_dir.iterdir()] == ["input.mp4"]
    assert "secret-name" not in caplog.text
    assert not any("secret-name" in str(p) for p in harness.tmp.rglob("*"))


@pytest.fixture
def small_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    yield from build_harness(tmp_path, monkeypatch, max_upload_bytes=8192)


def test_oversized_upload_rejected_while_streaming(small_harness: Harness) -> None:
    h = small_harness
    client, info = h.login()
    response = h.upload(client, info, mp4_bytes(8193))
    assert response.status_code == 413 and response.json()["code"] == "upload_too_large"
    assert response.json()["details"]["limit_bytes"] == 8192
    _assert_no_leftovers(h)
    ok = h.upload(client, info, mp4_bytes(8192))
    assert ok.status_code == 202


def test_declared_content_length_over_cap_rejected_before_reading(small_harness: Harness) -> None:
    h = small_harness
    client, info = h.login()
    response = client.post(
        "/api/v1/videos",
        content=b"x" * 10,
        headers={**h.csrf(info), "Content-Type": "multipart/form-data; boundary=abc", "Content-Length": "999999999"},
    )
    assert response.status_code == 413
    _assert_no_leftovers(h)


@pytest.mark.parametrize(
    "payload",
    [
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 200,  # image
        b"PK\x03\x04" + b"\x00" * 200,  # zip (decompression bomb candidate)
        b"<html><script>alert(1)</script></html>" + b" " * 100,
        b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 100,  # HEIF image, not a video brand
        b"\x1aE\xdf\xa3" + b"\x00" * 200,  # Matroska/WebM
    ],
)
def test_wrong_magic_rejected(harness: Harness, payload: bytes) -> None:
    client, info = harness.login()
    response = harness.upload(client, info, payload, filename="video.mp4")
    assert response.status_code == 415 and response.json()["code"] == "unsupported_media_type"
    _assert_no_leftovers(harness)


def test_empty_missing_and_non_multipart(harness: Harness) -> None:
    client, info = harness.login()
    empty = harness.upload(client, info, b"")
    assert empty.status_code == 422 and empty.json()["code"] == "upload_empty"
    missing = client.post(
        "/api/v1/videos", files={"other": ("a.mp4", mp4_bytes(), "video/mp4")}, headers=harness.csrf(info)
    )
    assert missing.status_code == 422 and missing.json()["details"] == {"field": "file"}
    raw = client.post(
        "/api/v1/videos", content=mp4_bytes(), headers={**harness.csrf(info), "Content-Type": "video/mp4"}
    )
    assert raw.status_code == 415
    _assert_no_leftovers(harness)


def test_model_not_ready_rejects_before_reading(harness: Harness) -> None:
    harness.inference.ready = False
    harness.inference.reason = "hash_mismatch"
    harness.services.model_status.invalidate()
    client, info = harness.login()
    response = harness.upload(client, info)
    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "model_not_ready" and body["details"]["reason"] == "hash_mismatch"
    _assert_no_leftovers(harness)


def test_inference_unreachable_is_model_not_ready(harness: Harness) -> None:
    harness.inference.unreachable = True
    harness.services.model_status.invalidate()
    client, info = harness.login()
    response = harness.upload(client, info)
    assert response.status_code == 503 and response.json()["details"]["reason"] == "inference_unreachable"


def test_model_status_is_cached(harness: Harness) -> None:
    client, info = harness.login()
    harness.services.model_status.invalidate()
    before = harness.inference.calls
    for _ in range(3):
        assert harness.upload(client, info).status_code == 202
    assert harness.inference.calls - before == 1


@pytest.fixture
def quota_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    yield from build_harness(tmp_path, monkeypatch, max_queued_per_session=2, max_queued_total=3)


def test_session_and_global_queue_limits(quota_harness: Harness) -> None:
    h = quota_harness
    a, ainfo = h.login()
    assert h.upload(a, ainfo).status_code == 202
    assert h.upload(a, ainfo).status_code == 202
    third = h.upload(a, ainfo)
    assert third.status_code == 429 and third.json()["code"] == "session_queue_full"
    assert "retry-after" in third.headers
    b, binfo = h.login()
    assert h.upload(b, binfo).status_code == 202
    full = h.upload(b, binfo)
    assert full.status_code == 429 and full.json()["code"] == "queue_full"
    assert len(list((h.settings.artifact_root / "jobs").iterdir())) == 3
    assert list(h.settings.tmp_root.iterdir()) == []


@pytest.fixture
def limited_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    yield from build_harness(tmp_path, monkeypatch, max_requests_per_minute=2)


def _freeze_rate_window(h: Harness) -> None:
    # Pin the limiter clock inside one window so the test cannot straddle a minute boundary.
    h.services.limiter.hit = functools.partial(h.services.limiter.hit, now=1_000_020.0)  # type: ignore[method-assign]


def test_expensive_requests_rate_limited_per_session_and_ip(limited_harness: Harness) -> None:
    h = limited_harness
    _freeze_rate_window(h)
    a, ainfo = h.login()
    assert h.upload(a, ainfo, mp4_bytes(100)).status_code == 202
    assert h.upload(a, ainfo, mp4_bytes(100)).status_code == 202
    limited = h.upload(a, ainfo, mp4_bytes(100))
    assert limited.status_code == 429 and limited.json()["code"] == "rate_limited"
    assert int(limited.headers["retry-after"]) >= 1
    # A second session from the same client IP shares the per-IP budget.
    b, binfo = h.login()
    assert h.upload(b, binfo, mp4_bytes(100)).json()["code"] == "rate_limited"


def test_upload_requires_session_and_csrf(harness: Harness) -> None:
    anon = harness.new_client()
    assert anon.post("/api/v1/videos", files={"file": ("a.mp4", mp4_bytes(), "video/mp4")}).status_code == 401
    client, _ = harness.login()
    no_csrf = client.post("/api/v1/videos", files={"file": ("a.mp4", mp4_bytes(), "video/mp4")})
    assert no_csrf.status_code == 403
    _assert_no_leftovers(harness)


def test_redis_down_fails_closed(harness: Harness) -> None:
    client, info = harness.login()
    harness.server.connected = False
    try:
        response = harness.upload(client, info)
    finally:
        harness.server.connected = True
    assert response.status_code == 503 and response.json()["code"] == "service_unavailable"
    _assert_no_leftovers(harness)


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (b"\x00\x00\x00\x20ftypmp42\x00\x00\x00\x00mp42isomavc1", "mp4"),
        (b"\x00\x00\x00\x14ftypqt  \x00\x00\x02\x00qt  ", "mov"),
        (b"\x00\x00\x00\x18ftypM4V \x00\x00\x00\x01M4V mp42", "mp4"),
        (b"\x00\x00\x00\x18ftypXXXX\x00\x00\x00\x00YYYYqt  ", "mov"),
        (b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00mif1miaf", None),
        (b"\x00\x00\x00\x08moov\x00\x00\x00\x00\x00\x00\x00\x00", None),
        (b"\xff\xff\xff\xffftypisom\x00\x00\x00\x00isom", None),  # absurd box size
        (b"ftyp", None),
    ],
)
def test_sniff_container(head: bytes, expected: str | None) -> None:
    assert sniff_container(head + b"\x00" * 32) == expected
