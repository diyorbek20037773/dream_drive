# ruff: noqa: F811, S105, S106 - pytest fixtures are imported from the shared harness; test-only passwords
"""Oversized, corrupt and decompression-bomb-like uploads are rejected.

The API rejects by size and container signature while streaming. Content-level checks (corrupt streams, huge
decoded dimensions, excessive duration) are made by the worker with ffprobe before any decoding; those tests need
the ffprobe binary (present in the backend image). On a host without it they are skipped explicitly, and they fail
instead when DD_REQUIRE_LIVE=1 (make qa).
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_api_harness import Harness, build_harness, harness, mp4_bytes  # noqa: F401

from app.jobs import video_job

REQUIRE_LIVE = os.environ.get("DD_REQUIRE_LIVE") == "1"
HAVE_FFPROBE = shutil.which("ffprobe") is not None


def _require_ffprobe() -> None:
    if HAVE_FFPROBE:
        return
    if REQUIRE_LIVE:
        pytest.fail("ffprobe is required for content validation tests (DD_REQUIRE_LIVE=1)")
    pytest.skip("ffprobe binary not installed on this host (runs in the backend test image)")


@pytest.fixture
def small(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    yield from build_harness(tmp_path, monkeypatch, max_upload_bytes=64 * 1024)


def _leftovers(h: Harness) -> list[Path]:
    roots = [h.settings.tmp_root, h.settings.artifact_root / "jobs"]
    return [p for r in roots if r.exists() for p in r.rglob("*") if p.is_file()]


def test_chunked_body_without_length_is_capped(small: Harness) -> None:
    client, info = small.login()
    boundary = "dreamdriveboundary"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.mp4"\r\n'
        "Content-Type: video/mp4\r\n\r\n"
    ).encode() + mp4_bytes(64)

    def body() -> Iterator[bytes]:
        yield head
        for _ in range(64):
            yield b"\x00" * 4096  # 256 KiB total > 64 KiB cap, sent without Content-Length

    response = client.post(
        "/api/v1/videos",
        content=body(),
        headers={**small.csrf(info), "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert response.status_code == 413 and response.json()["code"] == "upload_too_large"
    assert _leftovers(small) == []


def test_large_non_file_fields_count_against_the_cap(small: Harness) -> None:
    client, info = small.login()
    response = client.post(
        "/api/v1/videos",
        data={"padding": "x" * (2 * 1024 * 1024)},
        files={"file": ("a.mp4", mp4_bytes(), "video/mp4")},
        headers=small.csrf(info),
    )
    assert response.status_code == 413
    assert _leftovers(small) == []


def test_oversized_json_bodies_rejected(harness: Harness) -> None:
    client, info = harness.login()
    big = {"start_node": "A", "destination_node": "C", "pad": "x" * 200_000}
    response = client.post("/api/v1/runs", json=big, headers=harness.csrf(info))
    assert response.status_code == 413 and response.json()["code"] == "upload_too_large"


def test_only_first_file_part_is_kept(harness: Harness) -> None:
    client, info = harness.login()
    first, second = mp4_bytes(1000), mp4_bytes(3000)
    response = client.post(
        "/api/v1/videos",
        files=[("file", ("a.mp4", first, "video/mp4")), ("file", ("b.mp4", second, "video/mp4"))],
        headers=harness.csrf(info),
    )
    assert response.status_code == 202 and response.json()["input"]["bytes"] == len(first)


def _encode(path: Path, width: int, height: int, frames: int, fps: int) -> None:
    import av

    with av.open(str(path), "w", format="mp4") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        image = np.zeros((height, width, 3), np.uint8)
        for i in range(frames):
            image[:, :, 1] = (i * 13) % 255
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def _upload_and_validate(h: Harness, data: bytes, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    def must_not_run(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("the analysis pipeline must not run for a rejected input")

    monkeypatch.setattr(video_job, "run_analysis", must_not_run)
    client, info = h.login()
    response = h.upload(client, info, data)
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    assert video_job.run_video_job(job_id) == "failed"
    return client.get(f"/api/v1/jobs/{job_id}").json()


def test_huge_dimension_video_rejected_before_decoding(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_ffprobe()
    path = tmp_path / "uhd.mp4"
    _encode(path, 3840, 2160, frames=2, fps=1)  # tiny file, 4K decode: a decompression-heavy input
    job = _upload_and_validate(harness, path.read_bytes(), monkeypatch)
    assert job["error_code"] == "video_too_large_dimensions"


def test_overlong_video_rejected(harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _require_ffprobe()
    path = tmp_path / "long.mp4"
    _encode(path, 64, 64, frames=130, fps=1)  # 130 s > MAX_VIDEO_SECONDS=120
    job = _upload_and_validate(harness, path.read_bytes(), monkeypatch)
    assert job["error_code"] == "video_too_long"


def test_corrupt_video_with_valid_signature_fails_validation(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    _require_ffprobe()
    job = _upload_and_validate(harness, mp4_bytes(20_000), monkeypatch)
    assert job["state"] == "failed" and job["error_code"] in ("video_invalid", "video_no_stream")
    assert "/" not in (job["error_message"] or "")
