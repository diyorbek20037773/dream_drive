# ruff: noqa: F811, S105, S106 - pytest fixtures are imported from the shared harness; test-only passwords
"""GET /jobs/{id}/artifacts/{name}: allowlist, ownership, byte ranges, content types and dispositions."""

from __future__ import annotations

from typing import Any

import pytest
from test_api_harness import Harness, harness, mp4_bytes  # noqa: F401
from test_jobs_video import ANNOTATED, pipeline  # noqa: F401

from app.jobs import video_job
from app.storage.artifacts import RangeNotSatisfiable, parse_range


@pytest.fixture
def done_job(harness: Harness, pipeline: dict[str, Any]) -> tuple[Any, dict[str, Any], str]:
    client, info = harness.login()
    job_id = harness.upload(client, info, mp4_bytes(3000)).json()["id"]
    assert video_job.run_video_job(job_id) == "succeeded"
    return client, info, job_id


def test_full_download_headers(done_job: tuple[Any, dict[str, Any], str]) -> None:
    client, _info, job_id = done_job
    response = client.get(f"/api/v1/jobs/{job_id}/artifacts/annotated.mp4")
    assert response.status_code == 200
    assert response.content == ANNOTATED
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-length"] == str(len(ANNOTATED))
    assert response.headers["content-disposition"] == f'inline; filename="dreamdrive-{job_id[:8]}-annotated.mp4"'
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize(
    ("header", "start", "end"),
    [
        ("bytes=0-99", 0, 99),
        ("bytes=100-", 100, len(ANNOTATED) - 1),
        ("bytes=-10", len(ANNOTATED) - 10, len(ANNOTATED) - 1),
        ("bytes=5000-999999", 5000, len(ANNOTATED) - 1),
    ],
)
def test_range_requests(done_job: tuple[Any, dict[str, Any], str], header: str, start: int, end: int) -> None:
    client, _info, job_id = done_job
    response = client.get(f"/api/v1/jobs/{job_id}/artifacts/annotated.mp4", headers={"Range": header})
    assert response.status_code == 206
    assert response.headers["content-range"] == f"bytes {start}-{end}/{len(ANNOTATED)}"
    assert response.content == ANNOTATED[start : end + 1]
    assert response.headers["content-length"] == str(end - start + 1)


def test_unsatisfiable_and_ignored_ranges(done_job: tuple[Any, dict[str, Any], str]) -> None:
    client, _info, job_id = done_job
    url = f"/api/v1/jobs/{job_id}/artifacts/annotated.mp4"
    bad = client.get(url, headers={"Range": f"bytes={len(ANNOTATED)}-"})
    assert bad.status_code == 416 and bad.json()["code"] == "range_not_satisfiable"
    assert bad.headers["content-range"] == f"bytes */{len(ANNOTATED)}"
    for ignored in ("bytes=0-1,5-6", "items=0-5", "bytes=9-3", "garbage"):
        full = client.get(url, headers={"Range": ignored})
        assert full.status_code == 200 and full.content == ANNOTATED, ignored


def test_types_and_dispositions(done_job: tuple[Any, dict[str, Any], str]) -> None:
    client, _info, job_id = done_job
    base = f"/api/v1/jobs/{job_id}/artifacts"
    html = client.get(f"{base}/report.html")
    assert html.headers["content-type"].startswith("text/html")
    assert html.headers["content-disposition"].startswith("attachment;")
    assert "sandbox" in html.headers["content-security-policy"]
    report = client.get(f"{base}/report.json")
    assert report.headers["content-type"] == "application/json" and report.json()["limitations"]
    poster = client.get(f"{base}/poster.jpg")
    assert poster.headers["content-type"] == "image/jpeg" and poster.headers["content-disposition"].startswith("inline")
    original = client.get(f"{base}/input")
    assert original.status_code == 200 and original.headers["content-type"] == "video/mp4"
    assert original.content == mp4_bytes(3000)
    assert f"dreamdrive-{job_id[:8]}-input.mp4" in original.headers["content-disposition"]


def test_unlisted_names_and_missing_outputs(harness: Harness, pipeline: dict[str, Any]) -> None:
    client, info = harness.login()
    job_id = harness.upload(client, info).json()["id"]
    pending = client.get(f"/api/v1/jobs/{job_id}/artifacts/annotated.mp4")
    assert pending.status_code == 409 and pending.json()["code"] == "artifact_unavailable"
    assert client.get(f"/api/v1/jobs/{job_id}/artifacts/input").status_code == 200
    for name in (
        "dreamdrive.db",
        "input.mp4",
        "annotated.mp4.part",
        "%2e%2e%2fdreamdrive.db",
        "..%2f..%2fdreamdrive.db",
    ):
        response = client.get(f"/api/v1/jobs/{job_id}/artifacts/{name}")
        assert response.status_code in (404, 422), name
        assert "SQLite" not in response.text


def test_parse_range_unit() -> None:
    assert parse_range(None, 10) is None
    assert parse_range("bytes=0-0", 10) == (0, 0)
    assert parse_range("bytes=-20", 10) == (0, 9)
    with pytest.raises(RangeNotSatisfiable):
        parse_range("bytes=10-", 10)
    with pytest.raises(RangeNotSatisfiable):
        parse_range("bytes=-0", 10)
    with pytest.raises(RangeNotSatisfiable):
        parse_range("bytes=0-", 0)
