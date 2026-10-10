"""Resumable model download: interrupted transfers keep the .part file and continue with an HTTP Range request.

A loopback server cuts the first response short; the downloader must resume from the kept bytes, verify the SHA256
and publisher MD5 over the whole file, and fall back to a full download when the server ignores Range.
"""

from __future__ import annotations

import hashlib
import http.server
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from conftest import rfdetr_entry, write_manifest

from app import model_init

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB
SHA = hashlib.sha256(PAYLOAD).hexdigest()
MD5 = hashlib.md5(PAYLOAD, usedforsecurity=False).hexdigest()
CUT = 300_000


class _FlakyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    ranges: list[str | None] = []
    honour_range = True
    cut_first = True

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        cls = type(self)
        rng = self.headers.get("Range")
        cls.ranges.append(rng)
        if rng and cls.honour_range:
            start = int(rng.removeprefix("bytes=").split("-")[0])
            body = PAYLOAD[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        else:
            body = PAYLOAD
            self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if cls.cut_first and len(cls.ranges) == 1:
            self.wfile.write(body[:CUT])  # connection drops mid-transfer
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        return


@pytest.fixture
def flaky() -> Iterator[str]:
    _FlakyHandler.ranges = []
    _FlakyHandler.honour_range = True
    _FlakyHandler.cut_first = True
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FlakyHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/model.pt"
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def settings_for(tmp_path: Path, make_settings: Callable[..., Any]) -> Callable[..., Any]:
    def _make(url: str, attempts: int = 3, sha: str = SHA) -> Any:
        manifest = write_manifest(
            tmp_path / "m.yaml",
            file_name="yolopv2.pt",
            sha256=sha,
            size=len(PAYLOAD),
            url=url,
            rfdetr=rfdetr_entry(sha256=sha, size=len(PAYLOAD), url=url, md5=MD5),
        )
        return make_settings(
            model_root=tmp_path / "models",
            model_manifest_path=manifest,
            model_init_connect_timeout_s=2,
            model_init_read_timeout_s=2,
            model_init_attempts=attempts,
            model_init_retry_backoff_s=0,
        )

    return _make


def test_interrupted_download_resumes_with_range(flaky: str, settings_for: Callable[..., Any]) -> None:
    settings = settings_for(flaky)
    assert model_init.run(settings, ["rfdetr_nano"]) == model_init.EXIT_OK
    target = settings.model_root / "rf-detr-nano.pth"
    assert hashlib.sha256(target.read_bytes()).hexdigest() == SHA
    assert not target.with_name(target.name + ".part").exists()
    assert _FlakyHandler.ranges[0] is None
    assert _FlakyHandler.ranges[1] == f"bytes={CUT}-"


def test_server_ignoring_range_restarts_from_zero(flaky: str, settings_for: Callable[..., Any]) -> None:
    _FlakyHandler.honour_range = False
    settings = settings_for(flaky)
    assert model_init.run(settings, ["rfdetr_nano"]) == model_init.EXIT_OK
    target = settings.model_root / "rf-detr-nano.pth"
    assert target.read_bytes() == PAYLOAD


def test_partial_file_is_kept_across_runs_then_resumed(flaky: str, settings_for: Callable[..., Any]) -> None:
    settings = settings_for(flaky, attempts=1)
    assert model_init.run(settings, ["rfdetr_nano"]) == model_init.EXIT_DOWNLOAD_ERROR
    part = settings.model_root / "rf-detr-nano.pth.part"
    assert part.stat().st_size == CUT
    assert model_init.run(settings, ["rfdetr_nano"]) == model_init.EXIT_OK
    assert (settings.model_root / "rf-detr-nano.pth").read_bytes() == PAYLOAD
    assert _FlakyHandler.ranges[-1] == f"bytes={CUT}-"


def test_resumed_bytes_still_verified_against_manifest(flaky: str, settings_for: Callable[..., Any]) -> None:
    settings = settings_for(flaky, sha="0" * 64)
    assert model_init.run(settings, ["rfdetr_nano"]) == model_init.EXIT_MISMATCH
    assert not (settings.model_root / "rf-detr-nano.pth").exists()
    assert not (settings.model_root / "rf-detr-nano.pth.part").exists()
