"""model_init download/verify/rename behaviour against a local loopback HTTP server (no internet).

Both required checkpoints (yolopv2.pt and rf-detr-nano.pth) are bootstrapped; the RF-DETR entry also carries a
publisher MD5 that is checked while streaming.
"""

from __future__ import annotations

import hashlib
import http.server
import socket
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from conftest import rfdetr_entry, write_manifest

from app import model_init

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB of deterministic bytes
PAYLOAD_MD5 = hashlib.md5(PAYLOAD, usedforsecurity=False).hexdigest()


class _Handler(http.server.BaseHTTPRequestHandler):
    payload = PAYLOAD
    hits: list[str] = []

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        type(self).hits.append(self.path)
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/model.pt")
            self.end_headers()
            return
        if self.path == "/missing":
            self.send_response(404)
            self.end_headers()
            return
        body = self.payload if self.path == "/model.pt" else b"tampered" + self.payload[8:]
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        return


@pytest.fixture
def server() -> Iterator[str]:
    _Handler.hits = []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def init_settings(tmp_path: Path, make_settings: Callable[..., Any]) -> Callable[[str], Any]:
    def _make(
        url: str, sha: str | None = None, size: int | None = None, rf_url: str | None = None, rf_md5: str = PAYLOAD_MD5
    ) -> Any:
        manifest = write_manifest(
            tmp_path / "m.yaml",
            file_name="yolopv2.pt",
            sha256=sha or hashlib.sha256(PAYLOAD).hexdigest(),
            size=size or len(PAYLOAD),
            url=url,
            rfdetr=rfdetr_entry(
                sha256=sha or hashlib.sha256(PAYLOAD).hexdigest(),
                size=len(PAYLOAD),
                url=rf_url or url,
                md5=rf_md5,
            ),
        )
        return make_settings(
            model_root=tmp_path / "models",
            model_manifest_path=manifest,
            model_init_connect_timeout_s=2,
            model_init_read_timeout_s=2,
        )

    return _make


def test_download_verify_and_atomic_rename(server: str, init_settings: Callable[..., Any]) -> None:
    settings = init_settings(f"{server}/redirect")
    assert model_init.run(settings) == model_init.EXIT_OK
    for name in ("yolopv2.pt", "rf-detr-nano.pth"):
        assert (settings.model_root / name).read_bytes() == PAYLOAD
        assert not (settings.model_root / f"{name}.part").exists()
    assert _Handler.hits == ["/redirect", "/model.pt", "/redirect", "/model.pt"]


def test_single_model_selection(server: str, init_settings: Callable[..., Any]) -> None:
    settings = init_settings(f"{server}/model.pt")
    assert model_init.run(settings, ["yolopv2"]) == model_init.EXIT_OK
    assert sorted(p.name for p in settings.model_root.iterdir()) == ["yolopv2.pt"]
    assert _Handler.hits == ["/model.pt"]


def test_publisher_md5_mismatch_exits_2(server: str, init_settings: Callable[..., Any]) -> None:
    """Manifest MD5 inconsistent with the bytes (SHA256 matches): the RF-DETR download is rejected and deleted."""
    settings = init_settings(f"{server}/model.pt", rf_md5="0" * 32)
    assert model_init.run(settings) == model_init.EXIT_MISMATCH
    assert (settings.model_root / "yolopv2.pt").read_bytes() == PAYLOAD
    assert not (settings.model_root / "rf-detr-nano.pth").exists()
    assert not (settings.model_root / "rf-detr-nano.pth.part").exists()


def test_existing_file_with_inconsistent_publisher_md5_is_manifest_error(
    init_settings: Callable[..., Any],
) -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{s.getsockname()[1]}/model.pt"
    settings = init_settings(dead, rf_md5="0" * 32)
    settings.model_root.mkdir(parents=True)
    for name in ("yolopv2.pt", "rf-detr-nano.pth"):
        (settings.model_root / name).write_bytes(PAYLOAD)
    assert model_init.run(settings) == model_init.EXIT_MANIFEST


def test_already_verified_file_needs_no_network(init_settings: Callable[..., Any]) -> None:
    with socket.socket() as s:  # a port nobody listens on: any network access would fail
        s.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{s.getsockname()[1]}/model.pt"
    settings = init_settings(dead)
    settings.model_root.mkdir(parents=True)
    (settings.model_root / "yolopv2.pt").write_bytes(PAYLOAD)
    (settings.model_root / "rf-detr-nano.pth").write_bytes(PAYLOAD)
    assert model_init.run(settings) == model_init.EXIT_OK


def test_hash_mismatch_deletes_part_and_exits_2(server: str, init_settings: Callable[..., Any]) -> None:
    settings = init_settings(f"{server}/tampered")  # same size, different bytes
    assert model_init.run(settings) == model_init.EXIT_MISMATCH
    assert not (settings.model_root / "yolopv2.pt").exists()
    assert not (settings.model_root / "yolopv2.pt.part").exists()


def test_size_mismatch_exits_2(server: str, init_settings: Callable[..., Any]) -> None:
    settings = init_settings(f"{server}/model.pt", size=len(PAYLOAD) - 1)
    assert model_init.run(settings) == model_init.EXIT_MISMATCH
    assert sorted(p.name for p in settings.model_root.iterdir()) == ["rf-detr-nano.pth"]  # only the verified one


def test_bad_existing_file_is_replaced_only_by_verified_download(
    server: str, init_settings: Callable[..., Any]
) -> None:
    settings = init_settings(f"{server}/model.pt")
    settings.model_root.mkdir(parents=True)
    (settings.model_root / "yolopv2.pt").write_bytes(b"corrupt")
    assert model_init.run(settings) == model_init.EXIT_OK
    assert (settings.model_root / "yolopv2.pt").read_bytes() == PAYLOAD


def test_http_error_and_unreachable_host_exit_1(server: str, init_settings: Callable[..., Any]) -> None:
    assert model_init.run(init_settings(f"{server}/missing")) == model_init.EXIT_DOWNLOAD_ERROR
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{s.getsockname()[1]}/model.pt"
    settings = init_settings(dead)
    assert model_init.run(settings) == model_init.EXIT_DOWNLOAD_ERROR
    assert not (settings.model_root / "yolopv2.pt.part").exists()


def test_invalid_manifest_exits_3(tmp_path: Path, make_settings: Callable[..., Any]) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("{}", encoding="utf-8")
    assert model_init.run(make_settings(model_manifest_path=bad)) == model_init.EXIT_MANIFEST
