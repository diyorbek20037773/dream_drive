# ruff: noqa: F811, S105, S106 - pytest fixtures are imported from the shared harness; test-only passwords
"""CSRF on every cookie-authenticated mutation, session requirement and operator-only admin endpoint."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from test_api_harness import Harness, build_harness, harness  # noqa: F401

PLACEHOLDER_ID = "d" * 32


def _mutations(h: Harness) -> list[tuple[str, str]]:
    doc = h.client.app.openapi()  # type: ignore[attr-defined]
    ops = []
    for path, methods in doc["paths"].items():
        for method in methods:
            if method.upper() in ("POST", "DELETE") and path != "/api/v1/session":
                ops.append((method.upper(), path))
    return sorted(ops)


def _url(path: str) -> str:
    for param in ("{job_id}", "{run_id}", "{evaluation_id}"):
        path = path.replace(param, PLACEHOLDER_ID)
    return path


def _send(client: Any, method: str, url: str, headers: dict[str, str] | None = None) -> Any:
    if method == "DELETE":
        return client.delete(url, headers=headers or {})
    return client.post(url, json={}, headers=headers or {})


def test_mutation_inventory_is_complete(harness: Harness) -> None:
    ops = _mutations(harness)
    assert ("POST", "/api/v1/videos") in ops and ("DELETE", "/api/v1/jobs/{job_id}") in ops
    assert ("POST", "/api/v1/admin/cleanup") in ops and len(ops) >= 8


def test_every_mutation_requires_csrf(harness: Harness) -> None:
    client, info = harness.login()
    for method, path in _mutations(harness):
        url = _url(path)
        missing = _send(client, method, url)
        assert missing.status_code == 403 and missing.json()["code"] == "csrf_failed", (method, path)
        wrong = _send(client, method, url, {"X-CSRF-Token": info["csrf_token"][:-1] + "x"})
        assert wrong.status_code == 403, (method, path)
        # A cross-site page can send the cookie but not read/set the token; Origin does not help it.
        forged = _send(client, method, url, {"Origin": "https://evil.example"})
        assert forged.status_code == 403, (method, path)


def test_every_protected_route_requires_session(harness: Harness) -> None:
    anon = harness.new_client()
    doc = harness.client.app.openapi()  # type: ignore[attr-defined]
    for path, methods in doc["paths"].items():
        if not path.startswith("/api/v1/") or path in ("/api/v1/session", "/api/v1/capabilities"):
            continue
        for method in methods:
            url = _url(path).replace("{name}", "input")
            response = anon.request(method.upper(), url, json={} if method.upper() == "POST" else None)
            assert response.status_code == 401, (method, path, response.status_code)
            assert response.json()["code"] == "session_required"


@pytest.fixture
def op_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    yield from build_harness(tmp_path, monkeypatch, admin_password="operator-pass-123")


def test_admin_cleanup_requires_operator(op_harness: Harness) -> None:
    h = op_harness
    anon_client = h.new_client()
    assert anon_client.post("/api/v1/admin/cleanup").status_code == 401
    client, anon = h.login()
    denied = client.post("/api/v1/admin/cleanup", headers=h.csrf(anon))
    assert denied.status_code == 403 and denied.json()["code"] == "operator_required"
    _, op = h.login(client, password="operator-pass-123")
    assert client.post("/api/v1/admin/cleanup").status_code == 403  # operator still needs CSRF
    assert client.post("/api/v1/admin/cleanup", headers=h.csrf(op)).status_code == 200
    # The stale anonymous CSRF token does not authorize the operator session.
    assert client.post("/api/v1/admin/cleanup", headers=h.csrf(anon)).status_code == 403
