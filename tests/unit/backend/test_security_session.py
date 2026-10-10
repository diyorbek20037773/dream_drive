# ruff: noqa: F811, S105, S106 - pytest fixtures are imported from the shared harness; test-only passwords
"""Sessions, cookies, CSRF, operator login and rate limiting (docs/IMPLEMENTATION_SPEC.md §3.3)."""

from __future__ import annotations

import functools
from collections.abc import Iterator
from pathlib import Path

import fakeredis
import pytest
from test_api_harness import Harness, build_harness, harness  # noqa: F401

from app.config import Settings
from app.security import RateLimiter, SessionSigner, admin_password_matches, csrf_matches

ENVELOPE = {"code", "message", "request_id", "details"}


def _set_cookie_headers(response) -> list[str]:  # type: ignore[no-untyped-def]
    return response.headers.get_list("set-cookie")


def test_session_sets_signed_httponly_cookie_and_readable_csrf(harness: Harness) -> None:
    client = harness.new_client()
    response = client.post("/api/v1/session")
    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "anonymous"
    assert body["limits"]["max_upload_bytes"] == harness.settings.max_upload_bytes
    cookies = {c.split("=", 1)[0]: c for c in _set_cookie_headers(response)}
    session_cookie = cookies["dd_session"].lower()
    assert "httponly" in session_cookie and "samesite=lax" in session_cookie and "path=/" in session_cookie
    assert "secure" not in session_cookie  # APP_ENV=test
    csrf_cookie = cookies["dd_csrf"].lower()
    assert "httponly" not in csrf_cookie and body["csrf_token"].lower() in csrf_cookie
    # The cookie holds a signature, not the bare id.
    session_value = client.cookies.get("dd_session")
    assert session_value and "." in session_value
    assert response.headers["cache-control"] == "no-store"


def test_production_cookies_are_secure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gen = build_harness(tmp_path, monkeypatch, app_env="production")
    h = next(gen)
    try:
        response = h.new_client().post("/api/v1/session")
        for header in _set_cookie_headers(response):
            assert "secure" in header.lower()
    finally:
        gen.close()


def test_refresh_keeps_session_identity(harness: Harness) -> None:
    client, first = harness.login()
    _, second = harness.login(client)
    assert second["session_id_suffix"] == first["session_id_suffix"]
    assert second["csrf_token"] == first["csrf_token"]


def test_missing_forged_and_expired_session_rejected(harness: Harness) -> None:
    anon = harness.new_client()
    response = anon.get("/api/v1/jobs")
    assert response.status_code == 401 and response.json()["code"] == "session_required"
    assert set(response.json()) == ENVELOPE

    client, _ = harness.login()
    value = client.cookies.get("dd_session")
    assert value
    forged = harness.new_client()
    forged.cookies.set("dd_session", value[:-2] + ("AA" if not value.endswith("AA") else "BB"))
    assert forged.get("/api/v1/jobs").status_code == 401

    # A cookie signed with another secret is rejected too.
    other = SessionSigner("another-secret-" + "y" * 40).sign("f" * 32)
    foreign = harness.new_client()
    foreign.cookies.set("dd_session", other)
    assert foreign.get("/api/v1/jobs").status_code == 401

    signer = SessionSigner("s" * 48)
    token = signer.sign("a" * 32)
    assert signer.verify(token, 3600) is not None
    assert signer.verify(token, -1) is None  # past max-age


def test_capabilities_and_session_need_no_cookie(harness: Harness) -> None:
    anon = harness.new_client()
    assert anon.get("/api/v1/capabilities").status_code == 200
    assert anon.post("/api/v1/session").status_code == 200


def test_csrf_required_for_mutations(harness: Harness) -> None:
    client, info = harness.login()
    missing = client.post("/api/v1/jobs/" + "a" * 32 + "/cancel")
    assert missing.status_code == 403 and missing.json()["code"] == "csrf_failed"
    wrong = client.post("/api/v1/jobs/" + "a" * 32 + "/cancel", headers={"X-CSRF-Token": "not-the-token"})
    assert wrong.status_code == 403 and wrong.json()["code"] == "csrf_failed"
    # Another session's token does not work either.
    _, other_info = harness.login()
    crossed = client.post("/api/v1/jobs/" + "a" * 32 + "/cancel", headers=harness.csrf(other_info))
    assert crossed.status_code == 403
    ok = client.post("/api/v1/jobs/" + "a" * 32 + "/cancel", headers=harness.csrf(info))
    assert ok.status_code == 404 and ok.json()["code"] == "not_found"


def test_operator_login_disabled_without_password(harness: Harness) -> None:
    client = harness.new_client()
    response = client.post("/api/v1/session", json={"admin_password": "anything"})
    assert response.status_code == 403 and response.json()["code"] == "operator_login_disabled"


@pytest.fixture
def operator_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    yield from build_harness(tmp_path, monkeypatch, admin_password="correct horse battery staple")


def test_operator_login_and_admin_endpoint(operator_harness: Harness) -> None:
    h = operator_harness
    client, anon = h.login()
    denied = client.post("/api/v1/admin/cleanup", headers=h.csrf(anon))
    assert denied.status_code == 403 and denied.json()["code"] == "operator_required"

    wrong = client.post("/api/v1/session", json={"admin_password": "wrong"})
    assert wrong.status_code == 401 and wrong.json()["code"] == "invalid_credentials"
    assert "correct horse" not in wrong.text

    _, op = h.login(client, password="correct horse battery staple")
    assert op["role"] == "operator"
    # Privilege elevation rotates the session (a planted anonymous cookie never becomes an operator session).
    assert op["session_id_suffix"] != anon["session_id_suffix"] and op["csrf_token"] != anon["csrf_token"]
    result = client.post("/api/v1/admin/cleanup", headers=h.csrf(op))
    assert result.status_code == 200, result.text
    assert result.json()["retention_hours"] == h.settings.artifact_retention_hours
    # Refreshing without the password keeps the operator role.
    _, again = h.login(client)
    assert again["role"] == "operator" and again["session_id_suffix"] == op["session_id_suffix"]


def test_admin_requires_session_and_csrf(operator_harness: Harness) -> None:
    h = operator_harness
    assert h.new_client().post("/api/v1/admin/cleanup").status_code == 401
    client, op = h.login(password="correct horse battery staple")
    assert client.post("/api/v1/admin/cleanup").status_code == 403


def test_login_attempts_are_rate_limited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gen = build_harness(tmp_path, monkeypatch, admin_password="pw-123456", max_requests_per_minute=3)
    h = next(gen)
    try:
        h.services.limiter.hit = functools.partial(h.services.limiter.hit, now=1_000_020.0)  # type: ignore[method-assign]
        client = h.new_client()
        codes = [client.post("/api/v1/session", json={"admin_password": "bad"}).status_code for _ in range(5)]
        assert codes[:3] == [401, 401, 401]
        assert codes[3:] == [429, 429]
        limited = client.post("/api/v1/session", json={"admin_password": "bad"})
        assert limited.json()["code"] == "rate_limited" and int(limited.headers["retry-after"]) >= 1
    finally:
        gen.close()


def test_rate_limiter_fixed_window() -> None:
    limiter = RateLimiter(fakeredis.FakeRedis(), window_s=60)
    now = 1_000_020.0
    decisions = [limiter.hit("expensive:session", "abc", 2, now=now) for _ in range(3)]
    assert [d.allowed for d in decisions] == [True, True, False]
    assert 1 <= decisions[-1].retry_after_s <= 61
    # Next window resets the counter; other identities are independent.
    assert limiter.hit("expensive:session", "abc", 2, now=now + 60).allowed
    assert limiter.hit("expensive:session", "other", 2, now=now).allowed


def test_constant_time_helpers() -> None:
    settings = Settings(app_env="test", admin_password="secret-pass")
    assert admin_password_matches(settings, "secret-pass")
    assert not admin_password_matches(settings, "secret-pas")
    assert not admin_password_matches(Settings(app_env="test"), "")
    assert csrf_matches("token", "token") and not csrf_matches("token", None) and not csrf_matches("", "")


def test_production_requires_app_secret() -> None:
    from app.api.services import Services

    with pytest.raises(RuntimeError):
        Services.from_settings(Settings(app_env="production", app_secret=""))
