"""Session cookies, CSRF, operator login and rate limiting (docs/IMPLEMENTATION_SPEC.md section 3.3).

- ``dd_session``: the session id signed with ``APP_SECRET`` (itsdangerous, timestamped), HttpOnly, SameSite=Lax,
  Secure in production, max-age = SESSION_TTL_HOURS. The id is the owner id of jobs/runs; it is never readable by JS.
- ``dd_csrf``: a random per-session token, readable by JS. Every POST/DELETE except ``POST /session`` must send it
  back as ``X-CSRF-Token``; it is compared in constant time with the token stored for the session (a forged
  cross-site form cannot read or set that header).
- Operator login compares SHA-256 digests with ``hmac.compare_digest`` (constant time, length independent) and is
  disabled when ``ADMIN_PASSWORD`` is empty.
- Expensive requests are limited per session and per client IP with Redis fixed one-minute windows.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import redis
from fastapi import Request, Response
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app.config import Settings

SESSION_COOKIE = "dd_session"
CSRF_COOKIE = "dd_csrf"
_SESSION_SALT = "dreamdrive.session.v1"
RATE_WINDOW_S = 60
# New anonymous sessions per client IP and minute (refreshing an existing session does not count). Generous enough
# for a classroom behind one NAT, small enough to stop trivially flooding the sessions table.
NEW_SESSIONS_PER_IP_PER_MINUTE = 120


def new_session_id() -> str:
    return uuid.uuid4().hex


def new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def session_ttl_s(settings: Settings) -> int:
    return int(settings.session_ttl_hours * 3600)


class SessionSigner:
    def __init__(self, secret: str) -> None:
        if not secret:
            raise ValueError("session secret must not be empty")
        self._serializer = URLSafeTimedSerializer(secret, salt=_SESSION_SALT)

    def sign(self, session_id: str) -> str:
        return self._serializer.dumps(session_id)

    def verify(self, token: str, max_age_s: int) -> tuple[str, datetime] | None:
        """Return (session_id, issued_utc) for a valid, unexpired token; None otherwise."""
        if not token or len(token) > 512:
            return None
        try:
            value, issued = self._serializer.loads(token, max_age=max_age_s, return_timestamp=True)
        except (SignatureExpired, BadSignature):
            return None
        if not isinstance(value, str) or len(value) != 32:
            return None
        return value, issued.astimezone(UTC)


def set_session_cookies(response: Response, settings: Settings, signed_session: str, csrf_token: str) -> None:
    max_age = session_ttl_s(settings)
    secure = settings.is_production
    response.set_cookie(
        SESSION_COOKIE, signed_session, max_age=max_age, path="/", secure=secure, httponly=True, samesite="lax"
    )
    # Readable by the SPA (double-submit); it is not an authentication credential on its own.
    response.set_cookie(
        CSRF_COOKIE, csrf_token, max_age=max_age, path="/", secure=secure, httponly=False, samesite="lax"
    )


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def admin_password_matches(settings: Settings, supplied: str) -> bool:
    if not settings.admin_password:
        return False
    return hmac.compare_digest(_digest(supplied), _digest(settings.admin_password))


def csrf_matches(expected: str, supplied: str | None) -> bool:
    if not supplied or not expected:
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def client_ip(request: Request) -> str:
    """Client address for rate limiting.

    The API is reachable only through the gateway on the internal network. Caddy replaces X-Forwarded-For from
    untrusted clients with the address it saw, so the right-most entry is the gateway-observed client address.
    Without the header (direct internal calls, tests) the socket peer is used.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        candidate = forwarded.split(",")[-1].strip()
        if candidate and len(candidate) <= 64:
            return candidate
    return request.client.host if request.client else "unknown"


@dataclass(frozen=True, slots=True)
class RateDecision:
    allowed: bool
    retry_after_s: int
    count: int
    limit: int


class RateLimiter:
    """Fixed-window counter in Redis: ``INCR dd:rl:<scope>:<hash(ident)>:<window>`` with a TTL."""

    def __init__(self, client: redis.Redis, window_s: int = RATE_WINDOW_S) -> None:
        self.redis = client
        self.window_s = window_s

    def hit(self, scope: str, ident: str, limit: int, now: float | None = None) -> RateDecision:
        now = time.time() if now is None else now
        window = int(now // self.window_s)
        ident_hash = hashlib.sha256(ident.encode("utf-8")).hexdigest()[:24]
        key = f"dd:rl:{scope}:{ident_hash}:{window}"
        pipe = self.redis.pipeline(transaction=True)
        pipe.incr(key)
        pipe.expire(key, self.window_s + 5)
        count = int(pipe.execute()[0])
        retry_after = max(1, int((window + 1) * self.window_s - now) + 1)
        return RateDecision(allowed=count <= limit, retry_after_s=retry_after, count=count, limit=limit)
