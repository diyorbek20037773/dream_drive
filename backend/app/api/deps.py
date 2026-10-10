"""Request dependencies shared by the routers (docs/IMPLEMENTATION_SPEC.md §3.3).

The signatures here are part of the API contract: they declare the `dd_session` cookie security scheme and the
`X-CSRF-Token` header in the OpenAPI. Verification lives in app/security.py; this module adapts it to FastAPI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Annotated

import redis
from fastapi import Depends, Header, Request, Security
from fastapi.security import APIKeyCookie

from app.api.services import Services
from app.config import Settings, get_settings
from app.db import utc_now
from app.errors import ApiError
from app.schemas.api import ErrorCode, SessionRole
from app.security import client_ip, csrf_matches, session_ttl_s
from app.storage import repo

logger = logging.getLogger("dreamdrive.api.deps")

SESSION_COOKIE_NAME = "dd_session"
CSRF_COOKIE_NAME = "dd_csrf"
CSRF_HEADER_NAME = "X-CSRF-Token"
# last_seen_utc is refreshed at most this often per session (avoids a DB write on every request).
_TOUCH_INTERVAL = timedelta(minutes=5)

session_cookie_scheme = APIKeyCookie(
    name=SESSION_COOKIE_NAME,
    scheme_name="SessionCookie",
    description="Signed, HttpOnly session cookie set by POST /api/v1/session.",
    auto_error=False,
)


@dataclass(frozen=True, slots=True)
class SessionContext:
    """Verified session of the caller. ``session_id`` is the owner id for jobs/runs/evaluations."""

    session_id: str
    role: SessionRole
    csrf_token: str
    expires_utc: datetime

    @property
    def is_operator(self) -> bool:
        return self.role == "operator"


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


ServicesDep = Annotated[Services, Depends(get_services)]


def _session_required() -> ApiError:
    return ApiError(401, ErrorCode.SESSION_REQUIRED, "A valid session is required; call POST /api/v1/session first")


def load_session(services: Services, cookie_value: str | None) -> SessionContext | None:
    """Verify the signed cookie and load the session row; None when missing, forged, expired or deleted."""
    if not cookie_value:
        return None
    ttl = session_ttl_s(services.settings)
    verified = services.signer.verify(cookie_value, ttl)
    if verified is None:
        return None
    session_id, issued = verified
    row = repo.get_session(services.db, session_id)
    if row is None:
        return None
    if utc_now() - row.last_seen_utc > _TOUCH_INTERVAL:
        repo.touch_session(services.db, session_id)
    return SessionContext(
        session_id=row.id,
        role=row.role,  # type: ignore[arg-type]
        csrf_token=row.csrf_token,
        expires_utc=issued + timedelta(seconds=ttl),
    )


def get_session(
    request: Request,
    session_cookie: Annotated[str | None, Security(session_cookie_scheme)] = None,
) -> SessionContext:
    """Verify the signed `dd_session` cookie and load the session; 401 `session_required` if missing/invalid."""
    session = load_session(get_services(request), session_cookie)
    if session is None:
        raise _session_required()
    request.state.session_id = session.session_id
    return session


def require_csrf(
    session: Annotated[SessionContext, Depends(get_session)],
    x_csrf_token: Annotated[
        str | None,
        Header(alias=CSRF_HEADER_NAME, max_length=256, description="CSRF token returned by POST /api/v1/session."),
    ] = None,
) -> SessionContext:
    """Session plus double-submit CSRF check for cookie-authenticated mutations; 403 `csrf_failed` on mismatch."""
    if not csrf_matches(session.csrf_token, x_csrf_token):
        raise ApiError(403, ErrorCode.CSRF_FAILED, "Missing or invalid X-CSRF-Token header")
    return session


def require_operator(session: Annotated[SessionContext, Depends(require_csrf)]) -> SessionContext:
    """Operator role for admin/destructive actions (after session and CSRF checks)."""
    if not session.is_operator:
        raise ApiError(403, ErrorCode.OPERATOR_REQUIRED, "Operator role required")
    return session


def rate_limit_or_raise(services: Services, scope: str, ident: str, limit: int) -> None:
    try:
        decision = services.limiter.hit(scope, ident, limit)
    except redis.RedisError as exc:
        # Expensive requests need Redis anyway (queueing); fail closed rather than run without limits.
        raise ApiError(503, ErrorCode.SERVICE_UNAVAILABLE, "Queue service unavailable") from exc
    if not decision.allowed:
        raise ApiError(
            429,
            ErrorCode.RATE_LIMITED,
            "Too many requests; try again later",
            {"limit_per_minute": limit, "retry_after_s": decision.retry_after_s},
            headers={"Retry-After": str(decision.retry_after_s)},
        )


def enforce_expensive_rate_limit(
    request: Request,
    session: Annotated[SessionContext, Depends(require_csrf)],
) -> None:
    """Fixed-window limit for expensive requests per session and per client IP; 429 `rate_limited`."""
    services = get_services(request)
    limit = services.settings.max_requests_per_minute
    rate_limit_or_raise(services, "expensive:session", session.session_id, limit)
    rate_limit_or_raise(services, "expensive:ip", client_ip(request), limit)


SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[SessionContext, Depends(get_session)]
CsrfSessionDep = Annotated[SessionContext, Depends(require_csrf)]
OperatorDep = Annotated[SessionContext, Depends(require_operator)]
ExpensiveRateLimit = Depends(enforce_expensive_rate_limit)
