"""Anonymous signed session or configured operator login (docs/IMPLEMENTATION_SPEC.md §3.3)."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Body, Request, Response

from app.api.deps import SettingsDep, get_services, load_session, rate_limit_or_raise
from app.db import utc_now
from app.errors import ApiError, error_responses
from app.schemas.api import ErrorCode, Limits, SessionCreate, SessionInfo
from app.security import (
    NEW_SESSIONS_PER_IP_PER_MINUTE,
    SESSION_COOKIE,
    admin_password_matches,
    client_ip,
    new_csrf_token,
    new_session_id,
    session_ttl_s,
    set_session_cookies,
)
from app.storage import repo

router = APIRouter(tags=["session"])
logger = logging.getLogger("dreamdrive.api.session")


@router.post(
    "/session",
    response_model=SessionInfo,
    summary="Create or refresh the session",
    responses=error_responses(401, 403, 422, 429),
)
def create_session(
    request: Request,
    response: Response,
    settings: SettingsDep,
    body: Annotated[SessionCreate | None, Body()] = None,
) -> SessionInfo:
    """Sets the HttpOnly `dd_session` cookie and the JS-readable `dd_csrf` cookie.

    With `admin_password`: 401 `invalid_credentials` on mismatch, 403 `operator_login_disabled` when no
    operator password is configured. No CSRF header is required for this endpoint.
    """
    services = get_services(request)
    ip = client_ip(request)
    existing = load_session(services, request.cookies.get(SESSION_COOKIE))
    role = existing.role if existing else "anonymous"

    login = body is not None and body.admin_password is not None
    if login:
        if not settings.admin_password:
            raise ApiError(403, ErrorCode.OPERATOR_LOGIN_DISABLED, "Operator login is disabled on this instance")
        # Brute-force guard: login attempts share the expensive-request budget per client IP.
        rate_limit_or_raise(services, "login:ip", ip, settings.max_requests_per_minute)
        assert body is not None and body.admin_password is not None
        if not admin_password_matches(settings, body.admin_password):
            logger.warning("operator login rejected", extra={"code": "invalid_credentials"})
            raise ApiError(401, ErrorCode.INVALID_CREDENTIALS, "Invalid operator password")

    if existing is not None and not (login and existing.role != "operator"):
        session_id, csrf_token = existing.session_id, existing.csrf_token
        repo.touch_session(services.db, session_id, role)
    else:
        # New session; also on privilege elevation, so a planted pre-login cookie never becomes an operator session.
        if existing is None:
            rate_limit_or_raise(services, "session:ip", ip, NEW_SESSIONS_PER_IP_PER_MINUTE)
        role = "operator" if login else "anonymous"
        session_id, csrf_token = new_session_id(), new_csrf_token()
        repo.create_session(services.db, session_id, role, csrf_token)
        if login:
            logger.info("operator session created", extra={"code": "operator_login"})

    set_session_cookies(response, settings, services.signer.sign(session_id), csrf_token)
    response.headers["Cache-Control"] = "no-store"
    return SessionInfo(
        session_id_suffix=session_id[-6:],
        role=role,
        csrf_token=csrf_token,
        expires_utc=utc_now() + timedelta(seconds=session_ttl_s(settings)),
        limits=Limits.from_settings(settings),
    )
