"""API error type, request-id propagation and handlers that always answer with the ErrorEnvelope.

Responses never contain stack traces, exception strings or filesystem paths. Unexpected exceptions are logged
server-side with the request id and returned as a generic ``internal_error``.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.logging_setup import request_id_var
from app.schemas.api import ErrorCode, ErrorEnvelope

logger = logging.getLogger("dreamdrive.api.errors")

REQUEST_ID_HEADER = "X-Request-ID"
# Accept a caller/gateway supplied id only when it is short and harmless to echo into logs and headers.
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,64}$")

# Generic codes for errors raised by the framework (unknown route, wrong method, ...).
_STATUS_CODES: dict[int, ErrorCode] = {
    400: ErrorCode.HTTP_ERROR,
    404: ErrorCode.NOT_FOUND,
    405: ErrorCode.METHOD_NOT_ALLOWED,
    413: ErrorCode.UPLOAD_TOO_LARGE,
    415: ErrorCode.UNSUPPORTED_MEDIA_TYPE,
    422: ErrorCode.VALIDATION_ERROR,
    429: ErrorCode.RATE_LIMITED,
    501: ErrorCode.NOT_IMPLEMENTED,
    503: ErrorCode.SERVICE_UNAVAILABLE,
}
_STATUS_MESSAGES: dict[int, str] = {
    404: "Resource not found",
    405: "Method not allowed",
}


class ApiError(Exception):
    """Raise from routes/dependencies to return ``ErrorEnvelope`` with the given HTTP status."""

    def __init__(
        self,
        status: int,
        code: ErrorCode | str,
        message: str,
        details: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = str(code)
        self.message = message
        self.details: dict[str, Any] = dict(details or {})
        self.headers: dict[str, str] = dict(headers or {})


def not_implemented(feature: str) -> ApiError:
    """Placeholder error for routes whose behavior is not wired yet (visible 501, never a fake success)."""
    return ApiError(501, ErrorCode.NOT_IMPLEMENTED, f"{feature} is not implemented yet")


def new_request_id() -> str:
    return uuid.uuid4().hex


def get_request_id(request: Request) -> str:
    """Request id assigned by RequestIdMiddleware (or a fresh one if the middleware did not run)."""
    rid = getattr(request.state, "request_id", None)
    if isinstance(rid, str) and rid:
        return rid
    rid = new_request_id()
    request.state.request_id = rid
    return rid


class RequestIdMiddleware:
    """Pure ASGI middleware (safe for streaming/SSE): assigns ``scope.state.request_id`` and echoes the header."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = ""
        for name, value in scope.get("headers", []):
            if name == b"x-request-id":
                incoming = value.decode("latin-1")
                break
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else new_request_id()
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_var.set(request_id)  # log records of this request carry its id

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"]
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id_var.reset(token)


def error_response(
    request: Request,
    status: int,
    code: ErrorCode | str,
    message: str,
    details: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    request_id = get_request_id(request)
    body = {"code": str(code), "message": message, "request_id": request_id, "details": dict(details or {})}
    out_headers = {**(headers or {}), REQUEST_ID_HEADER: request_id}
    return JSONResponse(status_code=status, content=body, headers=out_headers)


def _validation_details(exc: RequestValidationError) -> dict[str, Any]:
    # Only location, message and type: echoing `input`/`ctx` could reflect large or sensitive payloads.
    errors = [
        {"loc": [str(p) for p in err.get("loc", ())], "msg": str(err.get("msg", "")), "type": str(err.get("type", ""))}
        for err in exc.errors()
    ]
    return {"errors": errors}


async def _handle_api_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return error_response(request, exc.status, exc.code, exc.message, exc.details, exc.headers)


async def _handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    return error_response(
        request, 422, ErrorCode.VALIDATION_ERROR, "Request validation failed", _validation_details(exc)
    )


async def _handle_http_exception(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    status = exc.status_code
    code = _STATUS_CODES.get(status, ErrorCode.HTTP_ERROR)
    # Framework HTTPException details are short fixed strings; still prefer our own wording for common statuses.
    message = _STATUS_MESSAGES.get(status) or (exc.detail if isinstance(exc.detail, str) else "HTTP error")
    headers = dict(exc.headers or {})
    return error_response(request, status, code, message, None, headers)


async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    request_id = get_request_id(request)
    logger.error(
        "unhandled exception",
        exc_info=(type(exc), exc, exc.__traceback__),
        extra={"request_id": request_id, "code": ErrorCode.INTERNAL_ERROR.value},
    )
    return error_response(request, 500, ErrorCode.INTERNAL_ERROR, "Internal server error")


def install_error_handling(app: FastAPI) -> None:
    """Register the request-id middleware and every error handler on ``app``."""
    app.add_middleware(RequestIdMiddleware)
    app.add_exception_handler(ApiError, _handle_api_error)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(Exception, _handle_unexpected)


def error_responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI ``responses=`` entries documenting ErrorEnvelope bodies for the given statuses."""
    descriptions = {
        400: "Bad request",
        401: "Session required (`session_required`)",
        403: "CSRF token invalid (`csrf_failed`) or operator role required (`operator_required`)",
        404: "Not found or not owned by this session (`not_found`)",
        409: "Conflicting resource state",
        413: "Upload exceeds the configured size limit (`upload_too_large`)",
        415: "Unsupported media type (`unsupported_media_type`)",
        416: "Requested byte range not satisfiable (`range_not_satisfiable`)",
        422: "Validation failed or unsupported mode/scenario (`validation_error`, `unsupported_*`, ...)",
        429: "Rate limit or queue admission denied (`rate_limited`, `*_queue_full`, `*_busy`); see Retry-After",
        503: "Required service or model not ready (`model_not_ready`, `service_unavailable`, ...)",
    }
    return {s: {"model": ErrorEnvelope, "description": descriptions.get(s, "Error")} for s in statuses}
