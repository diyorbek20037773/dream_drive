"""Pure ASGI middleware (safe for streaming/SSE): HTTP metrics + JSON access log, and request body limits."""

from __future__ import annotations

import json
import logging
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app import metrics

logger = logging.getLogger("dreamdrive.api.access")

_KNOWN_METHODS = {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}
# Long-lived responses whose duration says nothing about request handling cost.
_STREAMING_ROUTES = {"/api/v1/events"}
_QUIET_PREFIXES = ("/health/", "/metrics")
_API_PREFIX = "/api/v1"  # same as app.api.API_PREFIX (not imported to keep this module dependency-free)


def _route_template(scope: Scope) -> str:
    """Matched route template (bounded label). FastAPI applies an include prefix lazily, so the matched route's
    own path lacks the /api/v1 prefix of the router it was included into; restore it from the request path."""
    route = scope.get("route")
    path = getattr(route, "path", None)
    if not isinstance(path, str):
        return "unmatched"
    raw = str(scope.get("path", ""))
    if raw.startswith(_API_PREFIX + "/") and not path.startswith(_API_PREFIX + "/"):
        return _API_PREFIX + path
    return path


class HttpMetricsMiddleware:
    """Counts requests by route *template* (bounded labels) and writes one JSON access line without the raw URL."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - start
            method = scope.get("method", "GET")
            method = method if method in _KNOWN_METHODS else "OTHER"
            template = _route_template(scope)
            metrics.HTTP_REQUESTS.labels(template, method, metrics.status_class(status)).inc()
            if template not in _STREAMING_ROUTES:
                metrics.HTTP_DURATION.labels(template, method).observe(elapsed)
            level = logging.DEBUG if str(scope.get("path", "")).startswith(_QUIET_PREFIXES) else logging.INFO
            logger.log(
                level,
                "request",
                extra={
                    "method": method,
                    "route": template,
                    "status": status,
                    "duration_ms": round(elapsed * 1000.0, 2),
                    "request_id": scope.get("state", {}).get("request_id"),
                },
            )


class BodyLimitMiddleware:
    """Rejects request bodies above ``limit`` bytes (413 ``upload_too_large``) except on exempt paths.

    The video upload streams with its own MAX_UPLOAD_BYTES cap; every other endpoint takes small JSON bodies, so a
    huge body is refused before FastAPI buffers it in memory.
    """

    def __init__(self, app: ASGIApp, limit: int = 64 * 1024, exempt: tuple[str, ...] = ()) -> None:
        self.app = app
        self.limit = limit
        self.exempt = set(exempt)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in self.exempt:
            await self.app(scope, receive, send)
            return
        declared = None
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = None
                break
        if declared is not None and declared > self.limit:
            await self._reject(scope, send)
            return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    # Chunked body grew too large: end the request as if the client went away.
                    return {"type": "http.disconnect"}
            return message

        await self.app(scope, limited_receive, send)

    async def _reject(self, scope: Scope, send: Send) -> None:
        request_id = scope.get("state", {}).get("request_id", "")
        body = json.dumps(
            {
                "code": "upload_too_large",
                "message": "Request body too large",
                "request_id": request_id,
                "details": {"limit_bytes": self.limit},
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body})
