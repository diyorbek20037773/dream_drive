"""FastAPI application factory.

Creating the app opens no Redis/SQLite connection, so the OpenAPI schema can be exported offline
(`python -m app.export_openapi`). Connections are created lazily by the dependencies that need them; the lifespan
startup applies the SQLite migrations when the server actually starts.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.routing import APIRoute
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api import API_PREFIX, api_router, health_router
from app.api.middleware import BodyLimitMiddleware, HttpMetricsMiddleware
from app.api.services import Services
from app.api.videos import UPLOAD_FORM_COMPONENT, UPLOAD_FORM_NAME, UPLOAD_PATH
from app.config import Settings, get_settings
from app.errors import install_error_handling
from app.logging_setup import configure_logging

logger = logging.getLogger("dreamdrive.api")

API_TITLE = "DreamDrive API"
# Version of the HTTP contract (not the build); bump when the OpenAPI changes incompatibly.
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "DreamDrive simulation prototype API: uploaded-video perception analysis (observation only, no actuation) and "
    "simulator-only closed-loop driving. Every perception and command is tagged with its source and mode. "
    "Errors use the ErrorEnvelope schema."
)

OPENAPI_TAGS = [
    {"name": "health", "description": "Liveness and dependency readiness."},
    {"name": "session", "description": "Anonymous signed sessions and operator login."},
    {"name": "capabilities", "description": "Model, device, classes, simulator modes, queue and limits."},
    {"name": "videos", "description": "Video upload into the analysis queue."},
    {"name": "jobs", "description": "Video analysis jobs, results and artifacts."},
    {"name": "simulators", "description": "Simulator adapters, perception modes, scenarios and road graphs."},
    {"name": "runs", "description": "Simulator-only closed-loop runs."},
    {"name": "evaluations", "description": "Evaluation artifacts with real sample counts."},
    {"name": "events", "description": "Server-Sent Events."},
    {"name": "admin", "description": "Operator-only maintenance."},
]


def _operation_id(route: APIRoute) -> str:
    # Stable, readable operation ids (the handler function name) for generated client types.
    return route.name


def _metrics() -> Response:
    """Prometheus exposition for this API process. Internal only: the public gateway answers 404."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


_ERROR_ENVELOPE_REF = "#/components/schemas/ErrorEnvelope"


def _normalize_openapi(document: dict[str, Any]) -> dict[str, Any]:
    """Fix FastAPI's schema for routes whose response class is not JSON (the SSE stream).

    - FastAPI emits ``{"type": "string", "$ref": ...}`` for a model on a streaming class; ``type: string`` contradicts
      the referenced object schema, so only the ``$ref`` (the EventEnvelope in each ``data:`` line) is kept.
    - Documented error responses inherit the route media type; errors are always ``application/json`` ErrorEnvelope.
    - The upload handler streams the multipart body itself (byte cap, no buffering before the session/CSRF/quota
      checks), so its form schema is declared explicitly and registered here as a component.
    """
    for operations in document.get("paths", {}).values():
        for operation in operations.values():
            for response in operation.get("responses", {}).values():
                content = response.get("content", {})
                for media in content.values():
                    schema = media.get("schema")
                    if isinstance(schema, dict) and "$ref" in schema and schema.get("type") == "string":
                        del schema["type"]
                error_media = [m for m in content.values() if m.get("schema", {}).get("$ref") == _ERROR_ENVELOPE_REF]
                if error_media:
                    response["content"] = {"application/json": {"schema": {"$ref": _ERROR_ENVELOPE_REF}}}
    if f"{API_PREFIX}{UPLOAD_PATH}" in document.get("paths", {}):
        document.setdefault("components", {}).setdefault("schemas", {})[UPLOAD_FORM_NAME] = dict(UPLOAD_FORM_COMPONENT)
    return document


def _install_openapi(app: FastAPI) -> None:
    generate = app.openapi

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            app.openapi_schema = _normalize_openapi(generate())
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]


def _lifespan(settings: Settings) -> Any:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        services: Services = app.state.services
        for path in (settings.artifact_root, settings.tmp_root, settings.evaluation_root):
            path.mkdir(parents=True, exist_ok=True)
        services.db.migrate()  # fail fast: an API that cannot open or migrate its database must not start
        logger.info("api started", extra={"code": "api_started"})
        if not settings.app_secret:
            logger.warning("APP_SECRET is empty: using an ephemeral secret; sessions end when the API restarts")
        yield

    return lifespan


def create_app(settings: Settings | None = None, *, services: Services | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging("api", settings.log_level)
    expose_docs = not settings.is_production
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        openapi_tags=OPENAPI_TAGS,
        openapi_url=f"{API_PREFIX}/openapi.json" if expose_docs else None,
        docs_url=f"{API_PREFIX}/docs" if expose_docs else None,
        redoc_url=None,
        generate_unique_id_function=_operation_id,
        lifespan=_lifespan(settings),
    )
    app.state.services = services or Services.from_settings(settings)
    # Handlers declare `settings: SettingsDep`; they must see this app's settings, not the process-global ones.
    app.dependency_overrides[get_settings] = lambda: settings
    # Middleware added last runs first: request id (outermost) -> metrics/access log -> body limit -> routes.
    app.add_middleware(BodyLimitMiddleware, exempt=(f"{API_PREFIX}{UPLOAD_PATH}",))
    app.add_middleware(HttpMetricsMiddleware)
    install_error_handling(app)
    app.include_router(health_router)
    app.include_router(api_router)
    app.add_api_route("/metrics", _metrics, methods=["GET"], include_in_schema=False, name="metrics")
    _install_openapi(app)
    return app


app = create_app()
