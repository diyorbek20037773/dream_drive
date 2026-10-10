"""Process liveness and dependency readiness (outside /api/v1). Model readiness is reported by /capabilities."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.api.deps import SettingsDep, get_services
from app.schemas.api import HealthCheck, HealthStatus

router = APIRouter(prefix="/health", tags=["health"])
logger = logging.getLogger("dreamdrive.api.health")


@router.get("/live", response_model=HealthStatus, summary="API process liveness")
def health_live() -> HealthStatus:
    # Liveness only proves the process serves requests; it deliberately checks no dependency.
    return HealthStatus(status="ok", checks={}, checked_utc=datetime.now(UTC))


@router.get(
    "/ready",
    response_model=HealthStatus,
    summary="Database and queue readiness",
    responses={503: {"model": HealthStatus, "description": "A required dependency (database/queue) is not ready."}},
)
def health_ready(request: Request, settings: SettingsDep) -> HealthStatus | JSONResponse:
    """200 when SQLite and Redis are reachable, 503 with the same body shape otherwise."""
    services = get_services(request)
    checks: dict[str, HealthCheck] = {}
    try:
        services.db.check()
        checks["database"] = HealthCheck(ok=True)
    except Exception:  # noqa: BLE001 - readiness reports, never raises
        logger.warning("database not ready", extra={"code": "database_unavailable"})
        checks["database"] = HealthCheck(ok=False, reason="database_unavailable")
    try:
        services.redis.ping()
        checks["queue"] = HealthCheck(ok=True)
    except Exception:  # noqa: BLE001
        logger.warning("redis not ready", extra={"code": "queue_unavailable"})
        checks["queue"] = HealthCheck(ok=False, reason="queue_unavailable")
    ok = all(c.ok for c in checks.values())
    status = HealthStatus(status="ok" if ok else "unavailable", checks=checks, checked_utc=datetime.now(UTC))
    if ok:
        return status
    return JSONResponse(status_code=503, content=status.model_dump(mode="json"))
