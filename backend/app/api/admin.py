"""Operator-only maintenance actions."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.deps import OperatorDep, SettingsDep, get_services
from app.errors import error_responses
from app.jobs.maintenance import retention_cleanup
from app.schemas.api import CleanupResult

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post(
    "/cleanup",
    response_model=CleanupResult,
    summary="Run retention cleanup now (operator role)",
    responses=error_responses(401, 403, 422),
)
def run_cleanup(request: Request, session: OperatorDep, settings: SettingsDep) -> CleanupResult:
    """Removes expired artifact bytes and their database references."""
    services = get_services(request)
    stats = retention_cleanup(settings, services.db, services.redis)
    return CleanupResult(
        removed_jobs=stats.removed_jobs,
        removed_runs=stats.removed_runs,
        removed_evaluations=stats.removed_evaluations,
        removed_bytes=stats.removed_bytes,
        retention_hours=settings.artifact_retention_hours,
        started_utc=stats.started_utc,
        finished_utc=stats.finished_utc,
    )
