"""Model/device/classes and simulator modes with unavailable reasons. Public: no session required."""

from __future__ import annotations

import logging
import sqlite3

from fastapi import APIRouter, Request

from app.api import catalog
from app.api.deps import SettingsDep, get_services
from app.errors import error_responses
from app.schemas.api import (
    MAX_ACTIVE_SIMULATIONS,
    BuildInfo,
    Capabilities,
    Limits,
    MonitoringInfo,
    QueueStatus,
    SimulationCapability,
    VideoAnalysisCapability,
)
from app.storage import repo

router = APIRouter(tags=["capabilities"])
logger = logging.getLogger("dreamdrive.api.capabilities")


@router.get(
    "/capabilities",
    response_model=Capabilities,
    summary="System capabilities",
    responses=error_responses(503),
)
def get_capabilities(request: Request, settings: SettingsDep) -> Capabilities:
    """Unavailable features are reported with `available=false` and a reason, never hidden or faked."""
    services = get_services(request)
    model = services.model_status.get()
    info = model.info
    ready = model.ready
    video = VideoAnalysisCapability(
        available=ready,
        reason=None if ready else (model.reason or "model_not_ready"),
        model=info,
        device=info.device if info else None,
        precision=info.precision if info else None,
        # Only the classes of the verified checkpoint's class map; empty while no verified model is ready.
        supported_classes=sorted(set(info.class_map.values())) if info is not None and ready else [],
    )
    try:
        queue = QueueStatus(
            video_queued=repo.count_jobs(services.db, ("queued",)),
            video_running=repo.count_jobs(services.db, repo.JOB_IN_PROGRESS_STATES),
            capacity_total=settings.max_queued_total,
            per_session=settings.max_queued_per_session,
            sim_active=repo.count_runs(services.db, repo.RUN_ACTIVE_STATES),
            sim_capacity=MAX_ACTIVE_SIMULATIONS,
        )
    except sqlite3.Error:
        logger.warning("queue status unavailable", extra={"code": "database_unavailable"})
        queue = QueueStatus(
            capacity_total=settings.max_queued_total,
            per_session=settings.max_queued_per_session,
            sim_capacity=MAX_ACTIVE_SIMULATIONS,
            reason="database_unavailable",
        )
    return Capabilities(
        build=BuildInfo(
            commit=settings.git_commit if settings.commit_is_valid else None,
            version=settings.build_version,
            dirty=settings.build_dirty,
        ),
        video_analysis=video,
        simulation=SimulationCapability(simulators=catalog.simulators(settings, model)),
        queue=queue,
        limits=Limits.from_settings(settings),
        monitoring=MonitoringInfo(metrics=True),
    )
