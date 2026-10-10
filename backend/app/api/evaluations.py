"""Versioned evaluation artifacts with real sample counts; synthetic scenario suites can be started here.

Finished evaluations are artifacts under DATA_ROOT/evaluations read through ``app.evaluation.registry`` (system
results without personal data, visible to every session). Requests made through this API are additionally tracked
in SQLite so their owner sees them while queued/running or when they failed.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

import redis
from fastapi import APIRouter, Request
from pydantic import ValidationError

from app.api.deps import CsrfSessionDep, ExpensiveRateLimit, SessionDep, SettingsDep, get_services
from app.api.params import EvaluationIdPath
from app.config import Settings
from app.db import utc_now
from app.errors import ApiError, error_responses
from app.evaluation.datasets import LITEROAD_SYNTHETIC_SCENARIOS
from app.jobs.queue import enqueue_eval_job
from app.schemas.api import ErrorCode, Evaluation, EvaluationCreate, EvaluationDataset, EvaluationList
from app.storage import artifacts, repo

router = APIRouter(prefix="/evaluations", tags=["evaluations"])
logger = logging.getLogger("dreamdrive.api.evaluations")

SUITE_SOURCE = "literoad_synthetic_rgb_cv"


def _registry() -> Any:
    try:
        from app.evaluation import registry
    except ImportError as exc:
        raise ApiError(
            503, ErrorCode.EVALUATION_UNAVAILABLE, "Evaluation registry unavailable", {"reason": "module_missing"}
        ) from exc
    return registry


def _from_artifact(item: Any) -> Evaluation | None:
    if not isinstance(item, dict):
        return None
    try:
        return Evaluation.model_validate({"state": "succeeded", **item})
    except ValidationError:
        logger.warning("evaluation artifact does not match the Evaluation schema", extra={"code": "evaluation_invalid"})
        return None


def _from_row(row: repo.EvaluationRow, settings: Settings) -> Evaluation:
    if row.state == "succeeded" and row.summary:
        parsed = _from_artifact({**row.summary, "id": row.id, "state": "succeeded"})
        if parsed is not None:
            return parsed
    pending = row.state in repo.EVAL_ACTIVE_STATES
    return Evaluation(
        id=row.id,
        kind=row.kind,  # type: ignore[arg-type]
        state=row.state,  # type: ignore[arg-type]
        created_utc=row.created_utc,
        finished_utc=row.finished_utc,
        source=SUITE_SOURCE,
        sample_count=0,
        metrics={},
        limitations=[
            "No results yet: the suite is queued or running." if pending else "The suite did not produce results.",
            "Synthetic LiteRoad scenario validation; not a learned-model accuracy measurement.",
        ],
        code_commit=settings.git_commit if settings.commit_is_valid else None,
        model_sha256=None,
        seeds=row.seeds,
        error_code=row.error_code,
        dataset=EvaluationDataset.model_validate(LITEROAD_SYNTHETIC_SCENARIOS),
    )


@router.get("", response_model=EvaluationList, summary="List evaluation runs", responses=error_responses(401))
def list_evaluations(request: Request, session: SessionDep, settings: SettingsDep) -> EvaluationList:
    """Only real evaluation artifacts; an empty list is returned (and explained by the UI) when none exist."""
    services = get_services(request)
    registry = _registry()
    items: list[Evaluation] = []
    for raw in registry.list_evaluations(settings.evaluation_root):
        parsed = _from_artifact(raw)
        if parsed is not None:
            items.append(parsed)
    known = {e.id for e in items}
    for row in repo.list_owner_evaluations(services.db, session.session_id):
        if row.id in known or (row.artifact_id and row.artifact_id in known):
            continue
        items.append(_from_row(row, settings))
    items.sort(key=lambda e: e.created_utc, reverse=True)
    return EvaluationList(items=items)


@router.post(
    "",
    status_code=202,
    response_model=Evaluation,
    summary="Start a synthetic scenario suite",
    dependencies=[ExpensiveRateLimit],
    responses=error_responses(401, 403, 422, 429, 503),
)
def create_evaluation(
    body: EvaluationCreate, request: Request, session: CsrfSessionDep, settings: SettingsDep
) -> Evaluation:
    """202 with the queued evaluation. 429 `evaluation_busy`/`simulation_busy`/`rate_limited`."""
    services = get_services(request)
    if importlib.util.find_spec("app.evaluation.scenario_suite") is None:
        raise ApiError(
            503, ErrorCode.EVALUATION_UNAVAILABLE, "The scenario suite is not available", {"reason": "module_missing"}
        )
    if repo.count_runs(services.db, repo.RUN_ACTIVE_STATES) > 0:
        # The suite shares the CPU with the interactive run; keep the live run responsive.
        raise ApiError(
            429,
            ErrorCode.SIMULATION_BUSY,
            "A simulation is active; try again when it ends",
            headers={"Retry-After": "30"},
        )
    eval_id = artifacts.new_id()
    try:
        row = repo.insert_evaluation(
            services.db, eval_id=eval_id, owner_id=session.session_id, kind=body.kind, seeds=list(body.seeds)
        )
    except repo.AdmissionDenied as exc:
        raise ApiError(429, exc.code, exc.message, exc.details, headers={"Retry-After": "60"}) from exc
    try:
        enqueue_eval_job(services.redis, eval_id, len(body.seeds))
    except redis.RedisError as exc:
        repo.transition_evaluation(
            services.db, eval_id, ("queued",), "failed", error_code="queue_unavailable", finished_utc=utc_now()
        )
        raise ApiError(503, ErrorCode.SERVICE_UNAVAILABLE, "The job queue is unavailable") from exc
    logger.info("evaluation queued", extra={"eval_id": eval_id, "code": "evaluation_queued"})
    return _from_row(row, settings)


@router.get(
    "/{evaluation_id}",
    response_model=Evaluation,
    summary="Evaluation details and metrics",
    responses=error_responses(401, 404, 422),
)
def get_evaluation(
    request: Request, evaluation_id: EvaluationIdPath, session: SessionDep, settings: SettingsDep
) -> Evaluation:
    services = get_services(request)
    registry = _registry()
    found = _from_artifact(registry.load_evaluation(settings.evaluation_root, evaluation_id))
    if found is not None:
        return found
    row = repo.get_evaluation(services.db, session.session_id, evaluation_id)
    if row is None:
        raise ApiError(404, ErrorCode.NOT_FOUND, "Evaluation not found")
    if row.state == "succeeded" and row.artifact_id:
        artifact = _from_artifact(registry.load_evaluation(settings.evaluation_root, row.artifact_id))
        if artifact is not None:
            return artifact
    return _from_row(row, settings)
