"""Database rows -> API response models."""

from __future__ import annotations

import base64
import binascii
import json
import logging
from typing import Any

import redis
from pydantic import ValidationError

from app.config import Settings
from app.db import to_db_time
from app.errors import ApiError
from app.jobs.queue import run_live_key
from app.schemas.api import ErrorCode, Job, JobInput, JobVideo, Run, RunLive, RunRoute
from app.schemas.inference import ModelIdentity
from app.storage.repo import JobRow, RunRow

logger = logging.getLogger("dreamdrive.api.convert")


def encode_cursor(created_utc: Any, item_id: str) -> str:
    raw = f"{to_db_time(created_utc)}|{item_id}".encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str | None) -> tuple[str, str] | None:
    if cursor is None:
        return None
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        created, _, item_id = base64.urlsafe_b64decode(padded.encode()).decode().partition("|")
    except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
        raise ApiError(422, ErrorCode.VALIDATION_ERROR, "Invalid cursor", {"field": "cursor"}) from exc
    if not created or not item_id or len(item_id) > 64:
        raise ApiError(422, ErrorCode.VALIDATION_ERROR, "Invalid cursor", {"field": "cursor"})
    return created, item_id


def job_to_api(row: JobRow, settings: Settings) -> Job:
    video = None
    if row.probe:
        try:
            video = JobVideo.model_validate(row.probe)
        except ValidationError:
            logger.warning("stored probe does not match JobVideo", extra={"job_id": row.id})
    model = None
    if row.model:
        try:
            model = ModelIdentity.model_validate(row.model)
        except ValidationError:
            logger.warning("stored model identity invalid", extra={"job_id": row.id})
    fps = row.settings.get("analysis_fps") if row.settings else None
    return Job(
        id=row.id,
        state=row.state,  # type: ignore[arg-type]
        stage=row.stage,
        progress=min(1.0, max(0.0, row.progress)),
        frames_done=max(0, row.frames_done),
        frames_total=row.frames_total,
        error_code=row.error_code,
        error_message=row.error_message,
        created_utc=row.created_utc,
        started_utc=row.started_utc,
        finished_utc=row.finished_utc,
        input=JobInput(sha256=row.input_sha256, bytes=row.input_bytes, ext=row.input_ext),  # type: ignore[arg-type]
        video=video,
        model=model,
        analysis_fps=float(fps) if isinstance(fps, int | float) and fps > 0 else settings.analysis_fps,
    )


def _live_from_dict(data: Any, run_id: str) -> RunLive | None:
    if not isinstance(data, dict):
        return None
    try:
        return RunLive.model_validate(data)
    except ValidationError:
        logger.warning("live state does not match RunLive", extra={"run_id": run_id, "code": "live_invalid"})
        return None


def read_live(client: redis.Redis, row: RunRow) -> RunLive | None:
    """Latest live state: Redis while the run is active, the persisted final state once terminal."""
    if row.terminal:
        return _live_from_dict((row.summary or {}).get("live"), row.id)
    try:
        raw = client.get(run_live_key(row.id))
    except redis.RedisError:
        raw = None
    if raw is None:
        return _live_from_dict((row.summary or {}).get("live"), row.id)
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return _live_from_dict(data, row.id)


def run_to_api(row: RunRow, live: RunLive | None) -> Run:
    route = RunRoute.model_validate(
        {
            "node_ids": row.route.get("node_ids", []),
            "polyline": row.route.get("polyline", []),
            "length_m": row.route.get("length_m", 0.0),
        }
    )
    return Run(
        id=row.id,
        state=row.state,  # type: ignore[arg-type]
        outcome=row.outcome,
        simulator=row.simulator,  # type: ignore[arg-type]
        perception_source=row.perception_source,  # type: ignore[arg-type]
        scenario=row.scenario,  # type: ignore[arg-type]
        seed=row.seed,
        start_node=row.start_node,
        destination_node=row.destination_node,
        route=route,
        created_utc=row.created_utc,
        started_utc=row.started_utc,
        finished_utc=row.finished_utc,
        error_code=row.error_code,
        error_message=row.error_message,
        live=live,
    )
