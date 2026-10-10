"""RQ entry point for uploaded-video analysis: ``run_video_job(job_id)``.

State machine (SQLite is authoritative): queued -> validating (ffprobe limits, model identity) -> running
(app.video.pipeline.run_video_analysis) -> succeeded | failed | cancelled | interrupted. A job that is no longer
``queued`` when the worker picks it up (cancelled meanwhile, re-delivered) is left untouched.

Cancellation is cooperative: the API sets ``cancel_requested`` in the DB and a Redis key; ``cancel_check`` (called
by the pipeline between frames) observes either. A job deleted while in progress is removed (bytes + row) as soon as
the worker stops it. Error messages stored for the user come from coded pipeline errors and never contain paths.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import redis
from rq.timeouts import JobTimeoutException

from app import metrics
from app.db import utc_now
from app.jobs.context import SHUTDOWN, WorkerContext, get_context
from app.jobs.queue import job_cancel_key
from app.logging_setup import log_context
from app.storage import artifacts, repo

logger = logging.getLogger("dreamdrive.jobs.video")

KIND = "video_analysis"
PROGRESS_MIN_INTERVAL_S = 0.5
CANCEL_DB_POLL_S = 2.0
# Share of the progress bar used by per-frame analysis; encode/report stages fill the rest.
_ANALYSIS_SHARE = 0.95
_STAGE_PROGRESS = {"encoding": 0.96, "writing_report": 0.98, "done": 1.0}


class _MeteredClient:
    """Delegating InferenceClient wrapper that records per-frame inference metrics (server-reported model time)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def infer(self, *args: Any, **kwargs: Any) -> Any:
        try:
            result = self._inner.infer(*args, **kwargs)
        except Exception:
            metrics.FRAMES_PROCESSED.labels(KIND, "learned_model", "inference_failed").inc()
            raise
        metrics.INFERENCE_DURATION.labels(result.model.model_id, result.model.device, "ok").observe(
            max(0.0, float(result.timing_ms.inference) / 1000.0)
        )
        metrics.FRAMES_PROCESSED.labels(KIND, "learned_model", "ok").inc()
        return result


def make_inference_client(ctx: WorkerContext) -> Any:
    from app.perception.client import InferenceClient

    return InferenceClient(ctx.settings.inference_base_url, ctx.settings.inference_timeout_seconds)


def probe_input(input_path: Path, ctx: WorkerContext) -> dict[str, Any]:
    """ffprobe the stored upload against the configured limits; returns the ``Job.video`` dict.

    Raises app.video.errors.VideoError with a stable code (video_invalid, video_too_long, ...)."""
    from app.video.probe import ProbeLimits, probe_video

    s = ctx.settings
    probe = probe_video(input_path, ProbeLimits(s.max_video_seconds, s.max_video_width, s.max_video_height))
    return dict(probe.as_job_video())


def model_identity(client: Any) -> dict[str, Any] | None:
    """Identity of the ready model (``Job.model``); None when the service is unreachable/not ready."""
    from app.perception.client import InferenceClientError

    try:
        info = client.model_info()
    except InferenceClientError:
        return None
    if (
        not info.ready
        or not info.sha256
        or info.device not in ("cpu", "cuda")
        or info.precision not in ("fp32", "fp16")
    ):
        return None
    return {"model_id": info.model_id, "sha256": info.sha256, "device": info.device, "precision": info.precision}


def run_analysis(**kwargs: Any) -> dict[str, Any]:
    from app.video.pipeline import run_video_analysis

    return run_video_analysis(**kwargs)


class _Progress:
    def __init__(self, ctx: WorkerContext, job: repo.JobRow) -> None:
        self.ctx = ctx
        self.job = job
        self.running = False
        self._last_t = 0.0
        self._last_stage: str | None = None

    def mark_running(self) -> None:
        if self.running:
            return
        if repo.transition_job(self.ctx.db, self.job.id, ("validating",), "running", stage="analyzing"):
            self.ctx.bus.publish(
                self.job.owner_id, "job_state", self.job.id, {"job_id": self.job.id, "state": "running"}
            )
        self.running = True

    def __call__(self, frames_done: int, frames_total: int, stage: str) -> None:
        if stage != "probing":
            self.mark_running()
        now = time.monotonic()
        if stage == self._last_stage and now - self._last_t < PROGRESS_MIN_INTERVAL_S:
            return
        self._last_t, self._last_stage = now, stage
        total = max(int(frames_total or 0), int(frames_done or 0))
        if stage in _STAGE_PROGRESS:
            progress = _STAGE_PROGRESS[stage]
        elif total > 0:
            progress = _ANALYSIS_SHARE * min(1.0, frames_done / total)
        else:
            progress = 0.0
        repo.update_job(
            self.ctx.db,
            self.job.id,
            stage=stage,
            progress=round(progress, 4),
            frames_done=int(frames_done),
            frames_total=total or None,
        )
        self.ctx.bus.publish(
            self.job.owner_id,
            "job_progress",
            self.job.id,
            {
                "job_id": self.job.id,
                "progress": round(progress, 4),
                "stage": stage,
                "frames_done": int(frames_done),
                "frames_total": total,
            },
        )


class _CancelCheck:
    def __init__(self, ctx: WorkerContext, job_id: str) -> None:
        self.ctx = ctx
        self.job_id = job_id
        self._last_db = 0.0
        self.shutdown = False

    def __call__(self) -> bool:
        if SHUTDOWN.is_set():
            self.shutdown = True
            return True
        try:
            if self.ctx.redis.exists(job_cancel_key(self.job_id)):
                return True
        except redis.RedisError:
            pass  # fall through to the persistent flag
        now = time.monotonic()
        if now - self._last_db >= CANCEL_DB_POLL_S:
            self._last_db = now
            row = repo.get_job_any(self.ctx.db, self.job_id)
            if row is None or row.cancel_requested or row.delete_requested:
                return True
        return False


def _finish(ctx: WorkerContext, job: repo.JobRow, state: str, **fields: Any) -> None:
    applied = repo.transition_job(
        ctx.db,
        job.id,
        ("validating", "running", "cancelling"),
        state,
        finished_utc=utc_now(),
        **fields,
    )
    if applied:
        metrics.JOBS_TOTAL.labels(KIND, state).inc()
        ctx.bus.publish(
            job.owner_id,
            "job_state",
            job.id,
            {"job_id": job.id, "state": state, "error_code": fields.get("error_code")},
        )
        logger.info("job finished", extra={"state": state, "code": fields.get("error_code")})


def _cleanup_if_deleted(ctx: WorkerContext, job_id: str) -> None:
    row = repo.get_job_any(ctx.db, job_id)
    if row is None or not row.delete_requested:
        return
    removed = artifacts.remove_job_artifacts(ctx.settings.artifact_root, job_id)
    repo.delete_job_row(ctx.db, job_id, row.owner_id)
    logger.info("deleted job cleaned up after stop", extra={"code": "job_deleted", "removed_bytes": removed})


def run_video_job(job_id: str) -> str:
    """RQ job function. Returns the final state (for RQ's result record)."""
    ctx = get_context()
    with log_context(job_id=job_id):
        try:
            return _run(ctx, job_id)
        finally:
            try:
                ctx.redis.delete(job_cancel_key(job_id))
            except redis.RedisError:
                pass
            _cleanup_if_deleted(ctx, job_id)


def _run(ctx: WorkerContext, job_id: str) -> str:
    from app.video.errors import JobCancelled, VideoError

    db = ctx.db
    job = repo.get_job_any(db, job_id)
    if job is None:
        logger.warning("job row missing; skipping", extra={"code": "job_missing"})
        return "missing"
    if not repo.transition_job(db, job_id, ("queued",), "validating", stage="probing", started_utc=utc_now()):
        logger.info("job not queued any more; skipping", extra={"state": job.state})
        return job.state
    ctx.bus.publish(job.owner_id, "job_state", job_id, {"job_id": job_id, "state": "validating"})
    logger.info("job started", extra={"state": "validating"})

    out_dir = artifacts.job_dir(ctx.settings.artifact_root, job_id)
    input_path = out_dir / f"input.{job.input_ext}"
    progress = _Progress(ctx, job)
    cancel_check = _CancelCheck(ctx, job_id)
    client = _MeteredClient(make_inference_client(ctx))
    try:
        if not input_path.is_file():
            raise VideoError("input_missing", "The uploaded input is no longer available")
        repo.update_job(db, job_id, probe=probe_input(input_path, ctx))
        identity = model_identity(client)
        if identity is not None:
            repo.update_job(db, job_id, model=identity)
        if cancel_check():
            raise JobCancelled()
        summary = run_analysis(
            job_id=job_id,
            input_path=input_path,
            out_dir=out_dir,
            client=client,
            settings=ctx.settings,
            progress_cb=progress,
            cancel_check=cancel_check,
        )
        progress.mark_running()
        _finish(ctx, job, "succeeded", summary=summary, progress=1.0, stage="done", error_code=None, error_message=None)
        return "succeeded"
    except JobCancelled:
        if cancel_check.shutdown:
            _finish(
                ctx,
                job,
                "interrupted",
                error_code="worker_shutdown",
                error_message="The worker shut down while the job was running",
            )
            return "interrupted"
        _finish(ctx, job, "cancelled", stage="cancelled")
        return "cancelled"
    except VideoError as exc:
        _finish(ctx, job, "failed", error_code=exc.code, error_message=exc.message[:500])
        return "failed"
    except JobTimeoutException:
        _finish(ctx, job, "failed", error_code="job_timeout", error_message="The analysis exceeded its time budget")
        return "failed"
    except Exception:
        logger.exception("video job crashed", extra={"code": "internal_error"})
        _finish(ctx, job, "failed", error_code="internal_error", error_message="Internal error during analysis")
        return "failed"
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
