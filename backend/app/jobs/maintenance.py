"""Worker start-up reconciliation and retention cleanup.

Reconciliation (docs/ARCHITECTURE.md "Persistence and failure"): work that was in progress when the worker died is
marked ``interrupted`` (jobs/evaluations) or ``failed`` (runs) with ``error_code=worker_restarted``; it is never
silently retried. Queued jobs whose RQ entry vanished (Redis data loss) are marked interrupted (``queue_lost``).

Retention: terminal jobs/runs older than ARTIFACT_RETENTION_HOURS lose their bytes and rows; expired sessions,
old tombstones, orphaned artifact directories and stale temporary uploads are removed.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import redis

from app.config import Settings
from app.db import Database, utc_now
from app.events.bus import EventBus
from app.jobs.queue import (
    cancel_if_queued,
    job_cancel_key,
    rq_eval_id,
    rq_job_exists,
    rq_sim_id,
    rq_video_id,
    run_keys,
)
from app.storage import artifacts, repo

logger = logging.getLogger("dreamdrive.jobs.maintenance")

STALE_TMP_S = 3600


@dataclass(frozen=True)
class ReconcileStats:
    jobs_interrupted: int
    runs_failed: int
    evaluations_interrupted: int


def reconcile_after_restart(
    db: Database, client: redis.Redis, bus: EventBus | None, artifact_root: Path
) -> ReconcileStats:
    now = utc_now()
    jobs_interrupted = runs_failed = evals_interrupted = 0
    for job in repo.jobs_in_states(db, repo.JOB_ACTIVE_STATES):
        if job.state == "queued":
            try:
                if rq_job_exists(client, rq_video_id(job.id)):
                    continue
            except redis.RedisError:
                continue  # cannot tell; leave it queued rather than guess
            code, message = "queue_lost", "The queued job was lost from the queue"
        else:
            code, message = "worker_restarted", "The worker restarted while the job was running"
        if repo.transition_job(
            db, job.id, (job.state,), "interrupted", error_code=code, error_message=message, finished_utc=now
        ):
            jobs_interrupted += 1
            if bus is not None:
                bus.publish(
                    job.owner_id, "job_state", job.id, {"job_id": job.id, "state": "interrupted", "error_code": code}
                )
        if job.delete_requested:
            artifacts.remove_job_artifacts(artifact_root, job.id)
            repo.delete_job_row(db, job.id, job.owner_id)
        try:
            client.delete(job_cancel_key(job.id))
        except redis.RedisError:
            pass
    for run in repo.runs_in_states(db, repo.RUN_ACTIVE_STATES):
        try:
            cancel_if_queued(client, rq_sim_id(run.id))
            client.delete(*run_keys(run.id))
        except redis.RedisError:
            pass
        if repo.transition_run(
            db,
            run.id,
            (run.state,),
            "failed",
            outcome="failed_worker_restarted",
            error_code="worker_restarted",
            error_message="The worker restarted during the run",
            finished_utc=now,
        ):
            runs_failed += 1
            if bus is not None:
                bus.publish(
                    run.owner_id,
                    "run_state",
                    run.id,
                    {"run_id": run.id, "state": "failed", "outcome": "failed_worker_restarted"},
                )
    for ev in repo.evaluations_in_states(db, repo.EVAL_ACTIVE_STATES):
        if ev.state == "queued":
            try:
                if rq_job_exists(client, rq_eval_id(ev.id)):
                    continue
            except redis.RedisError:
                continue
            code = "queue_lost"
        else:
            code = "worker_restarted"
        if repo.transition_evaluation(db, ev.id, (ev.state,), "interrupted", error_code=code, finished_utc=now):
            evals_interrupted += 1
    stats = ReconcileStats(jobs_interrupted, runs_failed, evals_interrupted)
    logger.info(
        "startup reconciliation",
        extra={"code": "reconciled", "removed_jobs": jobs_interrupted, "removed_runs": runs_failed},
    )
    return stats


@dataclass(frozen=True)
class CleanupStats:
    removed_jobs: int
    removed_runs: int
    removed_evaluations: int
    removed_bytes: int
    removed_sessions: int
    started_utc: datetime
    finished_utc: datetime


def _older_than(path: Path, cutoff_ts: float) -> bool:
    try:
        return path.stat().st_mtime < cutoff_ts
    except OSError:
        return False


def retention_cleanup(
    settings: Settings, db: Database, client: redis.Redis | None = None, now: datetime | None = None
) -> CleanupStats:
    started = utc_now()
    now = now or started
    cutoff = now - timedelta(hours=settings.artifact_retention_hours)
    cutoff_ts = cutoff.timestamp()
    root = settings.artifact_root
    removed_bytes = removed_jobs = removed_runs = 0

    for job in repo.jobs_created_before(db, cutoff, repo.JOB_TERMINAL_STATES):
        removed_bytes += artifacts.remove_job_artifacts(root, job.id)
        repo.delete_job_row(db, job.id, job.owner_id)
        removed_jobs += 1
    for run in repo.runs_created_before(db, cutoff, repo.RUN_TERMINAL_STATES):
        removed_bytes += artifacts.remove_tree(artifacts.run_dir(root, run.id))
        repo.delete_run_row(db, run.id)
        if client is not None:
            try:
                client.delete(*run_keys(run.id))
            except redis.RedisError:
                pass
        removed_runs += 1
    removed_evaluations = repo.delete_expired_evaluation_rows(db, cutoff)
    removed_sessions = repo.delete_expired_sessions(db, now - timedelta(hours=settings.session_ttl_hours))
    repo.delete_expired_tombstones(db, cutoff)

    # Orphans: directories without a DB row (crash between file move and insert, manual edits), only once old.
    known_jobs = repo.job_ids(db)
    jobs_root = root / "jobs"
    if jobs_root.is_dir():
        for entry in jobs_root.iterdir():
            name = entry.name
            if name.startswith(".") and ".partial-" in name:
                if _older_than(entry, time.time() - STALE_TMP_S):
                    removed_bytes += artifacts.remove_tree(entry)
            elif artifacts.SERVER_ID_RE.match(name) and name not in known_jobs and _older_than(entry, cutoff_ts):
                removed_bytes += artifacts.remove_tree(entry)
    known_runs = repo.run_ids(db)
    runs_root = root / "runs"
    if runs_root.is_dir():
        for entry in runs_root.iterdir():
            if (
                artifacts.SERVER_ID_RE.match(entry.name)
                and entry.name not in known_runs
                and _older_than(entry, cutoff_ts)
            ):
                removed_bytes += artifacts.remove_tree(entry)
    tmp_root = settings.tmp_root
    if tmp_root.is_dir():
        for entry in tmp_root.iterdir():
            if entry.is_file() and _older_than(entry, time.time() - STALE_TMP_S):
                try:
                    size = entry.stat().st_size
                    os.unlink(entry)
                    removed_bytes += size
                except OSError:
                    pass
    finished = utc_now()
    logger.info(
        "retention cleanup",
        extra={
            "code": "retention_cleanup",
            "removed_jobs": removed_jobs,
            "removed_runs": removed_runs,
            "removed_bytes": removed_bytes,
        },
    )
    return CleanupStats(
        removed_jobs=removed_jobs,
        removed_runs=removed_runs,
        removed_evaluations=removed_evaluations,
        removed_bytes=removed_bytes,
        removed_sessions=removed_sessions,
        started_utc=started,
        finished_utc=finished,
    )
