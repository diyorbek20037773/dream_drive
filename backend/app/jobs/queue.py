"""Redis connection, RQ queues and the Redis key names shared by the API and the worker.

Queues: ``video`` (uploaded-video analysis and, as shared low-priority work, the synthetic scenario suite) and
``sim`` (one interactive closed-loop run at a time). Jobs are enqueued by dotted path so the API process never
imports the heavy pipeline/simulator modules. RQ job ids are prefixed (``video-<id>``) to keep kinds apart.

SQLite is the source of truth for job/run state; RQ only dispatches. Cancellation of queued work removes the RQ
job; running work observes a persistent DB flag plus a Redis key between frames/ticks.
"""

from __future__ import annotations

import logging

import redis
from rq import Queue
from rq.exceptions import InvalidJobOperation, NoSuchJobError
from rq.job import Job, JobStatus

from app.config import Settings

logger = logging.getLogger("dreamdrive.queue")

VIDEO_QUEUE = "video"
SIM_QUEUE = "sim"
EVAL_QUEUE = VIDEO_QUEUE  # shared low-priority queue (one CPU-heavy worker slot)
QUEUE_NAMES: tuple[str, ...] = (VIDEO_QUEUE, SIM_QUEUE)

VIDEO_JOB_FUNC = "app.jobs.video_job.run_video_job"
SIM_JOB_FUNC = "app.jobs.sim_job.run_simulation_job"
EVAL_JOB_FUNC = "app.jobs.eval_job.run_scenario_suite_job"

RESULT_TTL_S = 3600
FAILURE_TTL_S = 24 * 3600
RUN_KEY_TTL_S = 60
JOB_CANCEL_TTL_S = 24 * 3600
WORKER_HEARTBEAT_TTL_S = 60
# Per-frame budget used to bound a video job: every analyzed frame may take up to this long end to end on CPU.
_VIDEO_SECONDS_PER_FRAME_BUDGET = 4.0
_VIDEO_FIXED_BUDGET_S = 300
_EVAL_SECONDS_PER_SEED = 900


def redis_from_url(url: str, *, socket_timeout: float | None = 5.0) -> redis.Redis:
    return redis.Redis.from_url(
        url, socket_timeout=socket_timeout, socket_connect_timeout=3.0, health_check_interval=30
    )


def rq_video_id(job_id: str) -> str:
    return f"video-{job_id}"


def rq_sim_id(run_id: str) -> str:
    return f"sim-{run_id}"


def rq_eval_id(eval_id: str) -> str:
    return f"eval-{eval_id}"


# Redis keys -----------------------------------------------------------------------------------------------------------
def job_cancel_key(job_id: str) -> str:
    return f"dd:job:{job_id}:cancel"


def run_frame_key(run_id: str) -> str:
    return f"dd:run:{run_id}:frame"


def run_live_key(run_id: str) -> str:
    return f"dd:run:{run_id}:live"


def run_control_key(run_id: str) -> str:
    return f"dd:run:{run_id}:control"


def run_inject_key(run_id: str) -> str:
    return f"dd:run:{run_id}:inject"


def run_keys(run_id: str) -> tuple[str, ...]:
    return run_frame_key(run_id), run_live_key(run_id), run_control_key(run_id), run_inject_key(run_id)


def worker_heartbeat_key(slot: str) -> str:
    return f"dd:worker:heartbeat:{slot}"


# Timeouts ----------------------------------------------------------------------------------------------------
def video_job_timeout_s(settings: Settings) -> int:
    """Upper bound for one video job: fixed overhead + a per-analyzed-frame budget at the configured limits."""
    frames = settings.max_video_seconds * settings.analysis_fps
    return int(_VIDEO_FIXED_BUDGET_S + frames * _VIDEO_SECONDS_PER_FRAME_BUDGET)


def sim_job_timeout_s(settings: Settings) -> int:
    # The job's own watchdog ends the run at MAX_SIMULATION_WALL_SECONDS; RQ's limit is the last resort.
    return int(settings.max_simulation_wall_seconds + 60)


def eval_job_timeout_s(seed_count: int) -> int:
    return int(_EVAL_SECONDS_PER_SEED * max(1, seed_count))


# Enqueue / cancel ------------------------------------------------------------------------------------------
def enqueue_video_job(client: redis.Redis, settings: Settings, job_id: str) -> None:
    Queue(VIDEO_QUEUE, connection=client).enqueue(
        VIDEO_JOB_FUNC,
        job_id,
        job_id=rq_video_id(job_id),
        job_timeout=video_job_timeout_s(settings),
        result_ttl=RESULT_TTL_S,
        failure_ttl=FAILURE_TTL_S,
        description=f"video_analysis {job_id}",
    )


def enqueue_sim_job(client: redis.Redis, settings: Settings, run_id: str) -> None:
    Queue(SIM_QUEUE, connection=client).enqueue(
        SIM_JOB_FUNC,
        run_id,
        job_id=rq_sim_id(run_id),
        job_timeout=sim_job_timeout_s(settings),
        result_ttl=RESULT_TTL_S,
        failure_ttl=FAILURE_TTL_S,
        description=f"simulation {run_id}",
    )


def enqueue_eval_job(client: redis.Redis, eval_id: str, seed_count: int) -> None:
    Queue(EVAL_QUEUE, connection=client).enqueue(
        EVAL_JOB_FUNC,
        eval_id,
        job_id=rq_eval_id(eval_id),
        job_timeout=eval_job_timeout_s(seed_count),
        result_ttl=RESULT_TTL_S,
        failure_ttl=FAILURE_TTL_S,
        description=f"scenario_suite {eval_id}",
    )


def cancel_if_queued(client: redis.Redis, rq_job_id: str) -> bool:
    """Remove a not-yet-started RQ job. True if it was still waiting (and is now cancelled)."""
    try:
        job = Job.fetch(rq_job_id, connection=client)
    except NoSuchJobError:
        return False
    status = job.get_status(refresh=True)
    if status not in (JobStatus.QUEUED, JobStatus.DEFERRED, JobStatus.SCHEDULED):
        return False
    try:
        job.cancel()
    except InvalidJobOperation:
        return False
    return True


def rq_job_exists(client: redis.Redis, rq_job_id: str) -> bool:
    return bool(Job.exists(rq_job_id, connection=client))
