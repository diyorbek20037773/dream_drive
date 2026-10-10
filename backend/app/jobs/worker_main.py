"""Worker supervisor: ``python -m app.jobs.worker_main``.

- applies DB migrations, waits for Redis, reconciles work interrupted by a previous worker (never silently retried);
- starts two RQ ``SimpleWorker`` child processes (spawned, so no SQLite/Redis connection is shared across a fork):
  slot ``video`` (queue ``video``: uploads and the scenario suite) and slot ``sim`` (queue ``sim``: the interactive
  run). Jobs execute inside the child itself; inference is an HTTP call to the inference service;
- exports Prometheus metrics on WORKER_METRICS_PORT from the supervisor using prometheus_client multiprocess mode
  (``PROMETHEUS_MULTIPROC_DIR``) plus scrape-time queue/heartbeat/artifact gauges;
- runs retention cleanup every 10 minutes;
- exits non-zero when a child dies unexpectedly (Compose restarts the container, and reconciliation runs again).
On SIGTERM/SIGINT the children get a warm shutdown: running jobs stop between frames/ticks and record
``interrupted``/``failed`` with ``error_code=worker_shutdown``.
"""

from __future__ import annotations

import os
import tempfile

# Must happen before anything imports prometheus_client: the value backend is chosen at import time.
os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", os.path.join(tempfile.gettempdir(), "dd-worker-metrics"))

import logging  # noqa: E402
import multiprocessing  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402
from types import FrameType  # noqa: E402
from typing import Any  # noqa: E402

import redis  # noqa: E402
from prometheus_client import CollectorRegistry, multiprocess, start_http_server  # noqa: E402
from rq import Queue, SimpleWorker  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import get_database  # noqa: E402
from app.events.bus import EventBus  # noqa: E402
from app.jobs.context import SHUTDOWN  # noqa: E402
from app.jobs.maintenance import reconcile_after_restart, retention_cleanup  # noqa: E402
from app.jobs.monitoring import WorkerStateCollector  # noqa: E402
from app.jobs.queue import (  # noqa: E402
    SIM_QUEUE,
    VIDEO_QUEUE,
    WORKER_HEARTBEAT_TTL_S,
    redis_from_url,
    worker_heartbeat_key,
)
from app.logging_setup import configure_logging  # noqa: E402

logger = logging.getLogger("dreamdrive.worker")

SLOTS: dict[str, tuple[str, ...]] = {"video": (VIDEO_QUEUE,), "sim": (SIM_QUEUE,)}
HEARTBEAT_INTERVAL_S = 5.0
RETENTION_INTERVAL_S = 600.0
CHILD_STOP_GRACE_S = 50.0
REDIS_WAIT_S = 60.0


class DreamDriveWorker(SimpleWorker):
    """SimpleWorker whose warm shutdown also tells the running job to stop at the next frame/tick."""

    def request_stop(self, signum: Any, frame: Any) -> None:
        SHUTDOWN.set()
        super().request_stop(signum, frame)


def _heartbeat_loop(client: redis.Redis, slot: str) -> None:
    key = worker_heartbeat_key(slot)
    while True:
        try:
            client.set(key, f"{time.time():.3f}", ex=WORKER_HEARTBEAT_TTL_S)
        except redis.RedisError:
            logger.warning("heartbeat write failed", extra={"slot": slot, "code": "redis_error"})
        time.sleep(HEARTBEAT_INTERVAL_S)


def child_main(slot: str, queue_names: tuple[str, ...]) -> None:
    settings = get_settings()
    configure_logging(f"worker-{slot}", settings.log_level)
    client = redis_from_url(settings.redis_url, socket_timeout=None)
    threading.Thread(target=_heartbeat_loop, args=(client, slot), name=f"heartbeat-{slot}", daemon=True).start()
    queues = [Queue(name, connection=client) for name in queue_names]
    worker = DreamDriveWorker(queues, connection=client, name=f"dreamdrive-{slot}-{uuid.uuid4().hex[:8]}")
    logger.info("worker slot started", extra={"slot": slot, "queue": ",".join(queue_names), "pid": os.getpid()})
    worker.work(with_scheduler=False, logging_level=settings.log_level.upper())
    logger.info("worker slot stopped", extra={"slot": slot})


def _prepare_multiproc_dir() -> Path:
    path = Path(os.environ["PROMETHEUS_MULTIPROC_DIR"])
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _wait_for_redis(client: redis.Redis) -> None:
    deadline = time.monotonic() + REDIS_WAIT_S
    while True:
        try:
            client.ping()
            return
        except redis.RedisError:
            if time.monotonic() > deadline:
                raise
            time.sleep(1.0)


def main() -> int:
    settings = get_settings()
    configure_logging("worker", settings.log_level)
    _prepare_multiproc_dir()
    for path in (settings.artifact_root, settings.tmp_root, settings.evaluation_root):
        path.mkdir(parents=True, exist_ok=True)
    db = get_database(settings.sqlite_path)
    db.migrate()
    client = redis_from_url(settings.redis_url)
    _wait_for_redis(client)
    reconcile_after_restart(db, client, EventBus(client), settings.artifact_root)

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    registry.register(WorkerStateCollector(client, settings, SLOTS))
    start_http_server(settings.worker_metrics_port, registry=registry)

    mp = multiprocessing.get_context("spawn")
    children: dict[str, Any] = {}
    for slot, queues in SLOTS.items():
        proc = mp.Process(target=child_main, args=(slot, queues), name=f"dreamdrive-{slot}")
        proc.start()
        children[slot] = proc

    stopping = threading.Event()

    def _on_signal(signum: int, _frame: FrameType | None) -> None:
        if stopping.is_set():
            return
        stopping.set()
        logger.info("shutdown requested; stopping worker slots", extra={"code": "worker_stop"})
        for proc in children.values():
            if proc.is_alive() and proc.pid:
                os.kill(proc.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    next_cleanup = time.monotonic() + RETENTION_INTERVAL_S
    exit_code = 0
    while not stopping.is_set():
        dead = [slot for slot, proc in children.items() if not proc.is_alive()]
        if dead:
            for slot in dead:
                proc = children[slot]
                logger.error(
                    "worker slot exited unexpectedly",
                    extra={"slot": slot, "code": "worker_child_died", "status": proc.exitcode},
                )
                if proc.pid:
                    multiprocess.mark_process_dead(proc.pid)
            exit_code = 1
            stopping.set()
            for proc in children.values():
                if proc.is_alive() and proc.pid:
                    os.kill(proc.pid, signal.SIGTERM)
            break
        if time.monotonic() >= next_cleanup:
            next_cleanup = time.monotonic() + RETENTION_INTERVAL_S
            try:
                retention_cleanup(settings, db, client)
            except Exception:
                logger.exception("retention cleanup failed", extra={"code": "retention_failed"})
        stopping.wait(1.0)

    deadline = time.monotonic() + CHILD_STOP_GRACE_S
    for proc in children.values():
        proc.join(max(0.0, deadline - time.monotonic()))
    for proc in children.values():
        if proc.is_alive():
            logger.error("worker slot did not stop in time; killing", extra={"code": "worker_kill"})
            proc.kill()
            proc.join(5)
        if proc.pid:
            multiprocess.mark_process_dead(proc.pid)
    logger.info("worker stopped", extra={"status": exit_code})
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
