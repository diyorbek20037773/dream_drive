"""Scrape-time worker metrics: queue depth/oldest age, worker heartbeat age and artifact bytes.

Registered on the worker supervisor's registry next to the multiprocess collector. Values are measured when
Prometheus scrapes; nothing is a running estimate. A slot whose heartbeat key is missing reports +Inf so a "worker
heartbeat stale" alert fires instead of the series silently disappearing.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import time
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import redis
from prometheus_client.core import GaugeMetricFamily, Metric
from prometheus_client.registry import Collector
from rq import Queue
from rq.job import Job

from app import metrics
from app.config import Settings
from app.jobs.queue import QUEUE_NAMES, worker_heartbeat_key

logger = logging.getLogger("dreamdrive.jobs.monitoring")

ARTIFACT_SCAN_INTERVAL_S = 60.0


def _tree_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                pass
    return total


class WorkerStateCollector(Collector):
    def __init__(self, client: redis.Redis, settings: Settings, slots: Iterable[str]) -> None:
        self.redis = client
        self.settings = settings
        self.slots = tuple(slots)
        self._artifact_cache: tuple[float, dict[str, int]] | None = None

    def _artifact_bytes(self) -> dict[str, int]:
        now = time.monotonic()
        if self._artifact_cache and now - self._artifact_cache[0] < ARTIFACT_SCAN_INTERVAL_S:
            return self._artifact_cache[1]
        s = self.settings
        values = {
            "jobs": _tree_bytes(s.artifact_root / "jobs"),
            "runs": _tree_bytes(s.artifact_root / "runs"),
            "evaluations": _tree_bytes(s.evaluation_root),
            "tmp": _tree_bytes(s.tmp_root),
        }
        self._artifact_cache = (now, values)
        return values

    def collect(self) -> Iterator[Metric]:
        depth = GaugeMetricFamily(metrics.QUEUE_DEPTH_NAME, "Jobs waiting in the RQ queue.", labels=["queue"])
        oldest = GaugeMetricFamily(
            metrics.QUEUE_OLDEST_AGE_NAME,
            "Age of the oldest waiting job in the queue (0 when empty).",
            labels=["queue"],
        )
        try:
            for name in QUEUE_NAMES:
                queue = Queue(name, connection=self.redis)
                ids = queue.get_job_ids(0, 0)
                depth.add_metric([name], float(queue.count))
                age = 0.0
                if ids:
                    job = Job.fetch(ids[0], connection=self.redis)
                    if job.enqueued_at is not None:
                        enqueued = job.enqueued_at
                        if enqueued.tzinfo is None:
                            enqueued = enqueued.replace(tzinfo=UTC)
                        age = max(0.0, (datetime.now(UTC) - enqueued).total_seconds())
                oldest.add_metric([name], age)
        except redis.RedisError:
            logger.warning("queue metrics unavailable", extra={"code": "redis_error"})
        except Exception:  # noqa: BLE001 - a job vanished between listing and fetching; skip this scrape's age
            logger.debug("queue age lookup raced with a dequeue")
        yield depth
        yield oldest

        heartbeat = GaugeMetricFamily(
            metrics.WORKER_HEARTBEAT_AGE_NAME,
            "Seconds since the worker slot last wrote its heartbeat.",
            labels=["slot"],
        )
        now = time.time()
        for slot in self.slots:
            try:
                raw = self.redis.get(worker_heartbeat_key(slot))
            except redis.RedisError:
                raw = None
            age = math.inf
            if raw is not None:
                try:
                    age = max(0.0, now - float(raw))
                except ValueError:
                    age = math.inf
            heartbeat.add_metric([slot], age)
        yield heartbeat

        art = GaugeMetricFamily(metrics.ARTIFACT_BYTES_NAME, "Bytes stored on the data volume.", labels=["category"])
        for category, value in self._artifact_bytes().items():
            art.add_metric([category], float(value))
        yield art

        # Free bytes on the data volume; feeds the DreamDriveDataVolumeLow alert. Omitted (not zero) when unreadable.
        try:
            free = float(shutil.disk_usage(self.settings.data_root).free)
        except OSError:
            logger.warning("data volume usage unavailable", extra={"code": "disk_usage_error"})
        else:
            gauge = GaugeMetricFamily(metrics.DATA_VOLUME_FREE_NAME, "Free bytes on the data volume.")
            gauge.add_metric([], free)
            yield gauge
