"""Process-wide resources for RQ job functions (settings, SQLite, Redis, event bus) and the shutdown flag.

RQ calls job functions by dotted path with only the resource id, so the job code obtains its dependencies here.
Tests install their own context (temporary SQLite + fakeredis) with :func:`set_context`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import redis

from app.config import Settings, get_settings
from app.db import Database, get_database
from app.events.bus import EventBus
from app.jobs.queue import redis_from_url

# Set by the worker's SIGTERM handling (app/jobs/worker_main.py): running jobs stop between frames/ticks and record
# an interrupted/failed state with error_code "worker_shutdown" instead of being killed mid-write.
SHUTDOWN = threading.Event()


@dataclass
class WorkerContext:
    settings: Settings
    db: Database
    redis: redis.Redis
    bus: EventBus


_context: WorkerContext | None = None
_lock = threading.Lock()


def build_context(settings: Settings | None = None) -> WorkerContext:
    settings = settings or get_settings()
    client = redis_from_url(settings.redis_url)
    return WorkerContext(settings=settings, db=get_database(settings.sqlite_path), redis=client, bus=EventBus(client))


def get_context() -> WorkerContext:
    global _context
    with _lock:
        if _context is None:
            _context = build_context()
        return _context


def set_context(context: WorkerContext | None) -> None:
    global _context
    with _lock:
        _context = context
