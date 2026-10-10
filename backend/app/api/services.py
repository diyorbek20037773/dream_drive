"""Per-application service container (SQLite, Redis, event bus, rate limiter, session signer, model status).

Constructing it opens no connection: SQLite opens on first use and redis-py connects lazily, so ``create_app()`` and
the OpenAPI export work offline. Tests build their own container (temporary SQLite, fakeredis, fake inference).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import redis
import redis.asyncio as aioredis

from app import metrics
from app.config import Settings
from app.db import Database, get_database
from app.events.bus import HEARTBEAT_S, MAX_STREAM_S, EventBus
from app.jobs.queue import redis_from_url
from app.schemas.inference import ModelInfo
from app.security import RateLimiter, SessionSigner

logger = logging.getLogger("dreamdrive.api.services")

MODEL_STATUS_TTL_READY_S = 10.0
MODEL_STATUS_TTL_NOT_READY_S = 3.0
MODEL_STATUS_TIMEOUT_S = 3.0


@dataclass(frozen=True, slots=True)
class ModelStatus:
    info: ModelInfo | None
    reason: str | None

    @property
    def ready(self) -> bool:
        return self.info is not None and self.info.ready


class ModelStatusCache:
    """Short-lived cache of the inference service's ``/v1/model`` answer (readiness gate for uploads)."""

    def __init__(self, client_factory: Callable[[], Any]) -> None:
        self._factory = client_factory
        self._lock = threading.Lock()
        self._value: tuple[float, ModelStatus] | None = None

    def get(self) -> ModelStatus:
        now = time.monotonic()
        with self._lock:
            if self._value is not None:
                expires, status = self._value
                if now < expires:
                    return status
        status = self._fetch()
        ttl = MODEL_STATUS_TTL_READY_S if status.ready else MODEL_STATUS_TTL_NOT_READY_S
        with self._lock:
            self._value = (time.monotonic() + ttl, status)
        return status

    def invalidate(self) -> None:
        with self._lock:
            self._value = None

    def _fetch(self) -> ModelStatus:
        from app.perception.client import InferenceClientError

        client = None
        try:
            client = self._factory()
            info = client.model_info()
        except InferenceClientError as exc:
            return ModelStatus(None, "inference_unreachable" if exc.code == "inference_unreachable" else exc.code)
        except Exception:  # noqa: BLE001 - any failure means "not ready", never a fake ready model
            logger.warning("model status lookup failed", extra={"code": "inference_unavailable"})
            return ModelStatus(None, "inference_unavailable")
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        metrics.MODEL_READY.labels(info.model_id, info.device or "unknown").set(1.0 if info.ready else 0.0)
        return ModelStatus(info, None if info.ready else (info.reason or "model_not_ready"))


def _default_inference_factory(settings: Settings) -> Callable[[], Any]:
    def factory() -> Any:
        from app.perception.client import InferenceClient

        return InferenceClient(
            settings.inference_base_url, min(MODEL_STATUS_TIMEOUT_S, settings.inference_timeout_seconds)
        )

    return factory


@dataclass
class Services:
    settings: Settings
    db: Database
    redis: redis.Redis
    async_redis_factory: Callable[[], Any]
    model_status: ModelStatusCache
    signer: SessionSigner
    bus: EventBus = field(init=False)
    limiter: RateLimiter = field(init=False)
    sse_heartbeat_s: float = HEARTBEAT_S
    sse_max_stream_s: float = MAX_STREAM_S

    def __post_init__(self) -> None:
        self.bus = EventBus(self.redis)
        self.limiter = RateLimiter(self.redis)

    @classmethod
    def from_settings(cls, settings: Settings) -> Services:
        url = settings.redis_url

        def async_factory() -> Any:
            # One client per SSE stream: a blocking XREAD occupies its connection anyway.
            return aioredis.Redis.from_url(url, socket_timeout=30.0, socket_connect_timeout=3.0)

        return cls(
            settings=settings,
            db=get_database(settings.sqlite_path),
            redis=redis_from_url(url),
            async_redis_factory=async_factory,
            model_status=ModelStatusCache(_default_inference_factory(settings)),
            signer=SessionSigner(settings.effective_secret()),
        )
