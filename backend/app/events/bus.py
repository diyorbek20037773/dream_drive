"""Per-owner event streams on Redis Streams and their Server-Sent Events rendering.

Each owner (session id) has one stream ``dd:events:<owner_id>`` capped at ~1000 entries; an event can only be
published to and read from its owner's stream, so SSE filtering cannot leak another owner's ids. Entries hold the
validated ``EventEnvelope`` JSON (contracts/event.schema.json). The Redis stream id is the SSE ``id:`` so a client
reconnecting with ``Last-Event-ID`` resumes exactly after the last event it saw. Events carry ids and small state
only; camera frames are fetched through the REST frame endpoint, never pushed as base64.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import redis
from pydantic import ValidationError

from app.schemas.events import EventEnvelope, EventType

logger = logging.getLogger("dreamdrive.events")

STREAM_MAXLEN = 1000
# Streams of idle sessions expire; a session lives 24 h, so 48 h keeps every reconnect window covered.
STREAM_TTL_S = 48 * 3600
HEARTBEAT_S = 15.0
# Connections are recycled; EventSource reconnects automatically with Last-Event-ID (no events are lost).
MAX_STREAM_S = 300.0
RETRY_MS = 3000


def stream_key(owner_id: str) -> str:
    return f"dd:events:{owner_id}"


def build_envelope(event_type: EventType, resource_id: str, payload: dict[str, Any]) -> EventEnvelope:
    return EventEnvelope.model_validate(
        {
            "event_id": uuid.uuid4().hex,
            "timestamp_utc": datetime.now(UTC),
            "resource_id": resource_id,
            "type": event_type,
            "payload": {"kind": event_type, **payload},
        }
    )


class EventBus:
    """Synchronous publisher used by the API threads and the worker jobs."""

    def __init__(self, client: redis.Redis) -> None:
        self.redis = client

    def publish(self, owner_id: str, event_type: EventType, resource_id: str, payload: dict[str, Any]) -> str | None:
        """Append one event; returns the stream id. Never raises for Redis/validation trouble (logged instead):
        clients resync through REST, so a lost event must not fail a job or a request."""
        try:
            envelope = build_envelope(event_type, resource_id, payload)
        except ValidationError:
            logger.exception("event payload failed validation", extra={"code": "event_invalid"})
            return None
        key = stream_key(owner_id)
        try:
            pipe = self.redis.pipeline(transaction=True)
            pipe.xadd(
                key,
                {"type": event_type, "data": envelope.model_dump_json()},
                maxlen=STREAM_MAXLEN,
                approximate=True,
            )
            pipe.expire(key, STREAM_TTL_S)
            stream_id = pipe.execute()[0]
        except redis.RedisError:
            logger.warning("event publish failed", extra={"code": "event_publish_failed"})
            return None
        return stream_id.decode() if isinstance(stream_id, bytes) else str(stream_id)


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def format_sse(stream_id: str, event_type: str, data: str) -> str:
    # `data` is single-line JSON (json.dumps never emits raw newlines), so one data: line is enough.
    return f"id: {stream_id}\nevent: {event_type}\ndata: {data}\n\n"


def _entry_to_sse(stream_id: Any, fields: dict[Any, Any]) -> str | None:
    decoded = {_text(k): _text(v) for k, v in fields.items()}
    data = decoded.get("data")
    event_type = decoded.get("type")
    if not data or not event_type:
        return None
    try:
        json.loads(data)
    except ValueError:
        return None
    return format_sse(_text(stream_id), event_type, data)


async def sse_stream(
    aredis: Any,
    owner_id: str,
    last_event_id: str | None,
    is_disconnected: Callable[[], Awaitable[bool]],
    *,
    heartbeat_s: float = HEARTBEAT_S,
    max_stream_s: float = MAX_STREAM_S,
    block_ms: int = 5000,
) -> AsyncIterator[str]:
    """Yield SSE messages for one owner's stream until the client disconnects or ``max_stream_s`` elapses.

    Without ``Last-Event-ID`` the stream starts at the newest entry (no history replay: the client loads current
    state through REST first). With it, every retained entry after that id is delivered, then live entries.
    A ``: ping`` comment is sent whenever nothing was written for ``heartbeat_s``.
    """
    key = stream_key(owner_id)
    yield f"retry: {RETRY_MS}\n: connected\n\n"
    if last_event_id is None:
        latest = await aredis.xrevrange(key, count=1)
        cursor = _text(latest[0][0]) if latest else "0-0"
    else:
        cursor = last_event_id
        still_there = await aredis.xrange(key, min=cursor, max=cursor, count=1)
        oldest = await aredis.xrange(key, count=1)
        if not still_there and oldest and _stream_id_tuple(_text(oldest[0][0])) > _stream_id_tuple(cursor):
            # The resume point was trimmed (MAXLEN), so events may be missing: the client reloads state via REST.
            yield ": resync\n\n"
    started = last_write = time.monotonic()
    while True:
        now = time.monotonic()
        if now - started >= max_stream_s or await is_disconnected():
            return
        wait_ms = int(
            max(1.0, min(block_ms, (heartbeat_s - (now - last_write)) * 1000, (max_stream_s - (now - started)) * 1000))
        )
        response = await aredis.xread({key: cursor}, block=wait_ms, count=100)
        for _stream, entries in response or []:
            for stream_id, fields in entries:
                cursor = _text(stream_id)
                message = _entry_to_sse(stream_id, fields)
                if message is not None:
                    yield message
                    last_write = time.monotonic()
        if time.monotonic() - last_write >= heartbeat_s:
            yield ": ping\n\n"
            last_write = time.monotonic()


def _stream_id_tuple(stream_id: str) -> tuple[int, int]:
    ms, _, seq = stream_id.partition("-")
    return int(ms), int(seq or 0)
