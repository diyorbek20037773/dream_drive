"""Owner-filtered Server-Sent Events (contracts/event.schema.json envelopes)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Header, Query, Request
from fastapi.responses import StreamingResponse
from redis.exceptions import RedisError

from app.api.deps import SessionDep, SettingsDep, get_services
from app.errors import ApiError, error_responses
from app.events.bus import build_envelope, format_sse, sse_stream
from app.schemas.api import ErrorCode
from app.schemas.events import EventEnvelope

router = APIRouter(tags=["events"])
logger = logging.getLogger("dreamdrive.api.events")

# Redis stream ids look like "1728561234567-0".
STREAM_ID_PATTERN = r"^\d{1,20}-\d{1,20}$"


class EventStreamResponse(StreamingResponse):
    """Documents `text/event-stream` in the OpenAPI; implementations may return any streaming SSE response."""

    media_type = "text/event-stream"


@router.get(
    "/events",
    response_class=EventStreamResponse,
    summary="Server-Sent Events for the caller's jobs and runs",
    responses={
        200: {
            "model": EventEnvelope,
            "description": "SSE stream. Each message: `event: <type>`, `id: <stream id>`, `data: <EventEnvelope "
            "JSON>`; a `: ping` comment every 15 s. Missed camera frames are not replayed; resync via REST.",
        },
        **error_responses(401, 422, 503),
    },
)
async def stream_events(
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
    last_event_id_header: Annotated[
        str | None, Header(alias="Last-Event-ID", pattern=STREAM_ID_PATTERN, description="Resume after this id.")
    ] = None,
    last_event_id: Annotated[
        str | None,
        Query(pattern=STREAM_ID_PATTERN, description="Resume after this id (used when the header cannot be set)."),
    ] = None,
) -> EventStreamResponse:
    services = get_services(request)
    resume = last_event_id_header or last_event_id
    aredis = services.async_redis_factory()
    try:
        await aredis.ping()
    except RedisError as exc:
        await aredis.aclose()
        raise ApiError(503, ErrorCode.SERVICE_UNAVAILABLE, "Event stream unavailable") from exc
    owner = session.session_id

    async def body() -> AsyncIterator[str]:
        try:
            async for chunk in sse_stream(
                aredis,
                owner,
                resume,
                request.is_disconnected,
                heartbeat_s=services.sse_heartbeat_s,
                max_stream_s=services.sse_max_stream_s,
            ):
                yield chunk
        except RedisError:
            logger.warning("event stream lost its Redis connection", extra={"code": "redis_error"})
            envelope = build_envelope(
                "error", owner, {"code": "service_unavailable", "message": "Event stream interrupted; reconnecting"}
            )
            yield format_sse("0-0", "error", envelope.model_dump_json()).replace("id: 0-0\n", "", 1)
        finally:
            await aredis.aclose()

    return EventStreamResponse(
        body(),
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
