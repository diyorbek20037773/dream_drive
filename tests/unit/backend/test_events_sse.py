# ruff: noqa: F811, S105, S106 - pytest fixtures are imported from the shared harness; test-only passwords
"""Per-owner Redis Streams event bus and the SSE endpoint (ids, resume, heartbeat, owner filtering, schema)."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import fakeredis
import jsonschema
import pytest
from test_api_harness import Harness, harness  # noqa: F401

from app.events.bus import STREAM_MAXLEN, EventBus, sse_stream, stream_key
from app.schemas.events import EventEnvelope

ROOT = Path(__file__).resolve().parents[3]
EVENT_SCHEMA = json.loads((ROOT / "contracts" / "event.schema.json").read_text(encoding="utf-8"))


def _parse(text: str) -> list[dict[str, str]]:
    messages = []
    for block in text.split("\n\n"):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if line.startswith(":") or ":" not in line:
                continue
            key, _, value = line.partition(":")
            fields[key] = value.lstrip(" ")
        if "data" in fields:
            messages.append(fields)
    return messages


async def _collect(client: Any, owner: str, last_id: str | None, **kwargs: Any) -> str:
    async def never() -> bool:
        return False

    chunks = [c async for c in sse_stream(client, owner, last_id, never, **kwargs)]
    return "".join(chunks)


def test_publish_validates_envelope_and_trims() -> None:
    server = fakeredis.FakeServer()
    bus = EventBus(fakeredis.FakeRedis(server=server))
    stream_id = bus.publish("owner1", "job_state", "job1", {"job_id": "job1", "state": "queued"})
    assert stream_id and "-" in stream_id
    entries = bus.redis.xrange(stream_key("owner1"))
    envelope = json.loads(entries[0][1][b"data"])
    jsonschema.validate(envelope, EVENT_SCHEMA)
    EventEnvelope.model_validate(envelope)
    assert envelope["type"] == "job_state" and envelope["payload"]["kind"] == "job_state"
    assert bus.redis.ttl(stream_key("owner1")) > 0
    # Invalid payloads are rejected (logged), never written.
    assert bus.publish("owner1", "job_progress", "job1", {"job_id": "job1", "progress": 3.0}) is None
    assert len(bus.redis.xrange(stream_key("owner1"))) == 1
    for i in range(STREAM_MAXLEN + 50):
        bus.publish("owner2", "frame_ready", "r", {"run_id": "r", "frame_id": i, "sim_time_s": 0.0})
    assert bus.redis.xlen(stream_key("owner2")) <= STREAM_MAXLEN + 100


def test_publish_survives_redis_outage() -> None:
    server = fakeredis.FakeServer()
    bus = EventBus(fakeredis.FakeRedis(server=server))
    server.connected = False
    assert bus.publish("o", "job_state", "j", {"job_id": "j", "state": "failed"}) is None


def test_stream_owner_filter_resume_and_heartbeat() -> None:
    server = fakeredis.FakeServer()
    bus = EventBus(fakeredis.FakeRedis(server=server))
    a1 = bus.publish("A", "job_state", "jobA1", {"job_id": "jobA1", "state": "queued"})
    bus.publish("B", "job_state", "jobB1", {"job_id": "jobB1", "state": "queued"})
    a2 = bus.publish("A", "run_state", "runA", {"run_id": "runA", "state": "running", "outcome": None})

    def aclient() -> Any:  # one client per event loop (asyncio.run creates a new loop each time)
        return fakeredis.FakeAsyncRedis(server=server)

    text = asyncio.run(_collect(aclient(), "A", "0-0", heartbeat_s=0.2, max_stream_s=0.5, block_ms=100))
    messages = _parse(text)
    assert [m["id"] for m in messages] == [a1, a2]
    assert [m["event"] for m in messages] == ["job_state", "run_state"]
    assert "jobB1" not in text and ": ping" in text and text.startswith("retry: ")
    for m in messages:
        jsonschema.validate(json.loads(m["data"]), EVENT_SCHEMA)

    resumed = _parse(asyncio.run(_collect(aclient(), "A", a1, heartbeat_s=1, max_stream_s=0.3, block_ms=100)))
    assert [m["id"] for m in resumed] == [a2]

    # Without Last-Event-ID the stream starts at "now": no history replay, only new events.
    def later() -> None:
        threading.Event().wait(0.15)
        bus.publish("A", "job_state", "jobA2", {"job_id": "jobA2", "state": "running"})

    threading.Thread(target=later, daemon=True).start()
    live = _parse(asyncio.run(_collect(aclient(), "A", None, heartbeat_s=1, max_stream_s=0.6, block_ms=100)))
    assert [json.loads(m["data"])["resource_id"] for m in live] == ["jobA2"]


def test_resume_point_trimmed_announces_resync() -> None:
    server = fakeredis.FakeServer()
    sync = fakeredis.FakeRedis(server=server)
    bus = EventBus(sync)
    old = bus.publish("A", "job_state", "j", {"job_id": "j", "state": "queued"})
    newer = bus.publish("A", "job_state", "j", {"job_id": "j", "state": "running"})
    assert old and newer
    sync.xdel(stream_key("A"), old)
    text = asyncio.run(
        _collect(fakeredis.FakeAsyncRedis(server=server), "A", "1-0", heartbeat_s=1, max_stream_s=0.2, block_ms=50)
    )
    assert ": resync" in text


def test_sse_endpoint_is_owner_filtered(harness: Harness) -> None:
    a, ainfo = harness.login()
    b, binfo = harness.login()
    a_job = harness.upload(a, ainfo).json()["id"]
    b_job = harness.upload(b, binfo).json()["id"]
    response = a.get("/api/v1/events", params={"last_event_id": "0-0"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-store"
    messages = _parse(response.text)
    assert [json.loads(m["data"])["resource_id"] for m in messages] == [a_job]
    assert b_job not in response.text
    # Header form of the resume id works too, and an invalid id is a validation error.
    resumed = a.get("/api/v1/events", headers={"Last-Event-ID": messages[0]["id"]})
    assert _parse(resumed.text) == []
    assert a.get("/api/v1/events", params={"last_event_id": "abc"}).status_code == 422
    assert harness.new_client().get("/api/v1/events").status_code == 401


def test_sse_endpoint_delivers_live_events(harness: Harness) -> None:
    client, info = harness.login()
    owner = harness.services.db.execute("SELECT id FROM sessions ORDER BY created_utc DESC LIMIT 1").fetchone()[0]

    def later() -> None:
        threading.Event().wait(0.3)
        harness.services.bus.publish(owner, "error", owner, {"code": "obstacle_rejected", "message": "x"})

    threading.Thread(target=later, daemon=True).start()
    response = client.get("/api/v1/events")
    events = [m["event"] for m in _parse(response.text)]
    assert events == ["error"]


def test_sse_unavailable_without_redis(harness: Harness) -> None:
    client, _ = harness.login()
    harness.server.connected = False
    try:
        response = client.get("/api/v1/events")
    finally:
        harness.server.connected = True
    assert response.status_code == 503 and response.json()["code"] == "service_unavailable"


@pytest.mark.parametrize("event_type", ["job_state", "job_progress", "run_state", "frame_ready", "decision", "error"])
def test_every_emitted_event_type_matches_contract(event_type: str) -> None:
    payloads = {
        "job_state": {"job_id": "j", "state": "running", "error_code": None},
        "job_progress": {"job_id": "j", "progress": 0.5, "stage": "analyzing", "frames_done": 1, "frames_total": 2},
        "run_state": {"run_id": "r", "state": "stopped", "outcome": "stopped_operator"},
        "frame_ready": {"run_id": "r", "frame_id": 3, "sim_time_s": 0.15},
        "decision": {
            "run_id": "r",
            "frame_id": 3,
            "state": "follow_route",
            "reason_codes": ["ROUTE_FOLLOWING"],
            "command": {"steer": 0.1, "throttle": 0.2, "brake": 0.0},
        },
        "error": {"code": "obstacle_rejected", "message": "rejected"},
    }
    bus = EventBus(fakeredis.FakeRedis())
    assert bus.publish("o", event_type, "res", payloads[event_type])  # type: ignore[arg-type]
    envelope = json.loads(bus.redis.xrange(stream_key("o"))[0][1][b"data"])
    jsonschema.validate(envelope, EVENT_SCHEMA)
