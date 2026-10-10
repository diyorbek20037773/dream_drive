"""SSE ``decision`` events from the simulation job carry the compact maneuver and the frame sim time, are sent when
the maneuver changes, validate against the event schema, and stay valid for producers without a maneuver."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import fakeredis

from app.events.bus import EventBus, stream_key
from app.jobs import sim_job
from app.schemas.events import EventEnvelope
from app.storage.repo import RunRow


def _row() -> RunRow:
    now = datetime.now(UTC)
    return RunRow(
        id="run1",
        owner_id="owner1",
        simulator="lite",
        perception_source="synthetic_rgb_cv",
        scenario="obstacle_stop",
        seed=1,
        start_node="A",
        destination_node="C",
        route={},
        state="running",
        outcome=None,
        error_code=None,
        error_message=None,
        summary=None,
        obstacles_injected=0,
        created_utc=now,
        started_utc=now,
        finished_utc=None,
        updated_utc=now,
    )


def _tick(frame_id: int, primary: str | None, state: str = "follow_route") -> Any:
    live: dict[str, Any] = {
        "frame_id": frame_id,
        "sim_time_s": frame_id * 0.05,
        "decision": {"state": state, "reason_codes": ["ROUTE_FOLLOWING"]},
        "command": {"steer": 0.1, "throttle": 0.5, "brake": 0.0, "safety_override": False, "reason_codes": ["X"]},
        "perception": {"source": "synthetic_rgb_cv", "quality_status": "valid"},
    }
    if primary is not None:
        live["maneuver"] = {
            "primary": primary,
            "longitudinal": "accelerating",
            "lateral": "straight",
            "since_sim_time_s": frame_id * 0.05,
            "input_frame_id": frame_id,
        }
    return SimpleNamespace(
        live=live, jpeg=b"", run_state="running", terminal=False, frame_id=frame_id, sim_time_s=frame_id * 0.05
    )


def _decisions(client: fakeredis.FakeRedis) -> list[dict[str, Any]]:
    out = []
    for _id, fields in client.xrange(stream_key("owner1")):
        envelope = EventEnvelope.model_validate_json(fields[b"data"])
        if envelope.type == "decision":
            out.append(json.loads(fields[b"data"])["payload"])
    return out


def test_decision_event_carries_maneuver_and_fires_on_maneuver_change(monkeypatch: Any) -> None:
    client = fakeredis.FakeRedis()
    ctx = SimpleNamespace(redis=client, bus=EventBus(client), settings=None, db=None)
    pub = sim_job._Publisher(ctx, _row())  # type: ignore[arg-type]  # noqa: SLF001
    pub.last_state = "running"
    clock = [100.0]
    monkeypatch.setattr(sim_job.time, "monotonic", lambda: clock[0])

    pub.publish_tick(_tick(1, "ACCELERATE"))
    clock[0] += 0.25
    pub.publish_tick(_tick(6, "ACCELERATE"))  # unchanged decision + maneuver: not repeated within 1 s
    clock[0] += 0.25
    pub.publish_tick(_tick(11, "SLOW_DOWN"))  # same FSM decision, new maneuver: published (> 0.2 s since last)
    events = _decisions(client)
    assert [e["frame_id"] for e in events] == [1, 11]
    assert events[0]["maneuver"] == {
        "primary": "ACCELERATE",
        "longitudinal": "accelerating",
        "lateral": "straight",
        "since_sim_time_s": 0.05,
    }
    assert events[1]["maneuver"]["primary"] == "SLOW_DOWN" and events[1]["sim_time_s"] == 0.55
    assert events[1]["state"] == "follow_route" and events[1]["command"]["throttle"] == 0.5


def test_decision_event_without_maneuver_stays_valid(monkeypatch: Any) -> None:
    client = fakeredis.FakeRedis()
    ctx = SimpleNamespace(redis=client, bus=EventBus(client), settings=None, db=None)
    pub = sim_job._Publisher(ctx, _row())  # type: ignore[arg-type]  # noqa: SLF001
    pub.last_state = "running"
    monkeypatch.setattr(sim_job.time, "monotonic", lambda: 50.0)
    pub.publish_tick(_tick(3, None))
    events = _decisions(client)
    assert len(events) == 1 and events[0]["maneuver"] is None and events[0]["frame_id"] == 3
