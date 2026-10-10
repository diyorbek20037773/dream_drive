"""Test harness for the backend-core API/worker tests (imported by test_api_*, test_jobs_*, test_events_* and
tests/security). Builds an isolated app per test: temporary SQLite + artifact root, fakeredis (sync + async sharing
one server), a fake inference service and a fake LiteRoad runner, all injected at module boundaries.

The fakes only stand in for *other components* (inference service, video pipeline, simulator); the API, storage,
queue, events and worker job code under test is the real code. Values such as the all-zero SHA256 below are
test fixtures, never presented as real model identities.
"""

from __future__ import annotations

import hashlib
import json
import struct
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import fakeredis
import pytest
from fastapi.testclient import TestClient

from app.api import catalog
from app.api.services import ModelStatusCache, Services
from app.config import Settings
from app.db import Database
from app.events.bus import EventBus
from app.jobs import context as job_context
from app.jobs import video_job
from app.main import create_app
from app.schemas.inference import ModelInfo
from app.security import SessionSigner
from app.simulators.base import GraphEdge, GraphNode, RoadGraph

TEST_SECRET = "test-secret-" + "x" * 40
TEST_COMMIT = "0123456789abcdef0123456789abcdef01234567"
FAKE_SHA = "0" * 64  # test-only placeholder identity of the fake model


# ---------------------------------------------------------------------------------------------------------------------
# Fake inference service
# ---------------------------------------------------------------------------------------------------------------------
class FakeInference:
    def __init__(self, ready: bool = True, reason: str | None = None, unreachable: bool = False) -> None:
        self.ready = ready
        self.reason = reason
        self.unreachable = unreachable
        self.calls = 0

    def model_info(self) -> ModelInfo:
        self.calls += 1
        if self.unreachable:
            from app.perception.client import InferenceUnavailable

            raise InferenceUnavailable("inference_unreachable", "connection refused")
        return ModelInfo(
            model_id="yolopv2",
            ready=self.ready,
            reason=None if self.ready else (self.reason or "model_missing"),
            sha256=FAKE_SHA if self.ready else None,
            class_map={"3": "vehicle"} if self.ready else {},
            device="cpu" if self.ready else None,
            precision="fp32" if self.ready else None,
        )

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------------------------------------------------
# Fake LiteRoad runner (stands in for app.agents.runner)
# ---------------------------------------------------------------------------------------------------------------------
class RouteUnreachable(ValueError):
    pass


class UnsupportedMode(ValueError):
    pass


FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"fake-jpeg-bytes" + b"\xff\xd9"


def fake_live(frame_id: int, sim_time: float, state: str, codes: list[str], speed: float, progress: float) -> dict:
    command = None
    if state not in ("observe", "paused"):
        command = {
            "steer": 0.0,
            "throttle": 0.0 if state in ("controlled_stop", "safety_stop", "arrived") else 0.3,
            "brake": 1.0 if state in ("controlled_stop", "safety_stop", "arrived") else 0.0,
            "input_frame_id": frame_id,
            "source": "dreamdrive_controller",
            "safety_override": state == "safety_stop",
            "reason_codes": codes,
        }
    return {
        "frame_id": frame_id,
        "sim_time_s": sim_time,
        "wall_time_s": sim_time,
        "realtime_factor": 1.0,
        "ego": {"source": "ego_telemetry", "x_m": progress * 10.0, "y_m": 0.0, "yaw_rad": 0.0, "speed_mps": speed},
        "decision": {"state": state, "reason_codes": codes, "action_kind": "simulator_control"},
        "command": command,
        "perception": {
            "source": "synthetic_rgb_cv",
            "quality_status": "valid",
            "lane_status": "valid",
            "image_lane_offset_normalized": 0.0,
            "image_obstacle_cue": 0.0,
            "obstacle_boxes": [],
            "lane_points": [[320.0, 300.0]],
            "frame_width": 640,
            "frame_height": 360,
        },
        "route_progress": progress,
        "acquisition_age_s": 0.001,
    }


class FakeSession:
    """Deterministic stand-in with the SimulationSession surface of docs/IMPLEMENTATION_SPEC.md §4.1."""

    ticks_to_arrive = 8
    default_wall_limit_s = 120.0
    instances: list[FakeSession] = []

    @staticmethod
    def road_graph() -> RoadGraph:
        nodes = (
            GraphNode("A", 0.0, 0.0, "Bosh darvoza", "Main gate"),
            GraphNode("B", 60.0, 0.0, "Bozor", "Market"),
            GraphNode("C", 130.0, 0.0, "Maktab", "School"),
            GraphNode("Q", 215.0, 140.0, "Yopiq depo", "Closed depot"),
        )
        edges = (
            GraphEdge("AB", "A", "B", ((0.0, 0.0), (60.0, 0.0))),
            GraphEdge("BC", "B", "C", ((60.0, 0.0), (130.0, 0.0))),
        )
        return RoadGraph(nodes=nodes, edges=edges)

    @staticmethod
    def scenarios() -> list[dict]:
        return [
            {"id": "normal", "available": True, "reason": None},
            {"id": "obstacle_stop", "available": True, "reason": None},
            {"id": "pause_resume", "available": False, "reason": "test_only_helper"},
        ]

    @staticmethod
    def plan_route(start: str, dest: str) -> dict:
        if "Q" in (start, dest):
            raise RouteUnreachable(f"{dest} unreachable from {start}")
        return {"node_ids": [start, dest], "polyline": [[0.0, 0.0], [130.0, 0.0]], "length_m": 130.0}

    def __init__(self, cfg: Any, inference_client: Any = None) -> None:
        self.cfg = cfg
        self.frame_id = 0
        self.sim_time = 0.0
        self.paused = False
        self.stop_requested = False
        self.state = "initializing"
        self.outcome: str | None = None
        self.terminal = False
        self.injected: list[tuple[float, float]] = []
        self.wall_limit: float | None = None
        self.ignore_wall_limit = False
        self.pause_calls = 0
        self.resume_calls = 0
        FakeSession.instances.append(self)

    def _result(self, codes: list[str], decision: str) -> SimpleNamespace:
        progress = min(1.0, self.frame_id / max(1, self.ticks_to_arrive))
        speed = 0.0 if self.terminal or self.paused else 2.0
        return SimpleNamespace(
            frame_id=self.frame_id,
            sim_time_s=self.sim_time,
            jpeg=FAKE_JPEG + struct.pack(">I", self.frame_id),
            world_state=None,
            live=fake_live(self.frame_id, self.sim_time, decision, codes, speed, progress),
            run_state=self.state,
            outcome=self.outcome,
            terminal=self.terminal,
        )

    def start(self) -> SimpleNamespace:
        self.state = "running"
        return self._result(["INITIALIZING"], "observe")

    def pause(self) -> None:
        self.pause_calls += 1
        if not self.terminal and not self.stop_requested:
            self.paused = True
            self.state = "paused"

    def resume(self) -> None:
        self.resume_calls += 1
        if self.paused:
            self.paused = False
            self.state = "running"

    def request_stop(self) -> None:
        if not self.terminal:
            self.stop_requested = True
            self.paused = False
            self.state = "stopping"

    def inject_obstacle(self, distance_ahead_m: float, lateral_offset_m: float) -> bool:
        self.injected.append((distance_ahead_m, lateral_offset_m))
        return True

    def note_wall_time(self, wall_elapsed_s: float) -> None:
        limit = self.wall_limit if self.wall_limit is not None else FakeSession.default_wall_limit_s
        if not self.ignore_wall_limit and wall_elapsed_s > limit and not self.terminal:
            self.state, self.outcome, self.terminal = "failed", "failed_wall_time", True

    def tick(self) -> SimpleNamespace:
        if self.terminal:
            return self._result(["WALL_TIME_LIMIT"], "failed")
        if self.paused:
            return self._result(["OPERATOR_PAUSE"], "paused")
        self.frame_id += 1
        self.sim_time = round(self.frame_id * self.cfg.dt, 6)
        if self.stop_requested:
            self.state, self.outcome, self.terminal = "stopped", "stopped_operator", True
            return self._result(["OPERATOR_STOP"], "controlled_stop")
        if self.frame_id >= self.ticks_to_arrive:
            self.state, self.outcome, self.terminal = "arrived", "arrived", True
            return self._result(["ARRIVED_AT_DESTINATION"], "arrived")
        return self._result(["ROUTE_FOLLOWING"], "follow_route")

    def report(self, code_commit: str, wall_time_s: float) -> dict:
        return {
            "schema_version": "1.0",
            "run_id": self.cfg.run_id,
            "kind": "simulation",
            "status": self.state if self.state in ("stopped", "arrived") else "failed",
            "perception_source": "synthetic_rgb_cv",
            "code_commit": code_commit,
            "model_sha256": None,
            "input_manifest_sha256": None,
            "generated_utc": "2026-10-10T12:00:00+00:00",
            "metrics": {
                "ticks": {
                    "value": float(self.frame_id),
                    "unit": "ticks",
                    "sample_count": 1,
                    "status": "measured",
                    "note": "fake session",
                },
            },
            "limitations": ["fake session used by backend-core unit tests"],
            "evaluation_truth_used": True,
            "runtime_used_dynamic_ground_truth": False,
            "details": {"wall_time_s": wall_time_s},
        }


def fake_runner_module() -> SimpleNamespace:
    return SimpleNamespace(
        SimulationSession=FakeSession, RouteUnreachable=RouteUnreachable, UnsupportedMode=UnsupportedMode
    )


# ---------------------------------------------------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------------------------------------------------
def mp4_bytes(size: int = 4096, brand: bytes = b"isom") -> bytes:
    """Bytes that start with a valid ISO-BMFF ftyp box (enough for the upload sniff; not a decodable video)."""
    ftyp = struct.pack(">I", 24) + b"ftyp" + brand + b"\x00\x00\x02\x00" + brand + b"iso2"
    body = ftyp + struct.pack(">I", 8) + b"free"
    return body + bytes((i * 7) % 251 for i in range(max(0, size - len(body))))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Harness:
    settings: Settings
    services: Services
    client: TestClient
    redis: fakeredis.FakeRedis
    server: fakeredis.FakeServer
    inference: FakeInference
    tmp: Path
    sessions: dict[str, dict[str, str]] = field(default_factory=dict)

    # -- sessions --------------------------------------------------------------------------------------------------
    def new_client(self) -> TestClient:
        return TestClient(self.client.app, raise_server_exceptions=False)

    def login(self, client: TestClient | None = None, password: str | None = None) -> tuple[TestClient, dict[str, Any]]:
        client = client or self.new_client()
        body = {"admin_password": password} if password is not None else None
        response = client.post("/api/v1/session", json=body)
        assert response.status_code == 200, response.text
        return client, response.json()

    @staticmethod
    def csrf(info: dict[str, Any]) -> dict[str, str]:
        return {"X-CSRF-Token": info["csrf_token"]}

    def upload(
        self, client: TestClient, info: dict[str, Any], data: bytes | None = None, filename: str = "clip.mp4"
    ) -> Any:
        payload = mp4_bytes() if data is None else data
        return client.post(
            "/api/v1/videos",
            files={"file": (filename, payload, "video/mp4")},
            headers=self.csrf(info),
        )

    # -- worker side ---------------------------------------------------------------------------------------------------
    def worker_context(self) -> job_context.WorkerContext:
        return job_context.WorkerContext(
            settings=self.settings, db=self.services.db, redis=self.redis, bus=EventBus(self.redis)
        )

    def events(self, owner_id: str) -> list[dict[str, Any]]:
        entries = self.redis.xrange(f"dd:events:{owner_id}")
        return [json.loads(fields[b"data"]) for _id, fields in entries]

    def wait_for(self, predicate: Callable[[], bool], timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        raise AssertionError("condition not reached in time")


def make_settings(tmp: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "app_env": "test",
        "app_secret": TEST_SECRET,
        "sqlite_path": tmp / "dreamdrive.db",
        "data_root": tmp,
        "artifact_root": tmp / "artifacts",
        "redis_url": "redis://fake:6379/0",
        "git_commit": TEST_COMMIT,
        "max_queued_per_session": 5,
        "max_queued_total": 10,
        "max_requests_per_minute": 50,
        "sim_fixed_dt_seconds": 0.01,
        "log_level": "INFO",
    }
    values.update(overrides)
    return Settings(**values)


def build_harness(tmp: Path, monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> Iterator[Harness]:
    settings = make_settings(tmp, **overrides)
    server = fakeredis.FakeServer()
    redis_client = fakeredis.FakeRedis(server=server)
    inference = FakeInference()
    db = Database(settings.sqlite_path)
    services = Services(
        settings=settings,
        db=db,
        redis=redis_client,
        async_redis_factory=lambda: fakeredis.FakeAsyncRedis(server=server),
        model_status=ModelStatusCache(lambda: inference),
        signer=SessionSigner(settings.app_secret),
    )
    services.sse_heartbeat_s = 0.3
    services.sse_max_stream_s = 1.0
    # Collaborators owned by other components, replaced at their module boundary.
    monkeypatch.setattr(catalog, "load_runner", fake_runner_module)
    catalog.reset_cache()
    FakeSession.instances = []
    FakeSession.ticks_to_arrive = 8
    FakeSession.default_wall_limit_s = settings.max_simulation_wall_seconds
    monkeypatch.setattr(video_job, "make_inference_client", lambda ctx: inference)
    app = create_app(settings, services=services)
    client = TestClient(app, raise_server_exceptions=False)
    harness = Harness(settings, services, client, redis_client, server, inference, tmp)
    job_context.set_context(harness.worker_context())
    with client:
        yield harness
    job_context.set_context(None)
    job_context.SHUTDOWN.clear()
    catalog.reset_cache()
    db.close()


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    yield from build_harness(tmp_path, monkeypatch)


def run_in_thread(target: Callable[[], Any]) -> threading.Thread:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------------------------------------------------
# Harness self-checks
# ---------------------------------------------------------------------------------------------------------------------
def test_harness_session_and_isolated_storage(harness: Harness) -> None:
    client, info = harness.login()
    assert info["role"] == "anonymous" and len(info["csrf_token"]) >= 32
    assert harness.settings.sqlite_path.exists()
    assert str(harness.settings.sqlite_path).startswith(str(harness.tmp))


def test_mp4_fixture_bytes_are_sniffed_as_mp4() -> None:
    from app.api.videos import sniff_container

    assert sniff_container(mp4_bytes()[:64]) == "mp4"
    assert sniff_container(mp4_bytes(brand=b"qt  ")[:64]) == "mov"
