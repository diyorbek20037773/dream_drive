"""HTTP contract tests: every documented endpoint exists, the committed OpenAPI/TypeScript artifacts are current,
errors always use the ErrorEnvelope, and the state enums match docs/API.md.

Requires docs/API.md and frontend/src/api/{openapi.json,schema.d.ts} next to the backend (repo checkout, or the
same relative layout under /app in the backend test image). Missing files fail; they are never skipped.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, get_args

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

import app.schemas.world_state as ws
from app.config import Settings
from app.errors import ApiError, install_error_handling
from app.export_openapi import main as export_main
from app.export_openapi import openapi_document, render
from app.main import create_app
from app.schemas import api as api_schemas
from app.schemas.reason_codes import ReasonCode
from app.simulators.base import GraphEdge, GraphNode, RoadGraph

ROOT = Path(__file__).resolve().parents[3]
API_DOC = ROOT / "docs" / "API.md"
COMMITTED_OPENAPI = ROOT / "frontend" / "src" / "api" / "openapi.json"
COMMITTED_TS = ROOT / "frontend" / "src" / "api" / "schema.d.ts"

DOC_JOB_STATES = ["queued", "validating", "running", "cancelling", "cancelled", "succeeded", "failed", "interrupted"]
DOC_RUN_STATES = ["initializing", "running", "paused", "stopping", "stopped", "arrived", "failed"]

# Endpoints beyond the docs/API.md table: the first three are required by docs/IMPLEMENTATION_SPEC.md §3.4;
# GET /api/v1/runs lets the UI re-attach to an active run after a reload (contract addition).
SPEC_EXTRA_ENDPOINTS = {
    ("POST", "/api/v1/evaluations"),
    ("GET", "/api/v1/evaluations/{}"),
    ("POST", "/api/v1/admin/cleanup"),
    ("GET", "/api/v1/runs"),
}
CREATE_202 = {("POST", "/api/v1/videos"), ("POST", "/api/v1/runs"), ("POST", "/api/v1/evaluations")}
ENVELOPE_REF = "#/components/schemas/ErrorEnvelope"
ENVELOPE_KEYS = {"code", "message", "request_id", "details"}
_TABLE_ROW = re.compile(r"^\|\s*(GET|POST|PUT|PATCH|DELETE)\s+(/\S*)\s*\|")


def _norm(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "{}", path)


def _require(path: Path) -> Path:
    assert path.is_file(), f"required contract file missing: {path} (copy it into the test image; never skip)"
    return path


def _doc_endpoints() -> set[tuple[str, str]]:
    rows = set()
    for line in _require(API_DOC).read_text(encoding="utf-8").splitlines():
        match = _TABLE_ROW.match(line)
        if match:
            rows.add((match.group(1), _norm(match.group(2))))
    return rows


def _operations(doc: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(method.upper(), _norm(path)): op for path, ops in doc["paths"].items() for method, op in ops.items()}


def _ref_name(schema: dict[str, Any]) -> str:
    return schema["$ref"].rsplit("/", 1)[-1]


@pytest.fixture(scope="module")
def app() -> FastAPI:
    return create_app()


@pytest.fixture(scope="module")
def openapi(app: FastAPI) -> dict[str, Any]:
    return app.openapi()


@pytest.fixture(scope="module")
def client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------------------------------------------------
# Endpoint inventory
# ---------------------------------------------------------------------------------------------------------------------
def test_api_doc_table_is_parsed() -> None:
    endpoints = _doc_endpoints()
    # The table in docs/API.md lists 21 endpoints; a parse regression must not silently pass the next test.
    assert len(endpoints) >= 21, sorted(endpoints)
    assert ("GET", "/metrics") in endpoints


def test_every_documented_endpoint_exists_with_method(app: FastAPI, openapi: dict[str, Any]) -> None:
    operations = _operations(openapi)
    for method, path in sorted(_doc_endpoints() | SPEC_EXTRA_ENDPOINTS):
        if path == "/metrics":
            # Internal Prometheus route: served by the app, absent from the public OpenAPI.
            routes = [r for r in app.routes if getattr(r, "path", None) == "/metrics"]
            assert routes and "GET" in getattr(routes[0], "methods", set()), "GET /metrics route missing"
            assert "/metrics" not in openapi["paths"]
            continue
        assert (method, path) in operations, f"{method} {path} missing from the OpenAPI"


def test_no_undocumented_public_endpoints(openapi: dict[str, Any]) -> None:
    documented = _doc_endpoints() | SPEC_EXTRA_ENDPOINTS
    extra = set(_operations(openapi)) - documented
    assert not extra, f"endpoints missing from docs/API.md or the spec list: {sorted(extra)}"


def test_create_endpoints_answer_202(openapi: dict[str, Any]) -> None:
    operations = _operations(openapi)
    for key in CREATE_202:
        responses = operations[key]["responses"]
        assert "202" in responses and "200" not in responses, key


def test_operation_ids_unique_and_tagged(openapi: dict[str, Any]) -> None:
    ops = list(_operations(openapi).values())
    ids = [op["operationId"] for op in ops]
    assert len(ids) == len(set(ids))
    assert all(op.get("tags") for op in ops)


def test_mutations_declare_csrf_header_and_session_cookie(openapi: dict[str, Any]) -> None:
    for (method, path), op in _operations(openapi).items():
        if not path.startswith("/api/v1/") or path in {"/api/v1/session", "/api/v1/capabilities"}:
            continue
        assert op.get("security") == [{"SessionCookie": []}], f"{method} {path} lacks the session cookie scheme"
        if method in {"POST", "DELETE"}:
            headers = {p["name"] for p in op.get("parameters", []) if p["in"] == "header"}
            assert "X-CSRF-Token" in headers, f"{method} {path} lacks X-CSRF-Token"
    scheme = openapi["components"]["securitySchemes"]["SessionCookie"]
    assert scheme == {**scheme, "type": "apiKey", "in": "cookie", "name": "dd_session"}


def test_upload_is_multipart_with_file_field(openapi: dict[str, Any]) -> None:
    body = openapi["paths"]["/api/v1/videos"]["post"]["requestBody"]
    assert body["required"] is True
    schema = openapi["components"]["schemas"][_ref_name(body["content"]["multipart/form-data"]["schema"])]
    assert schema["required"] == ["file"]
    file_schema = schema["properties"]["file"]
    assert file_schema.get("format") == "binary" or file_schema.get("contentMediaType") == "application/octet-stream"


def test_sse_frame_and_artifact_media_types(openapi: dict[str, Any]) -> None:
    events = openapi["paths"]["/api/v1/events"]["get"]
    assert events["responses"]["200"]["content"] == {
        "text/event-stream": {"schema": {"$ref": "#/components/schemas/EventEnvelope"}}
    }
    assert {"Last-Event-ID", "last_event_id"} <= {p["name"] for p in events["parameters"]}

    frame = openapi["paths"]["/api/v1/runs/{run_id}/frame"]["get"]["responses"]
    assert set(frame["200"]["content"]) == {"image/jpeg"}
    assert {"X-Frame-Id", "X-Sim-Time", "X-Perception-Source", "Cache-Control"} <= set(frame["200"]["headers"])
    assert "204" in frame

    artifact = openapi["paths"]["/api/v1/jobs/{job_id}/artifacts/{name}"]["get"]
    assert {"200", "206", "416"} <= set(artifact["responses"])
    assert "video/mp4" in artifact["responses"]["206"]["content"]
    name_param = next(p for p in artifact["parameters"] if p["name"] == "name")
    name_schema = name_param["schema"]
    if "$ref" in name_schema:
        name_schema = openapi["components"]["schemas"][_ref_name(name_schema)]
    assert name_schema["enum"] == [
        "input",
        "annotated.mp4",
        "timeline.json",
        "report.json",
        "report.html",
        "poster.jpg",
    ]


def test_error_responses_use_the_envelope(openapi: dict[str, Any]) -> None:
    checked = 0
    for (method, path), op in _operations(openapi).items():
        for status, response in op["responses"].items():
            if int(status) < 400:
                continue
            if (method, path, status) == ("GET", "/health/ready", "503"):
                assert response["content"]["application/json"]["schema"]["$ref"].endswith("/HealthStatus")
                continue
            assert response["content"] == {"application/json": {"schema": {"$ref": ENVELOPE_REF}}}, (
                method,
                path,
                status,
            )
            checked += 1
    assert checked > 50
    assert "HTTPValidationError" not in openapi["components"]["schemas"]


def test_documented_error_statuses_present(openapi: dict[str, Any]) -> None:
    ops = _operations(openapi)
    assert {"401", "403", "413", "415", "422", "429", "503"} <= set(ops[("POST", "/api/v1/videos")]["responses"])
    assert {"401", "403", "422", "429", "503"} <= set(ops[("POST", "/api/v1/runs")]["responses"])
    assert {"404", "409"} <= set(ops[("GET", "/api/v1/runs/{}/report")]["responses"])
    assert {"409"} <= set(ops[("POST", "/api/v1/runs/{}/obstacles")]["responses"])
    assert {"401", "403"} <= set(ops[("POST", "/api/v1/admin/cleanup")]["responses"])
    for key in [k for k in ops if "{}" in k[1]]:
        assert "404" in ops[key]["responses"], f"{key} must document 404 for foreign/unknown ids"


# ---------------------------------------------------------------------------------------------------------------------
# Schemas and enums
# ---------------------------------------------------------------------------------------------------------------------
def test_error_envelope_schema_shape(openapi: dict[str, Any]) -> None:
    schema = openapi["components"]["schemas"]["ErrorEnvelope"]
    assert set(schema["properties"]) == ENVELOPE_KEYS
    assert set(schema["required"]) == ENVELOPE_KEYS
    codes = openapi["components"]["schemas"]["ErrorCode"]["enum"]
    for code in (
        "session_required",
        "csrf_failed",
        "rate_limited",
        "session_queue_full",
        "queue_full",
        "simulation_busy",
        "not_found",
        "destination_unreachable",
        "report_unavailable",
        "model_not_ready",
        "validation_error",
        "internal_error",
    ):
        assert code in codes


def test_job_and_run_state_enums(openapi: dict[str, Any]) -> None:
    schemas = openapi["components"]["schemas"]
    assert schemas["JobState"]["enum"] == DOC_JOB_STATES
    assert schemas["RunState"]["enum"] == DOC_RUN_STATES
    assert _ref_name(schemas["Job"]["properties"]["state"]) == "JobState"
    assert _ref_name(schemas["JobResults"]["properties"]["state"]) == "JobState"
    assert _ref_name(schemas["Run"]["properties"]["state"]) == "RunState"
    assert list(api_schemas.JOB_STATES) == DOC_JOB_STATES
    assert list(api_schemas.RUN_STATES) == DOC_RUN_STATES
    assert api_schemas.JOB_TERMINAL_STATES == {"cancelled", "succeeded", "failed", "interrupted"}
    assert api_schemas.RUN_TERMINAL_STATES == {"stopped", "arrived", "failed"}


def test_reason_codes_and_world_state_literals_exposed(openapi: dict[str, Any]) -> None:
    schemas = openapi["components"]["schemas"]
    assert schemas["ReasonCode"]["enum"] == [c.value for c in ReasonCode]
    assert schemas["DecisionState"]["enum"] == list(get_args(ws.DecisionState))
    assert schemas["PerceptionSource"]["enum"] == list(get_args(ws.PerceptionSource))
    assert schemas["ActionKind"]["enum"] == list(get_args(ws.ActionKind))
    assert schemas["QualityStatus"]["enum"] == list(get_args(ws.QualityStatus))
    assert schemas["LaneStatus"]["enum"] == list(get_args(ws.LaneStatus))
    live = schemas["LiveDecision"]["properties"]
    assert _ref_name(live["reason_codes"]["items"]) == "ReasonCode"
    command = schemas["LiveCommand"]["properties"]
    assert command["source"]["const"] == "dreamdrive_controller"
    assert schemas["LiveEgo"]["properties"]["source"]["const"] == "ego_telemetry"


def test_response_fields_are_required_for_generated_types(openapi: dict[str, Any]) -> None:
    schemas = openapi["components"]["schemas"]
    for name in ("Job", "Run", "RunLive", "JobResults", "Capabilities", "SessionInfo", "Evaluation", "Simulator"):
        assert set(schemas[name]["required"]) == set(schemas[name]["properties"]), name


def test_timestamps_are_utc_date_times(openapi: dict[str, Any]) -> None:
    schemas = openapi["components"]["schemas"]
    for name, schema in schemas.items():
        for field, prop in schema.get("properties", {}).items():
            if field.endswith("_utc"):
                variants = prop.get("anyOf", [prop])
                assert any(v.get("format") == "date-time" for v in variants), f"{name}.{field}"


# ---------------------------------------------------------------------------------------------------------------------
# Generated artifacts are current
# ---------------------------------------------------------------------------------------------------------------------
def test_export_is_deterministic_sorted_json() -> None:
    first = render(openapi_document())
    assert first == render(openapi_document())
    assert first.endswith("\n") and "\r" not in first
    assert json.dumps(json.loads(first), sort_keys=True, indent=2, ensure_ascii=False) + "\n" == first


def test_committed_openapi_matches_export() -> None:
    committed_text = _require(COMMITTED_OPENAPI).read_text(encoding="utf-8").replace("\r\n", "\n")
    fresh_text = render(openapi_document())
    assert json.loads(committed_text) == json.loads(fresh_text), (
        "frontend/src/api/openapi.json is stale: run scripts/gen-types.sh and commit the result"
    )
    assert committed_text == fresh_text, "openapi.json is not in canonical form: run scripts/gen-types.sh"


def test_export_check_mode_detects_drift(tmp_path: Path) -> None:
    current = tmp_path / "openapi.json"
    assert export_main(["--output", str(current)]) == 0
    assert export_main(["--check", str(current)]) == 0
    stale = json.loads(current.read_text(encoding="utf-8"))
    del stale["paths"]["/api/v1/runs"]
    current.write_text(json.dumps(stale), encoding="utf-8")
    assert export_main(["--check", str(current)]) == 1
    assert export_main(["--check", str(tmp_path / "missing.json")]) == 1


def test_generated_typescript_covers_committed_openapi() -> None:
    # A proxy for TypeScript freshness (the full check, scripts/gen-types.sh --check, needs Node).
    ts = _require(COMMITTED_TS).read_text(encoding="utf-8")
    committed = json.loads(_require(COMMITTED_OPENAPI).read_text(encoding="utf-8"))
    assert ts.startswith("/**\n * This file was auto-generated by openapi-typescript.")
    for path in committed["paths"]:
        assert f'"{path}": {{' in ts, f"schema.d.ts lacks {path}; run scripts/gen-types.sh"
    for name in committed["components"]["schemas"]:
        assert re.search(rf"^\s+{re.escape(name)}: ", ts, re.MULTILINE), f"schema.d.ts lacks {name}"


# ---------------------------------------------------------------------------------------------------------------------
# Runtime error handling
# ---------------------------------------------------------------------------------------------------------------------
class _Item(BaseModel):
    count: int


def _error_app() -> FastAPI:
    test_app = FastAPI()
    install_error_handling(test_app)

    @test_app.get("/conflict")
    def conflict() -> None:
        raise ApiError(
            409,
            api_schemas.ErrorCode.REPORT_UNAVAILABLE,
            "Run still running",
            {"state": "running"},
            headers={"Retry-After": "5"},
        )

    @test_app.post("/items")
    def create_item(item: _Item) -> _Item:
        return item

    @test_app.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret detail at /data/artifacts/jobs/x")

    return test_app


def _assert_envelope(response: Any, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert set(body) == ENVELOPE_KEYS
    assert body["code"] == code
    assert body["request_id"] and response.headers["x-request-id"] == body["request_id"]
    assert isinstance(body["details"], dict)
    return body


def test_api_error_envelope_with_details_and_headers() -> None:
    response = TestClient(_error_app()).get("/conflict")
    body = _assert_envelope(response, 409, "report_unavailable")
    assert body["details"] == {"state": "running"} and body["message"] == "Run still running"
    assert response.headers["retry-after"] == "5"


def test_validation_error_envelope_does_not_echo_input() -> None:
    response = TestClient(_error_app()).post("/items", json={"count": "not-a-number-SECRET"})
    body = _assert_envelope(response, 422, "validation_error")
    errors = body["details"]["errors"]
    assert errors and all(set(e) == {"loc", "msg", "type"} for e in errors)
    assert "SECRET" not in response.text


def test_unhandled_exception_hides_internals() -> None:
    response = TestClient(_error_app(), raise_server_exceptions=False).get("/boom")
    body = _assert_envelope(response, 500, "internal_error")
    assert body["message"] == "Internal server error"
    for leaked in ("secret", "/data", "Traceback", "RuntimeError"):
        assert leaked not in response.text


def test_unknown_route_and_wrong_method_use_envelope(client: TestClient) -> None:
    _assert_envelope(client.get("/api/v1/does-not-exist"), 404, "not_found")
    _assert_envelope(client.delete("/api/v1/capabilities"), 405, "method_not_allowed")


def test_request_id_propagation(client: TestClient) -> None:
    good = "gateway-req-0001"
    assert client.get("/health/live", headers={"X-Request-ID": good}).headers["x-request-id"] == good
    for bad in ("short", "has space in it", "x" * 65, "evilévalue1"):
        rid = client.get("/health/live", headers={"X-Request-ID": bad.encode("utf-8")}).headers["x-request-id"]
        assert rid != bad and re.fullmatch(r"[a-f0-9]{32}", rid)


def test_health_live_and_metrics(client: TestClient) -> None:
    live = client.get("/health/live")
    assert live.status_code == 200 and live.json()["status"] == "ok"
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.headers["content-type"].startswith("text/plain")


def test_app_creation_opens_no_connections(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "never-created.db"
    settings = Settings(app_env="test", sqlite_path=sqlite_path, redis_url="redis://127.0.0.1:1/0", data_root=tmp_path)
    created = create_app(settings)
    created.openapi()
    assert not sqlite_path.exists()


def test_docs_hidden_in_production() -> None:
    dev = TestClient(create_app(Settings(app_env="test")))
    assert dev.get("/api/v1/openapi.json").status_code == 200
    prod = TestClient(create_app(Settings(app_env="production", app_secret="x" * 48)))
    _assert_envelope(prod.get("/api/v1/openapi.json"), 404, "not_found")
    _assert_envelope(prod.get("/api/v1/docs"), 404, "not_found")


# ---------------------------------------------------------------------------------------------------------------------
# Request model validation
# ---------------------------------------------------------------------------------------------------------------------
def test_run_create_validation() -> None:
    ok = api_schemas.RunCreate(start_node="n1", destination_node="n5")
    assert (ok.simulator, ok.perception_source, ok.scenario, ok.seed) == ("lite", "synthetic_rgb_cv", "normal", 42)
    bad_inputs: list[dict[str, Any]] = [
        {"start_node": "n1", "destination_node": "n5", "seed": 1_000_001},
        {"start_node": "n1", "destination_node": "n5", "seed": -1},
        {"start_node": "n1", "destination_node": "n5", "scenario": "teleport"},
        {"start_node": "n1", "destination_node": "n5", "perception_source": "simulator_truth"},
        {"start_node": "n1", "destination_node": "n5", "ground_truth": True},
        {"start_node": "", "destination_node": "n5"},
    ]
    for data in bad_inputs:
        with pytest.raises(ValidationError):
            api_schemas.RunCreate.model_validate(data)


def test_obstacle_and_control_bounds() -> None:
    api_schemas.ObstacleInject(distance_ahead_m=8, lateral_offset_m=-1.5)
    api_schemas.ObstacleInject(distance_ahead_m=40, lateral_offset_m=1.5)
    for distance, lateral in ((7.9, 0), (40.1, 0), (20, 1.6), (20, -1.6)):
        with pytest.raises(ValidationError):
            api_schemas.ObstacleInject(distance_ahead_m=distance, lateral_offset_m=lateral)
    for action in ("pause", "resume", "stop"):
        assert api_schemas.RunControl.model_validate({"action": action}).action == action
    with pytest.raises(ValidationError):
        api_schemas.RunControl.model_validate({"action": "reverse"})


def test_evaluation_create_validation() -> None:
    assert api_schemas.EvaluationCreate().seeds == [1, 2, 3]
    for data in ({"seeds": [1, 1]}, {"seeds": []}, {"kind": "model_benchmark"}, {"seeds": list(range(21))}):
        with pytest.raises(ValidationError):
            api_schemas.EvaluationCreate.model_validate(data)


def test_utc_datetimes_are_aware_and_normalised() -> None:
    tashkent = timezone(timedelta(hours=5))
    health = api_schemas.HealthStatus(status="ok", checked_utc=datetime(2026, 10, 10, 17, 0, tzinfo=tashkent))
    assert health.checked_utc == datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
    assert health.checked_utc.utcoffset() == timedelta(0)
    with pytest.raises(ValidationError):
        api_schemas.HealthStatus(status="ok", checked_utc=datetime(2026, 10, 10, 12, 0))  # naive


def test_live_decision_rejects_unknown_reason_codes() -> None:
    api_schemas.LiveDecision(state="follow_route", reason_codes=["ROUTE_FOLLOWING"], action_kind="simulator_control")
    with pytest.raises(ValidationError):
        api_schemas.LiveDecision(state="follow_route", reason_codes=["FREE_TEXT"], action_kind="simulator_control")
    with pytest.raises(ValidationError):
        api_schemas.LiveDecision(state="follow_route", reason_codes=[], action_kind="simulator_control")


def test_road_graph_wire_shape_uses_from_key() -> None:
    graph = RoadGraph(
        nodes=(GraphNode("a", 0.0, 0.0, "A", "A"), GraphNode("b", 10.0, 0.0, "B", "B")),
        edges=(GraphEdge("e1", "a", "b", ((0.0, 0.0), (10.0, 0.0))),),
    )
    info = api_schemas.RoadGraphInfo.from_graph(graph, {"a", "b"})
    dumped = info.model_dump(mode="json", by_alias=True)
    assert dumped["edges"] == [{"id": "e1", "from": "a", "to": "b", "polyline": [[0.0, 0.0], [10.0, 0.0]]}]
    assert dumped["units"] == "m" and dumped["frame"] == "ENU x-east y-north yaw CCW"
    assert all(n["reachable_from_default"] for n in dumped["nodes"])


def test_limits_from_settings() -> None:
    limits = api_schemas.Limits.from_settings(Settings(app_env="test"))
    assert limits.max_upload_bytes == 104_857_600
    assert limits.max_video_duration_s == 120
    assert (limits.max_video_width_px, limits.max_video_height_px) == (1920, 1080)
    assert (limits.max_queued_per_session, limits.max_queued_total) == (2, 10)
    assert limits.max_active_simulations == 1 and limits.max_obstacles_per_run == 5
    assert limits.accepted_extensions == ["mp4", "mov"]
