"""Simulator-only closed-loop runs: create, observe, control, inject synthetic obstacles, camera frame and report.

Commands produced by a run act only on the simulator; nothing here talks to a physical vehicle.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import redis
from fastapi import APIRouter, Request
from fastapi.responses import Response
from pydantic import ValidationError

from app.api import catalog
from app.api.convert import decode_cursor, encode_cursor, read_live, run_to_api
from app.api.deps import CsrfSessionDep, ExpensiveRateLimit, SessionDep, SettingsDep, get_services
from app.api.params import PageCursor, PageLimit, RunIdPath
from app.api.services import Services
from app.db import utc_now
from app.errors import ApiError, error_responses
from app.jobs.queue import (
    RUN_KEY_TTL_S,
    cancel_if_queued,
    enqueue_sim_job,
    rq_sim_id,
    run_control_key,
    run_frame_key,
    run_inject_key,
    run_live_key,
)
from app.schemas.api import (
    MAX_ACTIVE_SIMULATIONS,
    MAX_OBSTACLES_PER_RUN,
    ErrorCode,
    ObstacleInject,
    ObstacleInjectResult,
    Run,
    RunControl,
    RunCreate,
    RunList,
    RunReportResponse,
)
from app.storage import artifacts, repo

router = APIRouter(prefix="/runs", tags=["runs"])
logger = logging.getLogger("dreamdrive.api.runs")

FRAME_HEADERS = {
    "X-Frame-Id": {"schema": {"type": "integer"}, "description": "Simulator frame id of this exact image."},
    "X-Sim-Time": {"schema": {"type": "number"}, "description": "Simulation time of the frame in seconds."},
    "X-Perception-Source": {"schema": {"type": "string"}, "description": "Perception source that consumed it."},
    "Cache-Control": {"schema": {"type": "string"}, "description": "no-store"},
}


def _not_found() -> ApiError:
    return ApiError(404, ErrorCode.NOT_FOUND, "Run not found")


def _owned_run(services: Services, owner_id: str, run_id: str) -> repo.RunRow:
    row = repo.get_run(services.db, owner_id, run_id)
    if row is None:
        raise _not_found()
    return row


def _api_run(services: Services, row: repo.RunRow) -> Run:
    return run_to_api(row, read_live(services.redis, row))


def _push(services: Services, key: str, value: str) -> None:
    try:
        pipe = services.redis.pipeline(transaction=True)
        pipe.rpush(key, value)
        pipe.expire(key, RUN_KEY_TTL_S)
        pipe.execute()
    except redis.RedisError as exc:
        raise ApiError(503, ErrorCode.SERVICE_UNAVAILABLE, "The control channel is unavailable") from exc


def _publish_state(services: Services, row: repo.RunRow, state: str, outcome: str | None = None) -> None:
    services.bus.publish(row.owner_id, "run_state", row.id, {"run_id": row.id, "state": state, "outcome": outcome})


@router.post(
    "",
    status_code=202,
    response_model=Run,
    summary="Start a simulation run",
    dependencies=[ExpensiveRateLimit],
    responses=error_responses(401, 403, 422, 429, 503),
)
def create_run(body: RunCreate, request: Request, session: CsrfSessionDep, settings: SettingsDep) -> Run:
    """202 with the run in `initializing`.

    422: `unknown_node`, `destination_unreachable`, `unsupported_mode`, `unsupported_scenario`.
    429: `simulation_busy`, `rate_limited`. 503: `simulator_unavailable`.
    """
    services = get_services(request)
    if body.simulator != "lite":
        raise ApiError(
            503,
            ErrorCode.SIMULATOR_UNAVAILABLE,
            "This simulator is not available",
            {"simulator": body.simulator, "reason": catalog.carla_reason(settings)},
        )
    try:
        runner = catalog.load_runner()
        nodes = catalog.node_ids()
        scenarios = catalog.scenario_availability()
    except Exception as exc:  # noqa: BLE001 - visible 503, never a fake run
        logger.exception("simulator unavailable", extra={"code": "simulator_unavailable"})
        raise ApiError(
            503,
            ErrorCode.SIMULATOR_UNAVAILABLE,
            "The simulator is not available",
            {"reason": "simulator_module_unavailable"},
        ) from exc
    for field_name, node in (("start_node", body.start_node), ("destination_node", body.destination_node)):
        if node not in nodes:
            raise ApiError(422, ErrorCode.UNKNOWN_NODE, "Unknown road graph node", {"field": field_name, "node": node})
    available, reason = scenarios.get(body.scenario, (False, "unknown_scenario"))
    if not available:
        raise ApiError(
            422,
            ErrorCode.UNSUPPORTED_SCENARIO,
            "This scenario is not supported",
            {"scenario": body.scenario, "reason": reason or "unavailable"},
        )
    if body.perception_source == "learned_model":
        learned_reason = catalog.learned_mode_reason(settings, services.model_status.get())
        if learned_reason is not None:
            raise ApiError(
                422,
                ErrorCode.UNSUPPORTED_MODE,
                "The learned-model perception mode is not available for simulation",
                {"perception_source": body.perception_source, "reason": learned_reason},
            )
    try:
        route: dict[str, Any] = runner.SimulationSession.plan_route(body.start_node, body.destination_node)
    except runner.RouteUnreachable as exc:
        raise ApiError(
            422,
            ErrorCode.DESTINATION_UNREACHABLE,
            "The destination cannot be reached from the start node",
            {"start_node": body.start_node, "destination_node": body.destination_node},
        ) from exc
    except runner.UnsupportedMode as exc:
        raise ApiError(
            422, ErrorCode.UNSUPPORTED_MODE, "Unsupported run configuration", {"reason": "unsupported"}
        ) from exc
    except (ValueError, KeyError) as exc:
        raise ApiError(422, ErrorCode.VALIDATION_ERROR, "Invalid route request") from exc

    run_id = artifacts.new_id()
    try:
        row = repo.insert_run(
            services.db,
            run_id=run_id,
            owner_id=session.session_id,
            simulator=body.simulator,
            perception_source=body.perception_source,
            scenario=body.scenario,
            seed=body.seed,
            start_node=body.start_node,
            destination_node=body.destination_node,
            route={
                "node_ids": list(route.get("node_ids", [])),
                "polyline": [[float(x), float(y)] for x, y in route.get("polyline", [])],
                "length_m": float(route.get("length_m", 0.0)),
            },
            max_active=MAX_ACTIVE_SIMULATIONS,
        )
    except repo.AdmissionDenied as exc:
        raise ApiError(429, exc.code, exc.message, exc.details, headers={"Retry-After": "10"}) from exc
    try:
        enqueue_sim_job(services.redis, settings, run_id)
    except redis.RedisError as exc:
        repo.transition_run(
            services.db,
            run_id,
            ("initializing",),
            "failed",
            outcome="failed_internal",
            error_code="queue_unavailable",
            error_message="The job queue is unavailable",
            finished_utc=utc_now(),
        )
        raise ApiError(503, ErrorCode.SERVICE_UNAVAILABLE, "The job queue is unavailable") from exc
    _publish_state(services, row, "initializing")
    logger.info("run queued", extra={"run_id": run_id, "code": "run_queued", "state": "initializing"})
    return run_to_api(row, None)


@router.get("", response_model=RunList, summary="List the caller's runs", responses=error_responses(401, 422))
def list_runs(
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
    limit: PageLimit = 20,
    cursor: PageCursor = None,
) -> RunList:
    """Newest first; lets the UI re-attach to an active run after reload."""
    services = get_services(request)
    rows = repo.list_runs(services.db, session.session_id, limit + 1, decode_cursor(cursor))
    page = rows[:limit]
    next_cursor = encode_cursor(page[-1].created_utc, page[-1].id) if len(rows) > limit else None
    return RunList(items=[_api_run(services, r) for r in page], next_cursor=next_cursor)


@router.get(
    "/{run_id}",
    response_model=Run,
    summary="Run state, ego telemetry, decision, command and metadata",
    responses=error_responses(401, 404, 422),
)
def get_run(request: Request, run_id: RunIdPath, session: SessionDep, settings: SettingsDep) -> Run:
    services = get_services(request)
    return _api_run(services, _owned_run(services, session.session_id, run_id))


@router.post(
    "/{run_id}/control",
    response_model=Run,
    summary="Pause, resume or stop (idempotent transitions)",
    responses=error_responses(401, 403, 404, 409, 422),
)
def control_run(
    request: Request, run_id: RunIdPath, body: RunControl, session: CsrfSessionDep, settings: SettingsDep
) -> Run:
    """Stop after a terminal state returns that state unchanged and never starts another run.
    409 `invalid_transition` for pause/resume on a terminal run."""
    services = get_services(request)
    row = _owned_run(services, session.session_id, run_id)
    action = body.action
    if row.terminal or row.state == "stopping":
        if action == "stop":
            return _api_run(services, row)
        raise ApiError(
            409, ErrorCode.INVALID_TRANSITION, f"Cannot {action} a run that is {row.state}", {"state": row.state}
        )
    if action == "stop":
        if row.state == "initializing":
            try:
                not_started = cancel_if_queued(services.redis, rq_sim_id(run_id))
            except redis.RedisError:
                not_started = False
            if not_started and repo.transition_run(
                services.db,
                run_id,
                ("initializing",),
                "stopped",
                outcome="stopped_operator",
                summary={"live": None, "wall_time_s": 0.0, "report_available": False},
                finished_utc=utc_now(),
            ):
                _publish_state(services, row, "stopped", "stopped_operator")
                return _api_run(services, repo.get_run_any(services.db, run_id) or row)
        _push(services, run_control_key(run_id), "stop")
        # Acknowledged immediately; the worker brakes to standstill and confirms `stopped`.
        if repo.transition_run(services.db, run_id, ("initializing", "running", "paused"), "stopping"):
            _publish_state(services, row, "stopping")
        logger.info("stop requested", extra={"run_id": run_id, "code": "stop_requested"})
    elif action == "pause":
        if row.state != "paused":
            _push(services, run_control_key(run_id), "pause")
    elif row.state == "paused":  # resume
        _push(services, run_control_key(run_id), "resume")
    return _api_run(services, repo.get_run_any(services.db, run_id) or row)


@router.post(
    "/{run_id}/obstacles",
    status_code=202,
    response_model=ObstacleInjectResult,
    summary="Inject a synthetic visual obstacle (scenario injection only)",
    responses=error_responses(401, 403, 404, 409, 422),
)
def inject_obstacle(
    request: Request, run_id: RunIdPath, body: ObstacleInject, session: CsrfSessionDep, settings: SettingsDep
) -> ObstacleInjectResult:
    """409 `run_not_active` unless running/paused; 409 `obstacle_limit_reached` after the per-run maximum."""
    services = get_services(request)
    row = _owned_run(services, session.session_id, run_id)
    active = ("running", "paused")
    if row.state not in active:
        raise ApiError(409, ErrorCode.RUN_NOT_ACTIVE, "The run is not running or paused", {"state": row.state})
    count = repo.reserve_obstacle(services.db, run_id, MAX_OBSTACLES_PER_RUN, active)
    if count is None:
        current = repo.get_run_any(services.db, run_id) or row
        if current.state not in active:
            raise ApiError(409, ErrorCode.RUN_NOT_ACTIVE, "The run is not running or paused", {"state": current.state})
        raise ApiError(
            409,
            ErrorCode.OBSTACLE_LIMIT_REACHED,
            "The obstacle limit for this run is reached",
            {"limit": MAX_OBSTACLES_PER_RUN},
        )
    payload = json.dumps({"distance_ahead_m": body.distance_ahead_m, "lateral_offset_m": body.lateral_offset_m})
    _push(services, run_inject_key(run_id), payload)
    return ObstacleInjectResult(
        run_id=run_id,
        distance_ahead_m=body.distance_ahead_m,
        lateral_offset_m=body.lateral_offset_m,
        injected_total=count,
        max_per_run=MAX_OBSTACLES_PER_RUN,
    )


@router.get(
    "/{run_id}/frame",
    response_class=Response,
    summary="Latest actual camera JPEG (no cache)",
    responses={
        200: {
            "description": "The exact RGB frame perception consumed, JPEG-encoded",
            "content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}},
            "headers": FRAME_HEADERS,
        },
        204: {"description": "No frame captured yet"},
        **error_responses(401, 404, 422),
    },
)
def get_run_frame(request: Request, run_id: RunIdPath, session: SessionDep, settings: SettingsDep) -> Response:
    services = get_services(request)
    _owned_run(services, session.session_id, run_id)
    try:
        # Written together by the worker in one MULTI; MGET reads a consistent (frame, live) pair.
        frame, live_raw = services.redis.mget([run_frame_key(run_id), run_live_key(run_id)])
    except redis.RedisError as exc:
        raise ApiError(503, ErrorCode.SERVICE_UNAVAILABLE, "Frame store unavailable") from exc
    if not frame or not live_raw:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})
    try:
        live = json.loads(live_raw)
        headers = {
            "X-Frame-Id": str(int(live["frame_id"])),
            "X-Sim-Time": f"{float(live['sim_time_s']):.3f}",
            "X-Perception-Source": str(live["perception"]["source"]),
            "Cache-Control": "no-store",
        }
    except (ValueError, KeyError, TypeError):
        return Response(status_code=204, headers={"Cache-Control": "no-store"})
    return Response(content=frame, media_type="image/jpeg", headers=headers)


@router.get(
    "/{run_id}/report",
    response_model=RunReportResponse,
    summary="Measured run report",
    responses=error_responses(401, 404, 409, 422),
)
def get_run_report(
    request: Request, run_id: RunIdPath, session: SessionDep, settings: SettingsDep
) -> RunReportResponse:
    """409 `report_unavailable` while the run is not terminal."""
    services = get_services(request)
    row = _owned_run(services, session.session_id, run_id)
    if not row.terminal:
        raise ApiError(
            409, ErrorCode.REPORT_UNAVAILABLE, "The report is available once the run has ended", {"state": row.state}
        )
    path = artifacts.run_dir(settings.artifact_root, run_id) / "report.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        reason = (row.summary or {}).get("report_error") or "not_generated"
        raise ApiError(
            409,
            ErrorCode.REPORT_UNAVAILABLE,
            "No report was generated for this run",
            {"state": row.state, "reason": reason},
        ) from exc
    except (OSError, ValueError) as exc:
        raise ApiError(
            409,
            ErrorCode.REPORT_UNAVAILABLE,
            "The stored report is unreadable",
            {"state": row.state, "reason": "unreadable"},
        ) from exc
    try:
        return RunReportResponse.model_validate(data)
    except ValidationError as exc:
        logger.error("stored run report invalid", extra={"run_id": run_id, "code": "report_invalid"})
        raise ApiError(
            409, ErrorCode.REPORT_UNAVAILABLE, "The stored report is invalid", {"state": row.state, "reason": "invalid"}
        ) from exc
