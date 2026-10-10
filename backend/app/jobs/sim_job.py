"""RQ entry point for one simulator-only closed-loop run: ``run_simulation_job(run_id)``.

Drives ``app.agents.runner.SimulationSession``: ``start()`` then ``tick()`` until terminal. Per tick the job
- applies pending operator controls (``dd:run:{id}:control`` list) and synthetic obstacle injections
  (``dd:run:{id}:inject`` list), both written by the API;
- paces the loop so simulated time never runs ahead of active wall time (pause time excluded);
- publishes the exact JPEG perception consumed and the ``Run.live`` JSON to Redis atomically (MULTI) with a TTL, so
  the frame endpoint's headers always describe that very image;
- mirrors run-state changes into SQLite and emits ``run_state``, ``frame_ready`` (<= 5 Hz) and ``decision`` events
  (decision or maneuver changes, <= 5 Hz; the payload carries the compact ``maneuver`` and the frame sim time).

A wall-clock watchdog (MAX_SIMULATION_WALL_SECONDS) ends the run through the session; a hard fallback fails the run if
the session does not terminate shortly after. The final state, outcome and report.json (session.report) are
persisted. Commands act on the simulator only; nothing here can reach a physical vehicle.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import redis
from rq.timeouts import JobTimeoutException

from app import metrics
from app.db import utc_now
from app.jobs.context import SHUTDOWN, WorkerContext, get_context
from app.jobs.queue import (
    RUN_KEY_TTL_S,
    run_control_key,
    run_frame_key,
    run_inject_key,
    run_live_key,
)
from app.logging_setup import log_context
from app.storage import artifacts, repo

logger = logging.getLogger("dreamdrive.jobs.sim")

KIND = "simulation"
FRAME_EVENT_MIN_INTERVAL_S = 0.2  # frame_ready <= 5 Hz
DECISION_EVENT_MIN_INTERVAL_S = 0.2
HARD_WATCHDOG_GRACE_S = 10.0
_MAX_PENDING_ITEMS = 20


def create_session(row: repo.RunRow, ctx: WorkerContext) -> tuple[Any, Any]:
    """Build the SimulationSession (and the inference client for learned_model mode)."""
    from app.agents.runner import SessionConfig, SimulationSession

    client = None
    if row.perception_source == "learned_model":
        from app.perception.client import InferenceClient

        client = InferenceClient(ctx.settings.inference_base_url, ctx.settings.sim_inference_timeout_seconds)
    cfg = SessionConfig(
        run_id=row.id,
        start_node=row.start_node,
        destination_node=row.destination_node,
        scenario=row.scenario,
        seed=row.seed,
        perception_source=row.perception_source,
        dt=ctx.settings.sim_fixed_dt_seconds,
    )
    return SimulationSession(cfg, inference_client=client), client


def _pop_all(client: redis.Redis, key: str) -> list[bytes]:
    items: list[bytes] = []
    for _ in range(_MAX_PENDING_ITEMS):
        value = client.lpop(key)
        if value is None:
            break
        items.append(value if isinstance(value, bytes) else str(value).encode())
    return items


class _Publisher:
    def __init__(self, ctx: WorkerContext, row: repo.RunRow) -> None:
        self.ctx = ctx
        self.row = row
        self.last_state: str | None = "initializing"
        self.last_frame_event = 0.0
        self.last_decision_event = 0.0
        self.last_decision_key: tuple[str, tuple[str, ...], str] | None = None
        self.last_frame_id: int | None = None
        self.started = False

    def publish_tick(self, res: Any) -> None:
        live = res.live
        try:
            pipe = self.ctx.redis.pipeline(transaction=True)
            if res.jpeg:
                pipe.set(run_frame_key(self.row.id), res.jpeg, ex=RUN_KEY_TTL_S)
            pipe.set(run_live_key(self.row.id), json.dumps(live, separators=(",", ":")), ex=RUN_KEY_TTL_S)
            pipe.expire(run_control_key(self.row.id), RUN_KEY_TTL_S)
            pipe.expire(run_inject_key(self.row.id), RUN_KEY_TTL_S)
            pipe.execute()
        except redis.RedisError:
            logger.warning("live frame publish failed", extra={"code": "redis_error"})
        self._metrics(res)
        if res.run_state != self.last_state and not res.terminal:
            self.state_changed(res.run_state, None)
        now = time.monotonic()
        # The final frame and decision of a run are always announced, whatever the throttle says.
        if res.terminal or now - self.last_frame_event >= FRAME_EVENT_MIN_INTERVAL_S:
            self.last_frame_event = now
            self.ctx.bus.publish(
                self.row.owner_id,
                "frame_ready",
                self.row.id,
                {"run_id": self.row.id, "frame_id": int(res.frame_id), "sim_time_s": max(0.0, float(res.sim_time_s))},
            )
        decision = live.get("decision") or {}
        maneuver = live.get("maneuver") or None
        primary = str(maneuver["primary"]) if maneuver else ""
        key = (str(decision.get("state")), tuple(decision.get("reason_codes") or ()), primary)
        changed = key != self.last_decision_key
        # Changes (decision or maneuver) are sent at most every 0.2 s (<= 5 Hz); an unchanged decision is repeated
        # once per second.
        if res.terminal or now - self.last_decision_event >= (DECISION_EVENT_MIN_INTERVAL_S if changed else 1.0):
            self.last_decision_key = key
            self.last_decision_event = now
            command = live.get("command")
            payload: dict[str, Any] = {
                "run_id": self.row.id,
                "frame_id": int(res.frame_id),
                "state": key[0],
                "reason_codes": list(key[1]),
                "command": None if not command else {k: float(command[k]) for k in ("steer", "throttle", "brake")},
                "sim_time_s": max(0.0, float(res.sim_time_s)),
            }
            if maneuver:
                payload["maneuver"] = {
                    k: maneuver[k] for k in ("primary", "longitudinal", "lateral", "since_sim_time_s")
                }
            self.ctx.bus.publish(self.row.owner_id, "decision", self.row.id, payload)

    def _metrics(self, res: Any) -> None:
        live = res.live
        perception = live.get("perception") or {}
        source = str(perception.get("source") or self.row.perception_source)
        if res.frame_id != self.last_frame_id:
            self.last_frame_id = res.frame_id
            metrics.FRAMES_PROCESSED.labels(KIND, source, str(perception.get("quality_status") or "unknown")).inc()
            command = live.get("command") or {}
            if command.get("safety_override"):
                reasons = command.get("reason_codes") or ["UNSPECIFIED"]
                metrics.SAFETY_OVERRIDES.labels(str(reasons[0]), self.row.perception_source).inc()
        rtf = live.get("realtime_factor")
        if isinstance(rtf, int | float):
            metrics.SIM_REALTIME_FACTOR.labels(self.row.simulator, self.row.perception_source).set(float(rtf))
        age = live.get("acquisition_age_s")
        if isinstance(age, int | float):
            metrics.FRAME_ACQUISITION_AGE.labels(source).set(float(age))

    def state_changed(self, state: str, outcome: str | None) -> None:
        fields: dict[str, Any] = {}
        if state in ("running", "paused", "stopping") and not self.started:
            fields["started_utc"] = utc_now()
            self.started = True
        repo.update_run(self.ctx.db, self.row.id, state=state, **fields)
        self.last_state = state
        self.ctx.bus.publish(
            self.row.owner_id, "run_state", self.row.id, {"run_id": self.row.id, "state": state, "outcome": outcome}
        )
        logger.info("run state", extra={"state": state})


def _apply_inputs(ctx: WorkerContext, row: repo.RunRow, session: Any) -> None:
    try:
        controls = _pop_all(ctx.redis, run_control_key(row.id))
        injects = _pop_all(ctx.redis, run_inject_key(row.id))
    except redis.RedisError:
        logger.warning("could not read run controls", extra={"code": "redis_error"})
        return
    for raw in controls:
        action = raw.decode("utf-8", "replace")
        if action == "pause":
            session.pause()
        elif action == "resume":
            session.resume()
        elif action == "stop":
            session.request_stop()
    for raw in injects:
        try:
            spec = json.loads(raw)
            accepted = bool(session.inject_obstacle(float(spec["distance_ahead_m"]), float(spec["lateral_offset_m"])))
        except (ValueError, KeyError, TypeError):
            accepted = False
        if not accepted:
            ctx.bus.publish(
                row.owner_id,
                "error",
                row.id,
                {"code": "obstacle_rejected", "message": "The simulator could not place the requested obstacle"},
            )


def run_simulation_job(run_id: str) -> str:
    ctx = get_context()
    with log_context(run_id=run_id):
        return _run(ctx, run_id)


def _run(ctx: WorkerContext, run_id: str) -> str:
    settings = ctx.settings
    row = repo.get_run_any(ctx.db, run_id)
    if row is None:
        logger.warning("run row missing; skipping", extra={"code": "run_missing"})
        return "missing"
    if row.state == "stopping" and row.started_utc is None:
        # Stopped by the operator before the worker picked it up: nothing was simulated.
        _persist_final(ctx, row, "stopped", "stopped_operator", None, None, None, None, 0.0)
        return "stopped"
    if row.state != "initializing":
        logger.info("run not initializing; skipping", extra={"state": row.state})
        return row.state

    pub = _Publisher(ctx, row)
    session: Any = None
    client: Any = None
    res: Any = None
    wall0 = time.monotonic()
    error: tuple[str, str] | None = None
    forced: tuple[str, str] | None = None
    try:
        session, client = create_session(row, ctx)
        res = session.start()
        pub.publish_tick(res)
        dt = float(settings.sim_fixed_dt_seconds)
        origin = time.monotonic() - float(res.sim_time_s)  # wall instant that corresponds to sim time 0
        was_paused = False
        while not res.terminal:
            elapsed = time.monotonic() - wall0
            session.note_wall_time(elapsed)
            if SHUTDOWN.is_set():
                forced = ("failed_worker_shutdown", "worker_shutdown")
                break
            if elapsed > settings.max_simulation_wall_seconds + HARD_WATCHDOG_GRACE_S:
                forced = ("failed_wall_time", "wall_time_limit")
                break
            _apply_inputs(ctx, row, session)
            paused = res.run_state == "paused"
            if paused:
                time.sleep(dt)  # no physics while paused; keep refreshing state at the tick rate
            else:
                now = time.monotonic()
                if was_paused:
                    origin = now - float(res.sim_time_s)  # pause time does not count as simulated time
                target = origin + float(res.sim_time_s) + dt
                if target > now:
                    time.sleep(target - now)
                elif now - target > dt:
                    origin = now - float(res.sim_time_s) - dt  # fell behind: never burst to catch up
            was_paused = paused
            res = session.tick()
            pub.publish_tick(res)
    except JobTimeoutException:
        forced = ("failed_wall_time", "job_timeout")
    except Exception as exc:
        from app.agents.runner import RouteUnreachable, UnsupportedMode

        if isinstance(exc, RouteUnreachable):
            error = ("destination_unreachable", "The destination is not reachable from the start node")
        elif isinstance(exc, UnsupportedMode):
            error = ("unsupported_mode", "The requested perception mode or scenario is not supported")
        else:
            logger.exception("simulation crashed", extra={"code": "internal_error"})
            error = ("internal_error", "Internal error during the simulation")
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001 - closing must not mask the run result
                logger.warning("inference client close failed")
    wall_s = time.monotonic() - wall0
    if error is not None or res is None:
        code, message = error or ("internal_error", "Internal error during the simulation")
        _persist_final(ctx, row, "failed", "failed_internal", code, message, session, res, wall_s)
        return "failed"
    if forced is not None:
        outcome, code = forced
        message = {
            "worker_shutdown": "The worker shut down during the run",
            "wall_time_limit": "The run exceeded the wall-clock limit",
            "job_timeout": "The run exceeded the job time limit",
        }[code]
        _persist_final(ctx, row, "failed", outcome, code, message, session, res, wall_s)
        return "failed"
    state = res.run_state if res.run_state in repo.RUN_TERMINAL_STATES else "failed"
    _persist_final(ctx, row, state, res.outcome, None, None, session, res, wall_s)
    return state


def _persist_final(
    ctx: WorkerContext,
    row: repo.RunRow,
    state: str,
    outcome: str | None,
    error_code: str | None,
    error_message: str | None,
    session: Any,
    res: Any,
    wall_s: float,
) -> None:
    report_available = False
    report_error: str | None = None
    if session is not None and res is not None:
        try:
            report = session.report(ctx.settings.git_commit, wall_s)
            artifacts.atomic_write_json(artifacts.run_dir(ctx.settings.artifact_root, row.id) / "report.json", report)
            report_available = True
        except Exception:  # noqa: BLE001 - e.g. no valid build commit: the run result itself still stands
            logger.exception("run report could not be written", extra={"code": "report_failed"})
            report_error = "report_generation_failed"
    summary: dict[str, Any] = {
        "live": res.live if res is not None else None,
        "wall_time_s": round(wall_s, 3),
        "report_available": report_available,
    }
    if report_error:
        summary["report_error"] = report_error
    applied = repo.transition_run(
        ctx.db,
        row.id,
        repo.RUN_ACTIVE_STATES,
        state,
        outcome=outcome,
        error_code=error_code,
        error_message=error_message,
        summary=summary,
        finished_utc=utc_now(),
    )
    try:
        ctx.redis.delete(run_control_key(row.id), run_inject_key(row.id))
    except redis.RedisError:
        pass
    if applied:
        metrics.JOBS_TOTAL.labels(KIND, state).inc()
        ctx.bus.publish(row.owner_id, "run_state", row.id, {"run_id": row.id, "state": state, "outcome": outcome})
        logger.info("run finished", extra={"state": state, "outcome": outcome, "code": error_code})
