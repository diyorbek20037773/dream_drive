"""Typed database access for sessions, jobs, runs and evaluations (docs/IMPLEMENTATION_SPEC.md section 3.2).

Ownership rule: owner-facing lookups take ``owner_id`` and filter on it, so a foreign id is indistinguishable from an
unknown one (the API answers 404 ``not_found`` for both). The ``*_any`` functions are for the worker and retention
code only. Admission checks (queue limits, one active simulation) and the insert run inside one ``BEGIN IMMEDIATE``
transaction, so concurrent requests cannot both pass a limit.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.db import Database, from_db_time, now_db, to_db_time

JOB_ACTIVE_STATES: tuple[str, ...] = ("queued", "validating", "running", "cancelling")
JOB_IN_PROGRESS_STATES: tuple[str, ...] = ("validating", "running", "cancelling")
JOB_TERMINAL_STATES: tuple[str, ...] = ("cancelled", "succeeded", "failed", "interrupted")
RUN_ACTIVE_STATES: tuple[str, ...] = ("initializing", "running", "paused", "stopping")
RUN_TERMINAL_STATES: tuple[str, ...] = ("stopped", "arrived", "failed")
EVAL_ACTIVE_STATES: tuple[str, ...] = ("queued", "running")
EVAL_TERMINAL_STATES: tuple[str, ...] = ("succeeded", "failed", "interrupted")


class AdmissionDenied(Exception):  # noqa: N818 - domain wording
    """A queue/concurrency limit refused the new resource. ``code`` is an ErrorCode value."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = details or {}


def _dumps(value: Any) -> str | None:
    return None if value is None else json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _loads(value: str | None) -> Any:
    return None if value is None else json.loads(value)


def _placeholders(values: Sequence[object]) -> str:
    return ",".join("?" for _ in values)


def _req_time(value: str) -> datetime:
    parsed = from_db_time(value)
    assert parsed is not None
    return parsed


# ---------------------------------------------------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SessionRow:
    id: str
    role: str
    csrf_token: str
    created_utc: datetime
    last_seen_utc: datetime


def _session(row: sqlite3.Row) -> SessionRow:
    return SessionRow(
        id=row["id"],
        role=row["role"],
        csrf_token=row["csrf_token"],
        created_utc=_req_time(row["created_utc"]),
        last_seen_utc=_req_time(row["last_seen_utc"]),
    )


def create_session(db: Database, session_id: str, role: str, csrf_token: str) -> SessionRow:
    now = now_db()
    db.execute(
        "INSERT INTO sessions (id, role, csrf_token, created_utc, last_seen_utc) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, csrf_token, now, now),
    )
    row = get_session(db, session_id)
    assert row is not None
    return row


def get_session(db: Database, session_id: str) -> SessionRow | None:
    row = db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return _session(row) if row else None


def touch_session(db: Database, session_id: str, role: str | None = None) -> None:
    if role is None:
        db.execute("UPDATE sessions SET last_seen_utc = ? WHERE id = ?", (now_db(), session_id))
    else:
        db.execute("UPDATE sessions SET last_seen_utc = ?, role = ? WHERE id = ?", (now_db(), role, session_id))


def delete_expired_sessions(db: Database, cutoff: datetime) -> int:
    cur = db.execute("DELETE FROM sessions WHERE last_seen_utc < ?", (to_db_time(cutoff),))
    return cur.rowcount


# ---------------------------------------------------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class JobRow:
    id: str
    owner_id: str
    kind: str
    state: str
    stage: str | None
    progress: float
    frames_done: int
    frames_total: int | None
    error_code: str | None
    error_message: str | None
    input_sha256: str
    input_bytes: int
    input_ext: str
    probe: dict[str, Any] | None
    model: dict[str, Any] | None
    settings: dict[str, Any]
    summary: dict[str, Any] | None
    cancel_requested: bool
    delete_requested: bool
    created_utc: datetime
    started_utc: datetime | None
    finished_utc: datetime | None
    updated_utc: datetime

    @property
    def terminal(self) -> bool:
        return self.state in JOB_TERMINAL_STATES


def _job(row: sqlite3.Row) -> JobRow:
    return JobRow(
        id=row["id"],
        owner_id=row["owner_id"],
        kind=row["kind"],
        state=row["state"],
        stage=row["stage"],
        progress=float(row["progress"] or 0.0),
        frames_done=int(row["frames_done"] or 0),
        frames_total=row["frames_total"],
        error_code=row["error_code"],
        error_message=row["error_message"],
        input_sha256=row["input_sha256"],
        input_bytes=int(row["input_bytes"]),
        input_ext=row["input_ext"],
        probe=_loads(row["probe_json"]),
        model=_loads(row["model_json"]),
        settings=_loads(row["settings_json"]) or {},
        summary=_loads(row["summary_json"]),
        cancel_requested=bool(row["cancel_requested"]),
        delete_requested=bool(row["delete_requested"]),
        created_utc=_req_time(row["created_utc"]),
        started_utc=from_db_time(row["started_utc"]),
        finished_utc=from_db_time(row["finished_utc"]),
        updated_utc=_req_time(row["updated_utc"]),
    )


_JOB_COLUMNS = {
    "state",
    "stage",
    "progress",
    "frames_done",
    "frames_total",
    "error_code",
    "error_message",
    "probe_json",
    "model_json",
    "summary_json",
    "cancel_requested",
    "delete_requested",
    "started_utc",
    "finished_utc",
}
_JSON_FIELDS = {"probe": "probe_json", "model": "model_json", "summary": "summary_json", "route": "route_json"}


def _assignments(fields: dict[str, Any], allowed: set[str]) -> tuple[list[str], list[Any]]:
    cols: list[str] = []
    values: list[Any] = []
    for key, value in fields.items():
        column = _JSON_FIELDS.get(key, key)
        if column not in allowed:
            raise ValueError(f"column not updatable: {key}")
        if column.endswith("_json"):
            value = _dumps(value)
        elif isinstance(value, datetime):
            value = to_db_time(value)
        elif isinstance(value, bool):
            value = int(value)
        cols.append(f"{column} = ?")
        values.append(value)
    cols.append("updated_utc = ?")
    values.append(now_db())
    return cols, values


def insert_job(
    db: Database,
    *,
    job_id: str,
    owner_id: str,
    input_sha256: str,
    input_bytes: int,
    input_ext: str,
    settings: dict[str, Any],
    max_per_session: int,
    max_total: int,
) -> JobRow:
    """Insert a queued job if the per-session and global admission limits allow it."""
    now = now_db()
    active = _placeholders(JOB_ACTIVE_STATES)
    with db.transaction() as conn:
        own = conn.execute(
            f"SELECT COUNT(*) FROM jobs WHERE owner_id = ? AND state IN ({active})",  # noqa: S608 - placeholders only
            (owner_id, *JOB_ACTIVE_STATES),
        ).fetchone()[0]
        if own >= max_per_session:
            raise AdmissionDenied(
                "session_queue_full",
                "This session already has the maximum number of queued or running videos",
                {"limit": max_per_session, "active": own},
            )
        total = conn.execute(
            f"SELECT COUNT(*) FROM jobs WHERE state IN ({active})",  # noqa: S608
            JOB_ACTIVE_STATES,
        ).fetchone()[0]
        if total >= max_total:
            raise AdmissionDenied("queue_full", "The analysis queue is full", {"limit": max_total, "active": total})
        conn.execute(
            "INSERT INTO jobs (id, owner_id, kind, state, stage, progress, frames_done, input_sha256, input_bytes,"
            " input_ext, settings_json, created_utc, updated_utc)"
            " VALUES (?, ?, 'video_analysis', 'queued', 'queued', 0, 0, ?, ?, ?, ?, ?, ?)",
            (job_id, owner_id, input_sha256, input_bytes, input_ext, _dumps(settings), now, now),
        )
    row = get_job_any(db, job_id)
    assert row is not None
    return row


def get_job(db: Database, owner_id: str, job_id: str) -> JobRow | None:
    """Owner-filtered lookup; deleted-but-still-stopping jobs are hidden."""
    row = db.execute(
        "SELECT * FROM jobs WHERE id = ? AND owner_id = ? AND delete_requested = 0", (job_id, owner_id)
    ).fetchone()
    return _job(row) if row else None


def get_job_any(db: Database, job_id: str) -> JobRow | None:
    row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _job(row) if row else None


def list_jobs(db: Database, owner_id: str, limit: int, after: tuple[str, str] | None = None) -> list[JobRow]:
    """Newest first. ``after`` is the (created_utc, id) of the last item of the previous page."""
    if after is None:
        rows = db.execute(
            "SELECT * FROM jobs WHERE owner_id = ? AND delete_requested = 0 ORDER BY created_utc DESC, id DESC LIMIT ?",
            (owner_id, limit),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM jobs WHERE owner_id = ? AND delete_requested = 0 AND (created_utc, id) < (?, ?)"
            " ORDER BY created_utc DESC, id DESC LIMIT ?",
            (owner_id, after[0], after[1], limit),
        ).fetchall()
    return [_job(r) for r in rows]


def count_jobs(db: Database, states: Iterable[str], owner_id: str | None = None) -> int:
    states = tuple(states)
    sql = f"SELECT COUNT(*) FROM jobs WHERE state IN ({_placeholders(states)})"  # noqa: S608
    params: tuple[Any, ...] = states
    if owner_id is not None:
        sql += " AND owner_id = ?"
        params = (*states, owner_id)
    return int(db.execute(sql, params).fetchone()[0])


def update_job(db: Database, job_id: str, **fields: Any) -> None:
    cols, values = _assignments(fields, _JOB_COLUMNS)
    db.execute(f"UPDATE jobs SET {', '.join(cols)} WHERE id = ?", (*values, job_id))  # noqa: S608 - allowlisted


def transition_job(db: Database, job_id: str, from_states: Iterable[str], to_state: str, **fields: Any) -> bool:
    """Conditional state change: applies only if the current state is in ``from_states``. True if applied."""
    from_states = tuple(from_states)
    cols, values = _assignments({"state": to_state, **fields}, _JOB_COLUMNS)
    cur = db.execute(
        f"UPDATE jobs SET {', '.join(cols)} WHERE id = ? AND state IN ({_placeholders(from_states)})",  # noqa: S608
        (*values, job_id, *from_states),
    )
    return cur.rowcount == 1


def delete_job_row(db: Database, job_id: str, owner_id: str) -> None:
    """Remove the row and leave an owner-scoped tombstone (idempotent repeated DELETE)."""
    with db.transaction() as conn:
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        conn.execute(
            "INSERT OR REPLACE INTO job_tombstones (id, owner_id, deleted_utc) VALUES (?, ?, ?)",
            (job_id, owner_id, now_db()),
        )


def has_job_tombstone(db: Database, owner_id: str, job_id: str) -> bool:
    row = db.execute("SELECT 1 FROM job_tombstones WHERE id = ? AND owner_id = ?", (job_id, owner_id)).fetchone()
    return row is not None


def delete_expired_tombstones(db: Database, cutoff: datetime) -> int:
    return db.execute("DELETE FROM job_tombstones WHERE deleted_utc < ?", (to_db_time(cutoff),)).rowcount


def jobs_in_states(db: Database, states: Iterable[str]) -> list[JobRow]:
    states = tuple(states)
    rows = db.execute(
        f"SELECT * FROM jobs WHERE state IN ({_placeholders(states)}) ORDER BY created_utc",  # noqa: S608
        states,
    ).fetchall()
    return [_job(r) for r in rows]


def jobs_created_before(db: Database, cutoff: datetime, states: Iterable[str]) -> list[JobRow]:
    states = tuple(states)
    rows = db.execute(
        f"SELECT * FROM jobs WHERE created_utc < ? AND state IN ({_placeholders(states)})",  # noqa: S608
        (to_db_time(cutoff), *states),
    ).fetchall()
    return [_job(r) for r in rows]


def job_ids(db: Database) -> set[str]:
    return {r[0] for r in db.execute("SELECT id FROM jobs").fetchall()}


# ---------------------------------------------------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RunRow:
    id: str
    owner_id: str
    simulator: str
    perception_source: str
    scenario: str
    seed: int
    start_node: str
    destination_node: str
    route: dict[str, Any]
    state: str
    outcome: str | None
    error_code: str | None
    error_message: str | None
    summary: dict[str, Any] | None
    obstacles_injected: int
    created_utc: datetime
    started_utc: datetime | None
    finished_utc: datetime | None
    updated_utc: datetime

    @property
    def terminal(self) -> bool:
        return self.state in RUN_TERMINAL_STATES


def _run(row: sqlite3.Row) -> RunRow:
    return RunRow(
        id=row["id"],
        owner_id=row["owner_id"],
        simulator=row["simulator"],
        perception_source=row["perception_source"],
        scenario=row["scenario"],
        seed=int(row["seed"]),
        start_node=row["start_node"],
        destination_node=row["destination_node"],
        route=_loads(row["route_json"]) or {},
        state=row["state"],
        outcome=row["outcome"],
        error_code=row["error_code"],
        error_message=row["error_message"],
        summary=_loads(row["summary_json"]),
        obstacles_injected=int(row["obstacles_injected"] or 0),
        created_utc=_req_time(row["created_utc"]),
        started_utc=from_db_time(row["started_utc"]),
        finished_utc=from_db_time(row["finished_utc"]),
        updated_utc=_req_time(row["updated_utc"]),
    )


_RUN_COLUMNS = {
    "state",
    "outcome",
    "error_code",
    "error_message",
    "summary_json",
    "started_utc",
    "finished_utc",
}


def insert_run(
    db: Database,
    *,
    run_id: str,
    owner_id: str,
    simulator: str,
    perception_source: str,
    scenario: str,
    seed: int,
    start_node: str,
    destination_node: str,
    route: dict[str, Any],
    max_active: int,
) -> RunRow:
    """Insert an initializing run unless ``max_active`` simulations are already active on this instance."""
    now = now_db()
    active = _placeholders(RUN_ACTIVE_STATES)
    with db.transaction() as conn:
        busy = conn.execute(
            f"SELECT COUNT(*) FROM runs WHERE state IN ({active})",  # noqa: S608
            RUN_ACTIVE_STATES,
        ).fetchone()[0]
        if busy >= max_active:
            raise AdmissionDenied(
                "simulation_busy", "A simulation is already active on this instance", {"limit": max_active}
            )
        conn.execute(
            "INSERT INTO runs (id, owner_id, simulator, perception_source, scenario, seed, start_node,"
            " destination_node, route_json, state, created_utc, updated_utc)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'initializing', ?, ?)",
            (
                run_id,
                owner_id,
                simulator,
                perception_source,
                scenario,
                seed,
                start_node,
                destination_node,
                _dumps(route),
                now,
                now,
            ),
        )
    row = get_run_any(db, run_id)
    assert row is not None
    return row


def get_run(db: Database, owner_id: str, run_id: str) -> RunRow | None:
    row = db.execute("SELECT * FROM runs WHERE id = ? AND owner_id = ?", (run_id, owner_id)).fetchone()
    return _run(row) if row else None


def get_run_any(db: Database, run_id: str) -> RunRow | None:
    row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return _run(row) if row else None


def list_runs(db: Database, owner_id: str, limit: int, after: tuple[str, str] | None = None) -> list[RunRow]:
    if after is None:
        rows = db.execute(
            "SELECT * FROM runs WHERE owner_id = ? ORDER BY created_utc DESC, id DESC LIMIT ?", (owner_id, limit)
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM runs WHERE owner_id = ? AND (created_utc, id) < (?, ?)"
            " ORDER BY created_utc DESC, id DESC LIMIT ?",
            (owner_id, after[0], after[1], limit),
        ).fetchall()
    return [_run(r) for r in rows]


def count_runs(db: Database, states: Iterable[str]) -> int:
    states = tuple(states)
    return int(
        db.execute(
            f"SELECT COUNT(*) FROM runs WHERE state IN ({_placeholders(states)})",  # noqa: S608
            states,
        ).fetchone()[0]
    )


def update_run(db: Database, run_id: str, **fields: Any) -> None:
    cols, values = _assignments(fields, _RUN_COLUMNS)
    db.execute(f"UPDATE runs SET {', '.join(cols)} WHERE id = ?", (*values, run_id))  # noqa: S608


def transition_run(db: Database, run_id: str, from_states: Iterable[str], to_state: str, **fields: Any) -> bool:
    from_states = tuple(from_states)
    cols, values = _assignments({"state": to_state, **fields}, _RUN_COLUMNS)
    cur = db.execute(
        f"UPDATE runs SET {', '.join(cols)} WHERE id = ? AND state IN ({_placeholders(from_states)})",  # noqa: S608
        (*values, run_id, *from_states),
    )
    return cur.rowcount == 1


def reserve_obstacle(db: Database, run_id: str, max_per_run: int, states: Iterable[str]) -> int | None:
    """Atomically count one more injection; None when the limit is reached or the run is not in ``states``."""
    states = tuple(states)
    marks = _placeholders(states)
    sql = f"UPDATE runs SET obstacles_injected = obstacles_injected + 1, updated_utc = ? WHERE id = ? AND obstacles_injected < ? AND state IN ({marks})"  # noqa: S608, E501
    cur = db.execute(sql, (now_db(), run_id, max_per_run, *states))
    if cur.rowcount != 1:
        return None
    row = db.execute("SELECT obstacles_injected FROM runs WHERE id = ?", (run_id,)).fetchone()
    return int(row[0])


def runs_in_states(db: Database, states: Iterable[str]) -> list[RunRow]:
    states = tuple(states)
    rows = db.execute(
        f"SELECT * FROM runs WHERE state IN ({_placeholders(states)}) ORDER BY created_utc",  # noqa: S608
        states,
    ).fetchall()
    return [_run(r) for r in rows]


def runs_created_before(db: Database, cutoff: datetime, states: Iterable[str]) -> list[RunRow]:
    states = tuple(states)
    rows = db.execute(
        f"SELECT * FROM runs WHERE created_utc < ? AND state IN ({_placeholders(states)})",  # noqa: S608
        (to_db_time(cutoff), *states),
    ).fetchall()
    return [_run(r) for r in rows]


def delete_run_row(db: Database, run_id: str) -> None:
    db.execute("DELETE FROM runs WHERE id = ?", (run_id,))


def run_ids(db: Database) -> set[str]:
    return {r[0] for r in db.execute("SELECT id FROM runs").fetchall()}


# ---------------------------------------------------------------------------------------------------------------------
# Evaluations
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EvaluationRow:
    id: str
    owner_id: str | None
    kind: str
    state: str
    created_utc: datetime
    finished_utc: datetime | None
    summary: dict[str, Any] | None
    error_code: str | None
    seeds: list[int] | None
    artifact_id: str | None


def _evaluation(row: sqlite3.Row) -> EvaluationRow:
    return EvaluationRow(
        id=row["id"],
        owner_id=row["owner_id"],
        kind=row["kind"],
        state=row["state"],
        created_utc=_req_time(row["created_utc"]),
        finished_utc=from_db_time(row["finished_utc"]),
        summary=_loads(row["summary_json"]),
        error_code=row["error_code"],
        seeds=_loads(row["seeds_json"]),
        artifact_id=row["artifact_id"],
    )


def insert_evaluation(
    db: Database, *, eval_id: str, owner_id: str, kind: str, seeds: list[int], max_active: int = 1
) -> EvaluationRow:
    now = now_db()
    active = _placeholders(EVAL_ACTIVE_STATES)
    with db.transaction() as conn:
        busy = conn.execute(
            f"SELECT COUNT(*) FROM evaluations WHERE state IN ({active})",  # noqa: S608
            EVAL_ACTIVE_STATES,
        ).fetchone()[0]
        if busy >= max_active:
            raise AdmissionDenied("evaluation_busy", "An evaluation suite is already queued or running")
        conn.execute(
            "INSERT INTO evaluations (id, owner_id, kind, state, created_utc, seeds_json, updated_utc)"
            " VALUES (?, ?, ?, 'queued', ?, ?, ?)",
            (eval_id, owner_id, kind, now, _dumps(seeds), now),
        )
    row = get_evaluation_any(db, eval_id)
    assert row is not None
    return row


def get_evaluation(db: Database, owner_id: str, eval_id: str) -> EvaluationRow | None:
    row = db.execute("SELECT * FROM evaluations WHERE id = ? AND owner_id = ?", (eval_id, owner_id)).fetchone()
    return _evaluation(row) if row else None


def get_evaluation_any(db: Database, eval_id: str) -> EvaluationRow | None:
    row = db.execute("SELECT * FROM evaluations WHERE id = ?", (eval_id,)).fetchone()
    return _evaluation(row) if row else None


def list_owner_evaluations(db: Database, owner_id: str, limit: int = 100) -> list[EvaluationRow]:
    rows = db.execute(
        "SELECT * FROM evaluations WHERE owner_id = ? ORDER BY created_utc DESC LIMIT ?", (owner_id, limit)
    ).fetchall()
    return [_evaluation(r) for r in rows]


def update_evaluation(db: Database, eval_id: str, **fields: Any) -> None:
    allowed = {"state", "finished_utc", "summary_json", "error_code", "artifact_id"}
    cols, values = _assignments(fields, allowed)
    db.execute(f"UPDATE evaluations SET {', '.join(cols)} WHERE id = ?", (*values, eval_id))  # noqa: S608


def transition_evaluation(db: Database, eval_id: str, from_states: Iterable[str], to_state: str, **fields: Any) -> bool:
    from_states = tuple(from_states)
    allowed = {"state", "finished_utc", "summary_json", "error_code", "artifact_id"}
    cols, values = _assignments({"state": to_state, **fields}, allowed)
    cur = db.execute(
        f"UPDATE evaluations SET {', '.join(cols)} WHERE id = ? AND state IN ({_placeholders(from_states)})",  # noqa: S608
        (*values, eval_id, *from_states),
    )
    return cur.rowcount == 1


def evaluations_in_states(db: Database, states: Iterable[str]) -> list[EvaluationRow]:
    states = tuple(states)
    rows = db.execute(
        f"SELECT * FROM evaluations WHERE state IN ({_placeholders(states)})",  # noqa: S608
        states,
    ).fetchall()
    return [_evaluation(r) for r in rows]


def delete_expired_evaluation_rows(db: Database, cutoff: datetime) -> int:
    """Remove failed/interrupted request rows; succeeded results live on as evaluation artifacts (registry)."""
    return db.execute(
        "DELETE FROM evaluations WHERE created_utc < ? AND state IN ('failed', 'interrupted')", (to_db_time(cutoff),)
    ).rowcount
