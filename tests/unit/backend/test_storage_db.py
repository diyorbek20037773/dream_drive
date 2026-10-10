"""SQLite layer, repository ownership/admission, retention cleanup and worker restart reconciliation."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from datetime import timedelta
from pathlib import Path

import fakeredis
import pytest
from rq import Queue
from test_api_harness import make_settings

from app.db import Database, apply_migrations, discover_migrations, utc_now
from app.events.bus import EventBus
from app.jobs.maintenance import reconcile_after_restart, retention_cleanup
from app.jobs.queue import enqueue_video_job
from app.storage import artifacts, repo


def _db(tmp_path: Path) -> Database:
    return Database(tmp_path / "t.db")


def _job(db: Database, owner: str = "o1", **kw: object) -> repo.JobRow:
    return repo.insert_job(
        db,
        job_id=str(kw.get("job_id") or artifacts.new_id()),
        owner_id=owner,
        input_sha256="a" * 64,
        input_bytes=10,
        input_ext="mp4",
        settings={"analysis_fps": 5.0},
        max_per_session=int(kw.get("per", 5)),  # type: ignore[arg-type]
        max_total=int(kw.get("total", 10)),  # type: ignore[arg-type]
    )


def test_migrations_apply_once_with_wal_and_foreign_keys(tmp_path: Path) -> None:
    db = _db(tmp_path)
    assert not db.path.exists()
    conn = db.conn()
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000
    versions = [r[0] for r in conn.execute("SELECT version FROM schema_migrations")]
    assert versions == [v for v, _ in discover_migrations()]
    assert apply_migrations(conn) == []
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"sessions", "jobs", "runs", "evaluations", "schema_migrations", "job_tombstones"} <= tables
    db.check()
    # A second process-like handle sees the same schema and applies nothing.
    other = Database(db.path)
    assert other.migrate() == []
    db.close()
    other.close()


def test_per_thread_connections(tmp_path: Path) -> None:
    db = _db(tmp_path)
    main = db.conn()
    seen: list[sqlite3.Connection] = []
    thread = threading.Thread(target=lambda: seen.append(db.conn()))
    thread.start()
    thread.join()
    assert seen and seen[0] is not main
    db.close()


def test_state_check_constraint(tmp_path: Path) -> None:
    db = _db(tmp_path)
    job = _job(db)
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE jobs SET state = 'exploded' WHERE id = ?", (job.id,))
    db.close()


def test_owner_filtering_and_admission(tmp_path: Path) -> None:
    db = _db(tmp_path)
    a = _job(db, "A", per=2, total=3)
    _job(db, "A", per=2, total=3)
    with pytest.raises(repo.AdmissionDenied) as per_session:
        _job(db, "A", per=2, total=3)
    assert per_session.value.code == "session_queue_full"
    _job(db, "B", per=2, total=3)
    with pytest.raises(repo.AdmissionDenied) as total:
        _job(db, "C", per=2, total=3)
    assert total.value.code == "queue_full"
    assert repo.get_job(db, "A", a.id) is not None
    assert repo.get_job(db, "B", a.id) is None  # foreign id == unknown id
    assert [j.owner_id for j in repo.list_jobs(db, "B", 10)] == ["B"]
    # Terminal jobs free capacity.
    assert repo.transition_job(db, a.id, ("queued",), "cancelled")
    _job(db, "C", per=2, total=3)
    db.close()


def test_conditional_transitions(tmp_path: Path) -> None:
    db = _db(tmp_path)
    job = _job(db)
    assert repo.transition_job(db, job.id, ("queued",), "validating", started_utc=utc_now())
    assert not repo.transition_job(db, job.id, ("queued",), "cancelled")
    with pytest.raises(ValueError):
        repo.update_job(db, job.id, owner_id="evil")  # not an updatable column
    db.close()


def test_concurrent_run_admission_allows_one(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.conn()
    results: list[str] = []
    barrier = threading.Barrier(6)

    def attempt(i: int) -> None:
        barrier.wait()
        try:
            repo.insert_run(
                db,
                run_id=artifacts.new_id(),
                owner_id=f"o{i}",
                simulator="lite",
                perception_source="synthetic_rgb_cv",
                scenario="normal",
                seed=1,
                start_node="A",
                destination_node="C",
                route={"node_ids": ["A", "C"], "polyline": [], "length_m": 1.0},
                max_active=1,
            )
            results.append("ok")
        except repo.AdmissionDenied as exc:
            results.append(exc.code)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["ok"] + ["simulation_busy"] * 5
    db.close()


def test_retention_cleanup_removes_expired_bytes_and_rows(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, artifact_retention_hours=1)
    db = Database(settings.sqlite_path)
    old = _job(db)
    fresh = _job(db)
    running_old = _job(db)
    for job in (old, fresh, running_old):
        d = artifacts.job_dir(settings.artifact_root, job.id)
        d.mkdir(parents=True)
        (d / "input.mp4").write_bytes(b"x" * 100)
    two_hours_ago = (utc_now() - timedelta(hours=2)).isoformat(timespec="microseconds")
    db.execute("UPDATE jobs SET state='succeeded', created_utc=? WHERE id=?", (two_hours_ago, old.id))
    db.execute("UPDATE jobs SET state='succeeded' WHERE id=?", (fresh.id,))
    db.execute("UPDATE jobs SET state='running', created_utc=? WHERE id=?", (two_hours_ago, running_old.id))
    repo.create_session(db, "s" * 32, "anonymous", "tok")
    db.execute(
        "UPDATE sessions SET last_seen_utc=? WHERE id=?", ((utc_now() - timedelta(days=3)).isoformat(), "s" * 32)
    )
    orphan = settings.artifact_root / "jobs" / ("f" * 32)
    orphan.mkdir(parents=True)
    (orphan / "input.mp4").write_bytes(b"y" * 50)
    past = time.time() - 3 * 3600
    os.utime(orphan, (past, past))
    settings.tmp_root.mkdir(parents=True, exist_ok=True)
    stale = settings.tmp_root / "upload-abc.part"
    stale.write_bytes(b"z" * 10)
    os.utime(stale, (past, past))
    fresh_tmp = settings.tmp_root / "upload-new.part"
    fresh_tmp.write_bytes(b"n")

    stats = retention_cleanup(settings, db)
    assert stats.removed_jobs == 1 and stats.removed_bytes == 100 + 50 + 10 and stats.removed_sessions == 1
    assert repo.get_job_any(db, old.id) is None and repo.has_job_tombstone(db, old.owner_id, old.id)
    assert not artifacts.job_dir(settings.artifact_root, old.id).exists()
    assert repo.get_job_any(db, fresh.id) is not None and repo.get_job_any(db, running_old.id) is not None
    assert not orphan.exists() and not stale.exists() and fresh_tmp.exists()
    db.close()


def test_reconcile_after_restart(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    db = Database(settings.sqlite_path)
    client = fakeredis.FakeRedis()
    bus = EventBus(client)
    running = _job(db, "A")
    repo.transition_job(db, running.id, ("queued",), "running")
    lost = _job(db, "A")  # queued in SQLite but never reached Redis
    kept = _job(db, "B")
    enqueue_video_job(client, settings, kept.id)
    deleted = _job(db, "B")
    repo.transition_job(db, deleted.id, ("queued",), "cancelling", delete_requested=True)
    artifacts.job_dir(settings.artifact_root, deleted.id).mkdir(parents=True)
    run = repo.insert_run(
        db,
        run_id=artifacts.new_id(),
        owner_id="A",
        simulator="lite",
        perception_source="synthetic_rgb_cv",
        scenario="normal",
        seed=1,
        start_node="A",
        destination_node="C",
        route={"node_ids": [], "polyline": [], "length_m": 0.0},
        max_active=1,
    )
    repo.transition_run(db, run.id, ("initializing",), "running")
    client.set(f"dd:run:{run.id}:live", b"{}")

    stats = reconcile_after_restart(db, client, bus, settings.artifact_root)
    assert stats.jobs_interrupted == 3 and stats.runs_failed == 1
    assert repo.get_job_any(db, running.id).error_code == "worker_restarted"  # type: ignore[union-attr]
    assert repo.get_job_any(db, lost.id).error_code == "queue_lost"  # type: ignore[union-attr]
    assert repo.get_job_any(db, kept.id).state == "queued"  # type: ignore[union-attr]
    assert repo.get_job_any(db, deleted.id) is None
    assert not artifacts.job_dir(settings.artifact_root, deleted.id).exists()
    failed_run = repo.get_run_any(db, run.id)
    assert failed_run is not None and failed_run.state == "failed" and failed_run.error_code == "worker_restarted"
    assert client.get(f"dd:run:{run.id}:live") is None
    assert Queue("video", connection=client).job_ids == [f"video-{kept.id}"]
    states = [e[1][b"type"] for e in client.xrange("dd:events:A")]
    assert states.count(b"job_state") == 2 and b"run_state" in states
    # Running it again changes nothing (idempotent).
    again = reconcile_after_restart(db, client, bus, settings.artifact_root)
    assert again.jobs_interrupted == 0 and again.runs_failed == 0
    db.close()


def test_artifact_paths_reject_bad_ids(tmp_path: Path) -> None:
    for bad in ("../x", "a" * 31, "A" * 32, "a" * 32 + "/../../etc", ""):
        with pytest.raises(ValueError):
            artifacts.job_dir(tmp_path, bad)
        with pytest.raises(ValueError):
            artifacts.run_dir(tmp_path, bad)
    job_id = "b" * 32
    assert artifacts.artifact_file(tmp_path, job_id, "../dreamdrive.db", "mp4") is None
    assert artifacts.artifact_file(tmp_path, job_id, "input", "exe") is None


def test_atomic_write_json(tmp_path: Path) -> None:
    target = tmp_path / "runs" / "r" / "report.json"
    artifacts.atomic_write_json(target, {"a": 1})
    assert target.read_text(encoding="utf-8").strip() == '{\n  "a": 1\n}'
    assert [p.name for p in target.parent.iterdir()] == ["report.json"]
