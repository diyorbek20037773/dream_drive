"""SQLite access: per-thread connections (WAL, foreign keys, busy timeout) and an ordered migration runner.

The API (threadpool handlers) and the worker processes share one database file on the `app-data` volume. WAL lets
readers proceed while one writer commits; writes that must be atomic with a check (queue admission, conditional
state transitions) use ``BEGIN IMMEDIATE`` so the check and the write happen under the same write lock.

Opening a :class:`Database` touches nothing on disk; the first :meth:`Database.conn` call creates the file and
applies pending migrations (``app/migrations/NNNN_name.sql``, recorded in ``schema_migrations``).
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("dreamdrive.db")

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
_MIGRATION_RE = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")
BUSY_TIMEOUT_MS = 10_000


def utc_now() -> datetime:
    return datetime.now(UTC)


def to_db_time(value: datetime) -> str:
    """Fixed-width ISO string (microseconds, +00:00) so lexical order equals time order."""
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def from_db_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def now_db() -> str:
    return to_db_time(utc_now())


def discover_migrations(directory: Path = MIGRATIONS_DIR) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in directory.glob("*.sql"):
        match = _MIGRATION_RE.match(path.name)
        if not match:
            raise RuntimeError(f"migration file name must look like 0001_name.sql: {path.name}")
        found.append((int(match.group(1)), path))
    found.sort()
    versions = [v for v, _ in found]
    if len(versions) != len(set(versions)):
        raise RuntimeError("duplicate migration version")
    return found


def _configure(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = ON")
    mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
    if str(mode).lower() != "wal":
        # e.g. a network filesystem; WAL is required for concurrent API + worker access (docs/ARCHITECTURE.md).
        raise RuntimeError(f"SQLite WAL mode unavailable (journal_mode={mode})")
    conn.execute("PRAGMA synchronous = NORMAL")


def apply_migrations(conn: sqlite3.Connection, directory: Path = MIGRATIONS_DIR) -> list[int]:
    """Apply pending migrations in version order. Safe to call concurrently from several processes."""
    applied_now: list[int] = []
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_utc TEXT NOT NULL)"
    )
    for version, path in discover_migrations(directory):
        done = conn.execute("SELECT 1 FROM schema_migrations WHERE version = ?", (version,)).fetchone()
        if done:
            continue
        sql = path.read_text(encoding="utf-8")
        # executescript() commits any open transaction first, so the BEGIN IMMEDIATE/COMMIT pair lives inside the
        # script: the migration and its schema_migrations row commit together or not at all.
        script = (
            "BEGIN IMMEDIATE;\n"
            f"{sql}\n;\n"
            f"INSERT OR IGNORE INTO schema_migrations (version, applied_utc) VALUES ({version}, '{now_db()}');\n"
            "COMMIT;"
        )
        try:
            conn.executescript(script)
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        applied_now.append(version)
        logger.info("migration applied", extra={"code": "migration_applied", "version": version})
    return applied_now


class Database:
    """Lazily opened SQLite database with one connection per thread."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._local = threading.local()
        self._migrate_lock = threading.Lock()
        self._migrated = False
        self._all: list[sqlite3.Connection] = []
        self._all_lock = threading.Lock()

    def _open(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: autocommit; transactions are explicit (see transaction()).
        conn = sqlite3.connect(
            str(self.path), timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None, check_same_thread=False
        )
        _configure(conn)
        with self._all_lock:
            self._all.append(conn)
        return conn

    def conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._open()
            self._local.conn = conn
        if not self._migrated:
            with self._migrate_lock:
                if not self._migrated:
                    apply_migrations(conn)
                    self._migrated = True
        return conn

    def migrate(self) -> list[int]:
        with self._migrate_lock:
            conn = getattr(self._local, "conn", None) or self._open()
            self._local.conn = conn
            applied = apply_migrations(conn)
            self._migrated = True
            return applied

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """``BEGIN IMMEDIATE`` ... ``COMMIT`` (rollback on error): read-check-write under the write lock."""
        conn = self.conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()

    def execute(self, sql: str, params: tuple[object, ...] | dict[str, object] = ()) -> sqlite3.Cursor:
        return self.conn().execute(sql, params)

    def check(self) -> None:
        """Readiness probe: the file opens, migrations are applied and a trivial query works."""
        conn = self.conn()
        conn.execute("SELECT 1").fetchone()
        latest = max((v for v, _ in discover_migrations()), default=0)
        row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
        if (row[0] or 0) < latest:
            raise RuntimeError("database schema is behind the application")

    def close(self) -> None:
        with self._all_lock:
            conns, self._all = self._all, []
        for conn in conns:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        self._local = threading.local()
        self._migrated = False


_databases: dict[str, Database] = {}
_databases_lock = threading.Lock()


def get_database(path: Path | str) -> Database:
    """Process-wide Database for a path (shared by API threads / worker code in one process)."""
    key = str(Path(path))
    with _databases_lock:
        db = _databases.get(key)
        if db is None:
            db = Database(path)
            _databases[key] = db
        return db
