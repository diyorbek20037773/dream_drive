"""Tests for deploy/lib/backup_data.py (stdlib unittest; run in the backend image or python:3.11-slim):

    make deploy-selftest
    # = docker run --rm -v "$PWD/deploy:/opt/dd/deploy:ro" -w /tmp python:3.11-slim-bookworm \
    #     python -m unittest discover -s /opt/dd/deploy/tests -v

The script is executed exactly as deploy/backup.sh runs it (``python -c <source> <command>``) against a WAL-mode
SQLite database that has an open writer, so the online-backup path is exercised for real.
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parents[1] / "lib" / "backup_data.py").read_text(encoding="utf-8")
JOB_OK = "a" * 32
JOB_FAILED = "b" * 32


def run(cmd: list[str], data_root: Path, stdin: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    env = {
        **os.environ,
        "DATA_ROOT": str(data_root),
        "SQLITE_PATH": str(data_root / "dreamdrive.db"),
        "ARTIFACT_ROOT": str(data_root / "artifacts"),
    }
    return subprocess.run([sys.executable, "-c", SCRIPT, *cmd], input=stdin, capture_output=True, env=env,
                          timeout=60, check=False)


def make_live_data(root: Path) -> sqlite3.Connection:
    (root / "artifacts" / "jobs" / JOB_OK).mkdir(parents=True)
    (root / "artifacts" / "jobs" / JOB_OK / "report.json").write_text('{"kind": "video_analysis"}')
    (root / "artifacts" / "jobs" / JOB_OK / "annotated.mp4").write_bytes(os.urandom(4096))
    (root / "evaluations").mkdir()
    (root / "evaluations" / "e1.json").write_text('{"id": "e1"}')
    (root / "tmp").mkdir()
    (root / "tmp" / "upload.part").write_bytes(b"in-flight upload, must not be backed up")
    conn = sqlite3.connect(str(root / "dreamdrive.db"))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE schema_migrations(version INT PRIMARY KEY, applied_utc TEXT)")
    conn.execute("CREATE TABLE jobs(id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, state TEXT)")
    conn.execute("INSERT INTO schema_migrations VALUES (1, '2026-10-10T00:00:00Z')")
    conn.execute("INSERT INTO jobs VALUES (?, 'owner1', 'succeeded')", (JOB_OK,))
    conn.execute("INSERT INTO jobs VALUES (?, 'owner1', 'failed')", (JOB_FAILED,))
    conn.commit()
    return conn  # kept open: the WAL file exists while the backup runs


class BackupRestoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.src = Path(self.tmp.name) / "src"
        self.dst = Path(self.tmp.name) / "dst"
        self.src.mkdir()
        self.dst.mkdir()
        self.conn = make_live_data(self.src)

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def backup(self) -> bytes:
        proc = run(["backup", "--release-json", '{"git_commit": "test"}'], self.src)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode())
        return proc.stdout

    def test_backup_contains_consistent_db_and_trees_but_not_tmp(self) -> None:
        # Uncommitted-to-checkpoint writes live only in the WAL; the online backup must include them.
        self.conn.execute("INSERT INTO jobs VALUES (?, 'owner2', 'queued')", ("c" * 32,))
        self.conn.commit()
        archive = self.backup()
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            names = tar.getnames()
            manifest = json.loads(tar.extractfile("backup-manifest.json").read())  # type: ignore[union-attr]
        self.assertEqual(names[0], "dreamdrive.db")
        self.assertEqual(names[-1], "backup-manifest.json")
        self.assertIn(f"artifacts/jobs/{JOB_OK}/report.json", names)
        self.assertIn("evaluations/e1.json", names)
        self.assertFalse(any(n.startswith("tmp/") for n in names))
        self.assertEqual(manifest["sqlite"]["integrity"], "ok")
        self.assertEqual(manifest["sqlite"]["tables"], {"jobs": 3, "schema_migrations": 1})
        self.assertEqual(manifest["files"]["artifacts"]["files"], 2)
        self.assertEqual(manifest["release"], {"git_commit": "test"})

    def test_restore_into_empty_volume_verifies_job_and_report(self) -> None:
        archive = self.backup()
        proc = run(["restore", "--require-job"], self.dst, stdin=archive)
        self.assertEqual(proc.returncode, 0, proc.stdout.decode() + proc.stderr.decode())
        result = json.loads(proc.stdout)
        self.assertTrue(result["ok"])
        self.assertEqual(result["tables"]["jobs"], 2)
        self.assertEqual(result["jobs_succeeded_with_report"], 1)
        self.assertEqual((self.dst / "artifacts" / "jobs" / JOB_OK / "report.json").read_text(),
                         '{"kind": "video_analysis"}')

    def test_restore_refuses_non_empty_target(self) -> None:
        archive = self.backup()
        (self.dst / "dreamdrive.db").write_bytes(b"existing")
        proc = run(["restore"], self.dst, stdin=archive)
        self.assertEqual(proc.returncode, 5)
        self.assertEqual((self.dst / "dreamdrive.db").read_bytes(), b"existing")

    def test_require_job_fails_without_rows(self) -> None:
        self.conn.execute("DELETE FROM jobs")
        self.conn.commit()
        proc = run(["restore", "--require-job"], self.dst, stdin=self.backup())
        self.assertEqual(proc.returncode, 7)
        self.assertIn("no_job_rows", json.loads(proc.stdout)["problems"])

    def test_restore_rejects_path_traversal(self) -> None:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            payload = b"evil"
            info = tarfile.TarInfo("artifacts/../../escape.txt")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        proc = run(["restore"], self.dst, stdin=buf.getvalue())
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(b"unsafe member name", proc.stderr)
        self.assertFalse((Path(self.tmp.name) / "escape.txt").exists())

    def test_backup_fails_cleanly_without_database(self) -> None:
        proc = run(["backup"], self.dst)
        self.assertEqual(proc.returncode, 3)
        self.assertEqual(proc.stdout, b"")


if __name__ == "__main__":
    unittest.main()
