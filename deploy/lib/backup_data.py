"""DreamDrive data backup/restore, executed INSIDE the backend image as uid 10001 (stdlib only).

deploy/backup.sh and deploy/restore-test.sh pass this file's source with ``python -c`` so that stdin/stdout stay
free for the archive stream:

    backup   -> stdout: tar.gz with a coherent SQLite copy (sqlite3 online backup API, never a raw copy of a
                live WAL database), artifacts/, evaluations/ and backup-manifest.json (written last)
    restore  -> stdin:  that tar.gz, extracted into an EMPTY data volume, then verified
    verify   -> checks the live data volume the same way restore does (integrity + counts)

Summaries go to stderr (backup) or stdout (restore/verify) as JSON. No file contents or secrets are printed.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

FORMAT = "dreamdrive-backup/1"
MANIFEST_NAME = "backup-manifest.json"
DB_NAME = "dreamdrive.db"
TREES = ("artifacts", "evaluations")

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data"))
SQLITE_PATH = Path(os.environ.get("SQLITE_PATH", str(DATA_ROOT / DB_NAME)))
ARTIFACT_ROOT = Path(os.environ.get("ARTIFACT_ROOT", str(DATA_ROOT / "artifacts")))
TREE_ROOTS = {"artifacts": ARTIFACT_ROOT, "evaluations": DATA_ROOT / "evaluations"}


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    names = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    # Table names come from sqlite_master of our own database and are quoted as identifiers (not user input).
    return {n: int(conn.execute(f'SELECT COUNT(*) FROM "{n.replace(chr(34), chr(34) * 2)}"').fetchone()[0])  # noqa: S608
            for n in names}


def integrity(conn: sqlite3.Connection) -> str:
    rows = [r[0] for r in conn.execute("PRAGMA integrity_check")]
    return "ok" if rows == ["ok"] else "; ".join(rows[:5])


def snapshot_database(dest: Path) -> dict[str, Any]:
    """Online backup of the live WAL database into a standalone rollback-journal file."""
    src = sqlite3.connect(str(SQLITE_PATH), timeout=60)
    try:
        src.execute("PRAGMA busy_timeout=60000")
        dst = sqlite3.connect(str(dest))
        try:
            src.backup(dst)
            dst.execute("PRAGMA journal_mode=DELETE")  # self-contained file, no -wal/-shm companions
            info: dict[str, Any] = {"integrity": integrity(dst), "tables": table_counts(dst)}
        finally:
            dst.close()
    finally:
        src.close()
    info["bytes"] = dest.stat().st_size
    info["sha256"] = sha256_file(dest)
    return info


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def add_tree(tar: tarfile.TarFile, label: str, root: Path, stats: dict[str, Any]) -> None:
    """Regular files and directories only; symlinks/devices are skipped and counted, vanished files tolerated."""
    counts = {"files": 0, "bytes": 0, "skipped": 0}
    stats[label] = counts
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        rel_dir = Path(dirpath).relative_to(root)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            arcname = str(PurePosixPath(label, *rel_dir.parts, name))
            try:
                st = path.lstat()
                if not path.is_file() or path.is_symlink():
                    counts["skipped"] += 1
                    continue
                with path.open("rb") as fh:
                    info = tar.gettarinfo(arcname=arcname, fileobj=fh)
                    info.uid = info.gid = 10001
                    info.uname = info.gname = ""
                    tar.addfile(info, fh)
                counts["files"] += 1
                counts["bytes"] += st.st_size
            except FileNotFoundError:
                counts["skipped"] += 1  # removed by retention cleanup while the backup ran


def cmd_backup(args: argparse.Namespace) -> int:
    if not SQLITE_PATH.is_file():
        eprint(json.dumps({"ok": False, "error": "database_missing", "path": str(SQLITE_PATH)}))
        return 3
    started = time.monotonic()
    out = sys.stdout.buffer
    with tempfile.TemporaryDirectory(prefix="dd-backup-") as tmp:
        db_copy = Path(tmp) / DB_NAME
        db_info = snapshot_database(db_copy)
        if db_info["integrity"] != "ok":
            eprint(json.dumps({"ok": False, "error": "integrity_check_failed", "detail": db_info["integrity"]}))
            return 4
        stats: dict[str, Any] = {}
        with tarfile.open(fileobj=out, mode="w|gz", format=tarfile.PAX_FORMAT) as tar:
            tar.add(str(db_copy), arcname=DB_NAME, recursive=False)
            for label in TREES:
                add_tree(tar, label, TREE_ROOTS[label], stats)
            manifest = {
                "format": FORMAT,
                "created_utc": utc_now(),
                "sqlite": db_info,
                "files": stats,
                "source_image": {
                    "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
                    "build_version": os.environ.get("BUILD_VERSION", "unknown"),
                },
                "release": json.loads(args.release_json) if args.release_json else None,
                "elapsed_s": round(time.monotonic() - started, 3),
            }
            payload = json.dumps(manifest, indent=2, sort_keys=True).encode()
            info = tarfile.TarInfo(MANIFEST_NAME)
            info.size = len(payload)
            info.mtime = int(time.time())
            info.mode = 0o640
            tar.addfile(info, io.BytesIO(payload))
    out.flush()
    eprint(json.dumps({"ok": True, "manifest": manifest}, sort_keys=True))
    return 0


def _destination(name: str) -> Path | None:
    """Map an archive member to its target path; None for the manifest. Raises on anything unexpected."""
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe member name: {name!r}")
    if name == MANIFEST_NAME:
        return None
    if name == DB_NAME:
        return SQLITE_PATH
    top, *rest = pure.parts
    if top not in TREE_ROOTS or not rest:
        raise ValueError(f"unexpected member: {name!r}")
    root = TREE_ROOTS[top].resolve()
    dest = root.joinpath(*rest).resolve()
    if root not in dest.parents:
        raise ValueError(f"member escapes {top}: {name!r}")
    return dest


def verify_tree(label: str) -> dict[str, int]:
    root = TREE_ROOTS[label]
    files = [p for p in root.rglob("*") if p.is_file()] if root.is_dir() else []
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def verify_data(manifest: dict[str, Any] | None, require_job: bool) -> dict[str, Any]:
    problems: list[str] = []
    result: dict[str, Any] = {"database": str(SQLITE_PATH)}
    if not SQLITE_PATH.is_file():
        return {**result, "ok": False, "problems": ["database_missing"]}
    conn = sqlite3.connect(str(SQLITE_PATH))
    try:
        result["integrity"] = integrity(conn)
        result["tables"] = table_counts(conn)
        if result["integrity"] != "ok":
            problems.append("integrity_check_failed")
        jobs_total = result["tables"].get("jobs")
        if jobs_total is None:
            problems.append("jobs_table_missing")
        succeeded: list[str] = []
        if jobs_total:
            succeeded = [r[0] for r in conn.execute("SELECT id FROM jobs WHERE state='succeeded' ORDER BY id")]
    finally:
        conn.close()
    with_report = [j for j in succeeded if (ARTIFACT_ROOT / "jobs" / j / "report.json").is_file()]
    result["jobs_succeeded"] = len(succeeded)
    result["jobs_succeeded_with_report"] = len(with_report)
    if require_job:
        if not jobs_total:
            problems.append("no_job_rows")
        if succeeded and not with_report:
            problems.append("no_report_for_succeeded_jobs")
    result["trees"] = {label: verify_tree(label) for label in TREES}
    if manifest is not None:
        expected = manifest.get("sqlite", {}).get("tables", {})
        if expected != result.get("tables"):
            problems.append("table_counts_differ_from_manifest")
        for label in TREES:
            want = manifest.get("files", {}).get(label, {}).get("files")
            if want is not None and want != result["trees"][label]["files"]:
                problems.append(f"{label}_file_count_differs_from_manifest")
    result["problems"] = problems
    result["ok"] = not problems
    return result


def cmd_restore(args: argparse.Namespace) -> int:
    if SQLITE_PATH.exists() and not args.force:
        print(json.dumps({"ok": False, "error": "target_not_empty", "path": str(SQLITE_PATH)}))
        return 5
    manifest: dict[str, Any] | None = None
    restored = 0
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz") as tar:
        for member in tar:
            dest = _destination(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"refusing non-regular member: {member.name!r}")
            src = tar.extractfile(member)
            if src is None:
                raise ValueError(f"unreadable member: {member.name!r}")
            if dest is None:
                manifest = json.loads(src.read().decode())
                continue
            dest.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
            if dest == SQLITE_PATH:
                # A stale WAL/SHM next to a replaced database would be replayed into it.
                for companion in (Path(f"{SQLITE_PATH}-wal"), Path(f"{SQLITE_PATH}-shm")):
                    companion.unlink(missing_ok=True)
            tmp = dest.with_name(dest.name + ".restore-part")
            with tmp.open("wb") as fh:
                shutil.copyfileobj(src, fh, 1 << 20)
            os.chmod(tmp, 0o640)
            os.replace(tmp, dest)
            restored += 1
    if manifest is None or manifest.get("format") != FORMAT:
        print(json.dumps({"ok": False, "error": "manifest_missing_or_unknown_format", "restored_files": restored}))
        return 6
    result = verify_data(manifest, args.require_job)
    result["restored_files"] = restored
    result["backup_created_utc"] = manifest.get("created_utc")
    result["backup_release"] = manifest.get("release")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 7


def cmd_verify(args: argparse.Namespace) -> int:
    result = verify_data(None, args.require_job)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 7


def main() -> int:
    parser = argparse.ArgumentParser(prog="backup_data")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backup")
    b.add_argument("--release-json", default=os.environ.get("DD_BACKUP_RELEASE_JSON", ""))
    r = sub.add_parser("restore")
    r.add_argument("--require-job", action="store_true")
    r.add_argument("--force", action="store_true", help="allow restoring over an existing database")
    v = sub.add_parser("verify")
    v.add_argument("--require-job", action="store_true")
    args = parser.parse_args()
    return {"backup": cmd_backup, "restore": cmd_restore, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
