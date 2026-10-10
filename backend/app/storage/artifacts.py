"""Artifact storage: UUID-named directories, allowlisted names, atomic writes and byte-range responses.

Layout under ``ARTIFACT_ROOT``::

    jobs/<job_id>/input.<mp4|mov>      uploaded bytes (never the client file name)
    jobs/<job_id>/annotated.mp4 ...    pipeline outputs, published atomically by app.video.pipeline
    runs/<run_id>/report.json          simulation report

Clients only ever name an artifact from the allowlist; the path is derived from the server-generated id, and every
resolved path is checked to stay inside the artifact root.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi.responses import Response, StreamingResponse

SERVER_ID_RE = re.compile(r"^[a-f0-9]{32}$")
CHUNK_BYTES = 256 * 1024

# Allowlisted artifact names -> (content type, inline?). `input` maps to input.<ext>.
ARTIFACT_TYPES: dict[str, tuple[str, bool]] = {
    "annotated.mp4": ("video/mp4", True),
    "timeline.json": ("application/json", False),
    "report.json": ("application/json", False),
    "report.html": ("text/html; charset=utf-8", False),
    "poster.jpg": ("image/jpeg", True),
}
INPUT_TYPES = {"mp4": "video/mp4", "mov": "video/quicktime"}
# Order used in JobResults.artifacts.
RESULT_ARTIFACTS = ("annotated.mp4", "poster.jpg", "timeline.json", "report.json", "report.html", "input")


def new_id() -> str:
    return uuid.uuid4().hex


def _checked_child(root: Path, *parts: str) -> Path:
    root_resolved = root.resolve()
    path = root_resolved.joinpath(*parts).resolve()
    if path != root_resolved and root_resolved not in path.parents:
        raise ValueError("path escapes the artifact root")
    return path


def job_dir(artifact_root: Path, job_id: str) -> Path:
    if not SERVER_ID_RE.match(job_id):
        raise ValueError("invalid job id")
    return _checked_child(artifact_root, "jobs", job_id)


def run_dir(artifact_root: Path, run_id: str) -> Path:
    if not SERVER_ID_RE.match(run_id):
        raise ValueError("invalid run id")
    return _checked_child(artifact_root, "runs", run_id)


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    name: str
    path: Path
    content_type: str
    inline: bool
    size: int


def artifact_file(artifact_root: Path, job_id: str, name: str, input_ext: str) -> ArtifactFile | None:
    """Resolve an allowlisted artifact of a job; None when the name is not allowed or the file does not exist."""
    base = job_dir(artifact_root, job_id)
    if name == "input":
        if input_ext not in INPUT_TYPES:
            return None
        filename, content_type, inline = f"input.{input_ext}", INPUT_TYPES[input_ext], True
    elif name in ARTIFACT_TYPES:
        filename = name
        content_type, inline = ARTIFACT_TYPES[name]
    else:
        return None
    path = _checked_child(base, filename)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    if not path.is_file():
        return None
    return ArtifactFile(name=name, path=path, content_type=content_type, inline=inline, size=stat.st_size)


def dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += (Path(dirpath) / f).stat().st_size
            except OSError:
                pass
    return total


def remove_tree(path: Path) -> int:
    """Delete a directory tree; returns the bytes it held (0 if it did not exist)."""
    if not path.exists():
        return 0
    size = dir_size(path)
    shutil.rmtree(path, ignore_errors=True)
    return size


def remove_job_artifacts(artifact_root: Path, job_id: str) -> int:
    """Remove the job directory and any partial output directories the pipeline left next to it."""
    removed = remove_tree(job_dir(artifact_root, job_id))
    jobs_root = artifact_root / "jobs"
    if jobs_root.is_dir():
        for partial in jobs_root.glob(f".{job_id}.partial-*"):
            removed += remove_tree(partial)
    return removed


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_bytes(path, (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


# ---------------------------------------------------------------------------------------------------------------------
# HTTP byte ranges (RFC 9110 section 14): single range per request; multi-range requests get the full body (200).
# ---------------------------------------------------------------------------------------------------------------------
class RangeNotSatisfiable(Exception):  # noqa: N818
    def __init__(self, size: int) -> None:
        super().__init__(f"range not satisfiable for {size} bytes")
        self.size = size


_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Return the inclusive (start, end) to serve, None for the full body. Raises RangeNotSatisfiable."""
    if not header:
        return None
    header = header.strip()
    if "," in header:
        return None  # multiple ranges: serving the whole representation is allowed
    match = _RANGE_RE.match(header)
    if not match:
        return None  # syntactically invalid Range headers are ignored
    first, last = match.group(1), match.group(2)
    if first == "" and last == "":
        return None
    if first == "":
        suffix = int(last)
        if suffix == 0 or size == 0:
            raise RangeNotSatisfiable(size)
        return max(0, size - suffix), size - 1
    start = int(first)
    if start >= size:
        raise RangeNotSatisfiable(size)
    end = size - 1 if last == "" else min(int(last), size - 1)
    if end < start:
        return None  # last-pos < first-pos is invalid syntax: ignore the header
    return start, end


def _iter_file(path: Path, start: int, length: int) -> Iterator[bytes]:
    with path.open("rb") as fh:
        fh.seek(start)
        remaining = length
        while remaining > 0:
            chunk = fh.read(min(CHUNK_BYTES, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def file_response(
    artifact: ArtifactFile, range_header: str | None, download_name: str, extra_headers: dict[str, str] | None = None
) -> Response:
    """Stream an artifact with Range support. Raises RangeNotSatisfiable for 416."""
    size = artifact.size
    disposition = "inline" if artifact.inline else "attachment"
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f'{disposition}; filename="{download_name}"',
        "Cache-Control": "private, no-cache",
        "X-Content-Type-Options": "nosniff",
        **(extra_headers or {}),
    }
    byte_range = parse_range(range_header, size)
    if byte_range is None:
        headers["Content-Length"] = str(size)
        return StreamingResponse(
            _iter_file(artifact.path, 0, size), status_code=200, media_type=artifact.content_type, headers=headers
        )
    start, end = byte_range
    length = end - start + 1
    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(length)
    return StreamingResponse(
        _iter_file(artifact.path, start, length), status_code=206, media_type=artifact.content_type, headers=headers
    )
