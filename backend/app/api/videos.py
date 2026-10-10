"""Video upload: validated multipart input becomes a durable queued job; no synchronous inference.

The handler parses the multipart stream itself instead of declaring an ``UploadFile`` parameter, because FastAPI
would buffer the whole body (to the system temp dir, without a size cap) *before* the session, CSRF, rate-limit,
model-readiness and queue checks run. Here those checks run first; then the single ``file`` part is streamed into
DATA_ROOT/tmp with a byte cap, hashed (SHA-256) on the fly and content-sniffed (ISO-BMFF ``ftyp`` brands), and only
then moved atomically into ARTIFACT_ROOT/jobs/<id>/input.<ext>. The client file name and Content-Type are ignored.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
import redis
from fastapi import APIRouter, Request
from python_multipart.multipart import MultipartParser, parse_options_header
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from app.api.convert import job_to_api
from app.api.deps import CsrfSessionDep, ExpensiveRateLimit, SettingsDep, get_services
from app.api.services import Services
from app.config import Settings
from app.db import utc_now
from app.errors import ApiError, error_responses
from app.jobs.queue import enqueue_video_job
from app.schemas.api import ErrorCode, Job
from app.storage import artifacts, repo

router = APIRouter(tags=["videos"])
logger = logging.getLogger("dreamdrive.api.videos")

UPLOAD_PATH = "/videos"
UPLOAD_FORM_NAME = "Body_upload_video"
# OpenAPI component for the multipart body (registered by app.main._normalize_openapi).
UPLOAD_FORM_COMPONENT: dict[str, Any] = {
    "properties": {
        "file": {
            "contentMediaType": "application/octet-stream",
            "description": "MP4/MOV video; the content is validated, not the name or type.",
            "title": "File",
            "type": "string",
        }
    },
    "required": ["file"],
    "title": UPLOAD_FORM_NAME,
    "type": "object",
}
_UPLOAD_OPENAPI: dict[str, Any] = {
    "requestBody": {
        "content": {"multipart/form-data": {"schema": {"$ref": f"#/components/schemas/{UPLOAD_FORM_NAME}"}}},
        "required": True,
    }
}

# Multipart framing, other small fields and headers on top of the file bytes.
MULTIPART_OVERHEAD_BYTES = 1024 * 1024
# Refuse uploads when the data volume would be left with less than this after storing a maximal upload.
STORAGE_RESERVE_BYTES = 256 * 1024 * 1024
SNIFF_BYTES = 64

# ISO-BMFF brands accepted as MP4 (ISO base media / MPEG-4 / Apple M4V / DASH). QuickTime is "qt  ".
MP4_BRANDS = frozenset(
    {
        b"isom",
        b"iso2",
        b"iso3",
        b"iso4",
        b"iso5",
        b"iso6",
        b"mp41",
        b"mp42",
        b"mp71",
        b"avc1",
        b"M4V ",
        b"M4VH",
        b"M4VP",
        b"MSNV",
        b"dash",
        b"mmp4",
    }
)
QT_BRAND = b"qt  "


def sniff_container(head: bytes) -> str | None:
    """Return "mp4" or "mov" when ``head`` starts with an ISO-BMFF ``ftyp`` box of an accepted brand."""
    if len(head) < 16 or head[4:8] != b"ftyp":
        return None
    size = int.from_bytes(head[0:4], "big")
    if size < 16 or size > 4096:
        return None
    major = head[8:12]
    compatible = {head[i : i + 4] for i in range(16, min(size, len(head)) - 3, 4)}
    if major == QT_BRAND:
        return "mov"
    if major in MP4_BRANDS or compatible & MP4_BRANDS:
        return "mp4"
    if QT_BRAND in compatible:
        return "mov"
    return None


class _UploadTooLarge(Exception):  # noqa: N818
    pass


class _NotAVideo(Exception):  # noqa: N818
    pass


@dataclass
class _Sink:
    """Receives multipart callbacks; writes the first ``file`` part to disk and discards everything else."""

    path: Path
    max_bytes: int
    fh: Any = None
    sha: Any = field(default_factory=hashlib.sha256)
    size: int = 0
    head: bytes = b""
    container: str | None = None
    got_file: bool = False
    _in_file: bool = False
    _field: bytearray = field(default_factory=bytearray)
    _value: bytearray = field(default_factory=bytearray)
    _headers: dict[str, str] = field(default_factory=dict)

    def callbacks(self) -> Any:  # python_multipart.MultipartCallbacks (a TypedDict)
        return {
            "on_part_begin": self._part_begin,
            "on_part_data": self._part_data,
            "on_part_end": self._part_end,
            "on_header_field": lambda d, s, e: self._field.extend(d[s:e]),
            "on_header_value": lambda d, s, e: self._value.extend(d[s:e]),
            "on_header_end": self._header_end,
            "on_headers_finished": self._headers_finished,
        }

    def _part_begin(self) -> None:
        self._headers = {}
        self._in_file = False

    def _header_end(self) -> None:
        self._headers[bytes(self._field).decode("latin-1").lower()] = bytes(self._value).decode("latin-1")
        self._field.clear()
        self._value.clear()

    def _headers_finished(self) -> None:
        disposition = self._headers.get("content-disposition", "")
        _kind, params = parse_options_header(disposition)
        name = params.get(b"name", b"").decode("latin-1")
        # Only the first part named "file" is the upload; its filename parameter is deliberately never read.
        if name == "file" and not self.got_file:
            self.got_file = True
            self._in_file = True
            self.fh = self.path.open("wb")

    def _part_data(self, data: bytes, start: int, end: int) -> None:
        if not self._in_file:
            return
        chunk = data[start:end]
        self.size += len(chunk)
        if self.size > self.max_bytes:
            raise _UploadTooLarge()
        if len(self.head) < SNIFF_BYTES:
            self.head += chunk[: SNIFF_BYTES - len(self.head)]
            if len(self.head) >= 16 and self.container is None:
                self.container = sniff_container(self.head)
                if self.container is None:
                    raise _NotAVideo()  # stop reading a non-video body early
        self.sha.update(chunk)
        self.fh.write(chunk)

    def _part_end(self) -> None:
        if self._in_file and self.fh is not None:
            self.fh.flush()
            os.fsync(self.fh.fileno())
            self.fh.close()
            self.fh = None
        self._in_file = False

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()
            self.fh = None


def _check_model_ready(services: Services) -> None:
    status = services.model_status.get()
    if not status.ready:
        raise ApiError(
            503,
            ErrorCode.MODEL_NOT_READY,
            "The inference model is not ready; video analysis is unavailable",
            {"reason": status.reason or "model_not_ready"},
            headers={"Retry-After": "30"},
        )


def _check_admission(services: Services, settings: Settings, owner_id: str) -> None:
    own = repo.count_jobs(services.db, repo.JOB_ACTIVE_STATES, owner_id)
    if own >= settings.max_queued_per_session:
        raise ApiError(
            429,
            ErrorCode.SESSION_QUEUE_FULL,
            "This session already has the maximum number of queued or running videos",
            {"limit": settings.max_queued_per_session, "active": own},
            headers={"Retry-After": "30"},
        )
    total = repo.count_jobs(services.db, repo.JOB_ACTIVE_STATES)
    if total >= settings.max_queued_total:
        raise ApiError(
            429,
            ErrorCode.QUEUE_FULL,
            "The analysis queue is full",
            {"limit": settings.max_queued_total, "active": total},
            headers={"Retry-After": "60"},
        )


def _check_storage(settings: Settings) -> None:
    settings.tmp_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(settings.tmp_root).free
    if free < settings.max_upload_bytes + STORAGE_RESERVE_BYTES:
        logger.error("upload refused: data volume low on space", extra={"code": "storage_low"})
        raise ApiError(503, ErrorCode.SERVICE_UNAVAILABLE, "Storage is temporarily full", {"reason": "storage_low"})


def _too_large(settings: Settings) -> ApiError:
    return ApiError(
        413,
        ErrorCode.UPLOAD_TOO_LARGE,
        "The upload exceeds the size limit",
        {"limit_bytes": settings.max_upload_bytes},
    )


async def _receive_upload(request: Request, settings: Settings, tmp_path: Path) -> _Sink:
    content_type = request.headers.get("content-type", "")
    kind, params = parse_options_header(content_type)
    boundary = params.get(b"boundary")
    if kind != b"multipart/form-data" or not boundary:
        raise ApiError(
            415, ErrorCode.UNSUPPORTED_MEDIA_TYPE, "Send the video as multipart/form-data with a 'file' field"
        )
    sink = _Sink(path=tmp_path, max_bytes=settings.max_upload_bytes)
    parser = MultipartParser(boundary, sink.callbacks())
    body_cap = settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES
    received = 0
    try:
        async for chunk in request.stream():
            if not chunk:
                continue
            received += len(chunk)
            if received > body_cap:
                raise _UploadTooLarge()
            # Parsing, hashing and writing happen off the event loop.
            await anyio.to_thread.run_sync(parser.write, chunk)
        await anyio.to_thread.run_sync(parser.finalize)
    except _UploadTooLarge as exc:
        raise _too_large(settings) from exc
    except _NotAVideo as exc:
        raise ApiError(415, ErrorCode.UNSUPPORTED_MEDIA_TYPE, "Only MP4 or MOV (ISO-BMFF) videos are accepted") from exc
    except ClientDisconnect as exc:
        raise ApiError(400, ErrorCode.HTTP_ERROR, "The upload was interrupted") from exc
    except ApiError:
        raise
    except Exception as exc:  # malformed multipart framing
        logger.info("malformed multipart upload", extra={"code": "validation_error"})
        raise ApiError(422, ErrorCode.VALIDATION_ERROR, "Malformed multipart body") from exc
    finally:
        sink.close()
    return sink


@router.post(
    UPLOAD_PATH,
    status_code=202,
    response_model=Job,
    summary="Upload a video for analysis",
    dependencies=[ExpensiveRateLimit],
    responses=error_responses(401, 403, 413, 415, 422, 429, 503),
    openapi_extra=_UPLOAD_OPENAPI,
)
async def upload_video(request: Request, session: CsrfSessionDep, settings: SettingsDep) -> Job:
    """202 with the queued job.

    Errors: 413 `upload_too_large`, 415 `unsupported_media_type`, 422 `upload_empty`/`video_*`,
    429 `rate_limited`/`session_queue_full`/`queue_full`, 503 `model_not_ready`.
    """
    services = get_services(request)
    declared = request.headers.get("content-length")
    if (
        declared is not None
        and declared.isdigit()
        and int(declared) > settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES
    ):
        raise _too_large(settings)
    # Cheap checks before a single body byte is read.
    await run_in_threadpool(_check_model_ready, services)
    await run_in_threadpool(_check_admission, services, settings, session.session_id)
    await run_in_threadpool(_check_storage, settings)

    fd, tmp_name = tempfile.mkstemp(prefix="upload-", suffix=".part", dir=settings.tmp_root)
    os.close(fd)
    tmp_path = Path(tmp_name)
    job_id = artifacts.new_id()
    out_dir: Path | None = None
    try:
        sink = await _receive_upload(request, settings, tmp_path)
        if not sink.got_file:
            raise ApiError(422, ErrorCode.VALIDATION_ERROR, "The multipart body has no 'file' part", {"field": "file"})
        if sink.size == 0:
            raise ApiError(422, ErrorCode.UPLOAD_EMPTY, "The uploaded file is empty")
        if sink.container is None:
            sink.container = sniff_container(sink.head)
        if sink.container is None:
            raise ApiError(415, ErrorCode.UNSUPPORTED_MEDIA_TYPE, "Only MP4 or MOV (ISO-BMFF) videos are accepted")
        ext = sink.container
        out_dir = artifacts.job_dir(settings.artifact_root, job_id)
        out_dir.mkdir(parents=True, exist_ok=False)
        os.replace(tmp_path, out_dir / f"input.{ext}")
        job_settings = {
            "analysis_fps": settings.analysis_fps,
            "inference_conf_threshold": settings.inference_conf_threshold,
            "max_video_seconds": settings.max_video_seconds,
            "max_video_width": settings.max_video_width,
            "max_video_height": settings.max_video_height,
        }
        try:
            row = await run_in_threadpool(
                lambda: repo.insert_job(
                    services.db,
                    job_id=job_id,
                    owner_id=session.session_id,
                    input_sha256=sink.sha.hexdigest(),
                    input_bytes=sink.size,
                    input_ext=ext,
                    settings=job_settings,
                    max_per_session=settings.max_queued_per_session,
                    max_total=settings.max_queued_total,
                )
            )
        except repo.AdmissionDenied as exc:
            raise ApiError(429, exc.code, exc.message, exc.details, headers={"Retry-After": "30"}) from exc
        out_dir = None  # owned by the job row from here on
        try:
            await run_in_threadpool(enqueue_video_job, services.redis, settings, job_id)
        except redis.RedisError as exc:
            await run_in_threadpool(
                lambda: repo.transition_job(
                    services.db,
                    job_id,
                    ("queued",),
                    "failed",
                    error_code="queue_unavailable",
                    error_message="The job queue is unavailable",
                    finished_utc=utc_now(),
                )
            )
            raise ApiError(503, ErrorCode.SERVICE_UNAVAILABLE, "The job queue is unavailable") from exc
        services.bus.publish(session.session_id, "job_state", job_id, {"job_id": job_id, "state": "queued"})
        logger.info("upload accepted", extra={"job_id": job_id, "code": "job_queued", "state": "queued"})
        return job_to_api(row, settings)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        if out_dir is not None:
            shutil.rmtree(out_dir, ignore_errors=True)
