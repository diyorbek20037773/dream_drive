"""One-shot model bootstrap: ``python -m app.model_init``.

For every required model (YOLOPv2 and RF-DETR Nano, docs/IMPLEMENTATION_SPEC.md section 2.1) downloads the
manifest's checkpoint_url into MODEL_ROOT/<file_name>.part with connect/read/total timeouts, verifies size + SHA256
(and the publisher MD5 when the manifest records one) while streaming, then atomically renames it into place.
The exit code is the most severe outcome over all models (3 > 2 > 1 > 0).

Exit codes: 0 verified file in place (also when it was already present: no network access in that case),
1 download/network/filesystem error, 2 size or SHA256 mismatch (the .part file is deleted),
3 manifest invalid.
Interrupted downloads are resumed: a network error keeps MODEL_ROOT/<file_name>.part, and the next attempt (in the same
run, up to MODEL_INIT_ATTEMPTS, or a later run) asks for the remaining bytes with an HTTP Range request. The digest
always covers the complete file, so a resumed file is verified exactly like a fresh one; a server that ignores Range
restarts the download from byte 0.
Proxies are not used (plain direct HTTPS connection); redirects are followed (max 5, never https->http).
"""

from __future__ import annotations

import hashlib
import http.client
import logging
import os
import ssl
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

from app.logging_setup import configure_logging
from app.manifest import ManifestError, ModelManifest, load_manifest, md5_file, model_path, verify_model_file
from app.settings import Settings

logger = logging.getLogger("app.model_init")

EXIT_OK, EXIT_DOWNLOAD_ERROR, EXIT_MISMATCH, EXIT_MANIFEST = 0, 1, 2, 3
MAX_REDIRECTS = 5
CHUNK_BYTES = 1024 * 1024


class DownloadError(Exception):
    pass


class MismatchError(Exception):
    pass


def _open(
    url: str, connect_timeout: float, read_timeout: float, start: int = 0
) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    """GET with explicit connect and per-read socket timeouts, following redirects manually.

    ``start > 0`` sends ``Range: bytes=<start>-``; the caller checks whether the answer is 206 (resume) or 200 (full).
    """
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        parts = urlsplit(current)
        if parts.scheme not in ("https", "http") or not parts.hostname:
            raise DownloadError(f"unsupported URL scheme for {parts.hostname or 'unknown host'}")
        conn: http.client.HTTPConnection
        if parts.scheme == "https":
            conn = http.client.HTTPSConnection(
                parts.hostname, parts.port, timeout=connect_timeout, context=ssl.create_default_context()
            )
        else:
            conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=connect_timeout)
        try:
            conn.connect()
            if conn.sock is not None:
                conn.sock.settimeout(read_timeout)
            target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
            headers = {"User-Agent": "dreamdrive-model-init/1.0", "Accept": "application/octet-stream"}
            if start > 0:
                headers["Range"] = f"bytes={start}-"
            conn.request("GET", target, headers=headers)
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            conn.close()
            raise DownloadError(f"request to {parts.hostname} failed: {type(exc).__name__}: {exc}") from exc
        if resp.status in (301, 302, 303, 307, 308):
            location = resp.getheader("Location")
            resp.close()
            conn.close()
            if not location:
                raise DownloadError(f"redirect from {parts.hostname} without Location")
            nxt = urljoin(current, location)
            if parts.scheme == "https" and urlsplit(nxt).scheme != "https":
                raise DownloadError("refusing https -> http redirect")
            # Only host names are logged: release-asset redirect URLs carry signed query strings.
            logger.info("following redirect", extra={"from_host": parts.hostname, "to_host": urlsplit(nxt).hostname})
            current = nxt
            continue
        if resp.status not in (200, 206) or (resp.status == 206 and start == 0):
            resp.close()
            conn.close()
            raise DownloadError(f"HTTP {resp.status} from {parts.hostname}")
        return conn, resp
    raise DownloadError("too many redirects")


def _hash_existing(part: Path, digest: Any, md5: Any) -> int:
    """Feed an existing partial file into the digests; returns its size."""
    size = 0
    with part.open("rb") as fh:
        while chunk := fh.read(CHUNK_BYTES):
            digest.update(chunk)
            if md5 is not None:
                md5.update(chunk)
            size += len(chunk)
    return size


def _attempt(manifest: ModelManifest, part: Path, settings: Settings, deadline: float) -> None:
    """One download attempt into ``part`` (resuming it when present). Raises DownloadError/MismatchError."""
    existing = part.stat().st_size if part.exists() else 0
    if existing > manifest.size_bytes:
        part.unlink()
        existing = 0
    conn, resp = _open(
        manifest.checkpoint_url, settings.model_init_connect_timeout_s, settings.model_init_read_timeout_s, existing
    )
    try:
        if existing and resp.status == 206:
            content_range = resp.getheader("Content-Range") or ""
            if not content_range.startswith(f"bytes {existing}-"):
                raise DownloadError(f"unexpected Content-Range for resume: {content_range!r}")
            mode = "ab"
            logger.info("resuming download", extra={"from_bytes": existing, "expected_bytes": manifest.size_bytes})
        else:
            if existing:
                logger.info("server ignored the Range request; restarting from byte 0")
            mode = "wb"
            declared = resp.getheader("Content-Length")
            if declared is not None and declared.isdigit() and int(declared) != manifest.size_bytes:
                raise MismatchError(f"server Content-Length {declared} != manifest size {manifest.size_bytes}")
        digest = hashlib.sha256()
        md5 = hashlib.md5(usedforsecurity=False) if manifest.checkpoint_md5 else None
        received = _hash_existing(part, digest, md5) if mode == "ab" else 0
        next_report = (int(10 * received / manifest.size_bytes) + 1) / 10
        with part.open(mode) as fh:
            while True:
                if time.monotonic() > deadline:
                    raise DownloadError(f"total download timeout {settings.model_init_total_timeout_s:.0f}s exceeded")
                try:
                    chunk = resp.read(CHUNK_BYTES)
                except (OSError, http.client.HTTPException) as exc:
                    raise DownloadError(f"read failed after {received} bytes: {type(exc).__name__}") from exc
                if not chunk:
                    break
                received += len(chunk)
                if received > manifest.size_bytes:
                    raise MismatchError(f"download exceeds manifest size {manifest.size_bytes} bytes")
                digest.update(chunk)
                if md5 is not None:
                    md5.update(chunk)
                fh.write(chunk)
                if received / manifest.size_bytes >= next_report:
                    logger.info("download progress", extra={"percent": int(100 * received / manifest.size_bytes)})
                    next_report += 0.1
            fh.flush()
            os.fsync(fh.fileno())
        if received < manifest.size_bytes:
            raise DownloadError(f"connection closed after {received} of {manifest.size_bytes} bytes")
        actual = digest.hexdigest()
        if actual != manifest.sha256:
            raise MismatchError(f"sha256 {actual} != manifest {manifest.sha256}")
        if md5 is not None and md5.hexdigest() != manifest.checkpoint_md5:
            raise MismatchError(f"md5 {md5.hexdigest()} != publisher md5 {manifest.checkpoint_md5}")
    finally:
        resp.close()
        conn.close()


def download_verified(manifest: ModelManifest, target: Path, settings: Settings) -> None:
    """Download (resuming <target>.part), verify, fsync and atomically rename. Raises DownloadError/MismatchError.

    A size/hash mismatch deletes the partial file; a network error keeps it so a later attempt can resume.
    """
    part = target.with_name(target.name + ".part")
    deadline = time.monotonic() + settings.model_init_total_timeout_s
    logger.info(
        "downloading checkpoint",
        extra={"host": urlsplit(manifest.checkpoint_url).hostname, "expected_bytes": manifest.size_bytes},
    )
    attempts = max(1, settings.model_init_attempts)
    for attempt in range(1, attempts + 1):
        try:
            _attempt(manifest, part, settings, deadline)
            break
        except MismatchError:
            part.unlink(missing_ok=True)
            raise
        except DownloadError as exc:
            kept = part.stat().st_size if part.exists() else 0
            if attempt == attempts or time.monotonic() > deadline:
                logger.warning("download interrupted; partial file kept for resume", extra={"bytes_kept": kept})
                raise
            backoff = min(30.0, settings.model_init_retry_backoff_s * 2 ** (attempt - 1))
            logger.warning(
                "download interrupted; retrying with resume",
                extra={"attempt": attempt, "bytes_kept": kept, "detail": str(exc), "backoff_s": backoff},
            )
            time.sleep(backoff)
    os.replace(part, target)
    _fsync_dir(target.parent)
    logger.info(
        "checkpoint downloaded and verified",
        extra={"file": target.name, "sha256": manifest.sha256, "bytes": manifest.size_bytes},
    )


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return  # not supported on this platform/filesystem; the rename itself is still atomic
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def run(settings: Settings, model_ids: Sequence[str] | None = None) -> int:
    """Bootstrap every required model (default: YOLOPv2 and RF-DETR Nano); returns the most severe exit code."""
    ids = list(model_ids) if model_ids is not None else [settings.model_id, settings.rfdetr_model_id]
    codes = [run_one(settings, model_id) for model_id in ids]
    for code in (EXIT_MANIFEST, EXIT_MISMATCH, EXIT_DOWNLOAD_ERROR):
        if code in codes:
            return code
    return EXIT_OK


def run_one(settings: Settings, model_id: str) -> int:
    try:
        manifest = load_manifest(settings.model_manifest_path, model_id)
    except ManifestError as exc:
        logger.error("manifest invalid", extra={"code": "manifest_invalid", "detail": str(exc), "model": model_id})
        return EXIT_MANIFEST
    target = model_path(settings.model_root, manifest)
    logger.info(
        "model-init start",
        extra={"model": manifest.model_id, "file": manifest.file_name, "expected_sha256": manifest.sha256},
    )

    if target.exists():
        result = verify_model_file(target, manifest)
        if result.ok and manifest.checkpoint_md5 and md5_file(target) != manifest.checkpoint_md5:
            logger.error(
                "checkpoint SHA256 matches but publisher MD5 does not: manifest inconsistent",
                extra={"code": "manifest_invalid", "file": target.name},
            )
            return EXIT_MANIFEST
        if result.ok:
            logger.info(
                "checkpoint already present and verified; no download needed",
                extra={"file": target.name, "sha256": result.actual_sha256},
            )
            return EXIT_OK
        # The verified download below replaces it atomically; the bad file is never loaded by the service.
        logger.warning(
            "existing checkpoint failed verification; re-downloading",
            extra={"code": result.reason, "detail": result.detail},
        )

    try:
        settings.model_root.mkdir(parents=True, exist_ok=True)
        download_verified(manifest, target, settings)
    except MismatchError as exc:
        logger.error(
            "checkpoint verification FAILED; partial file deleted", extra={"code": "hash_mismatch", "detail": str(exc)}
        )
        return EXIT_MISMATCH
    except (DownloadError, OSError) as exc:
        logger.error("checkpoint download FAILED", extra={"code": "download_failed", "detail": str(exc)})
        return EXIT_DOWNLOAD_ERROR
    return EXIT_OK


def main() -> int:
    settings = Settings()
    configure_logging(settings.log_level, component="model-init")
    return run(settings)


if __name__ == "__main__":
    sys.exit(main())
