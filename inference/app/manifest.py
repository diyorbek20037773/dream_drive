"""Resolved model manifest loading and checkpoint verification (size + SHA256, streaming).

The manifest (`config/models.resolved.yaml`) is JSON-compatible YAML resolved from the actual checkpoint bytes.
Only the file named by the manifest inside MODEL_ROOT is ever considered; nothing else is loaded.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_MD5_RE = re.compile(r"^[a-f0-9]{32}$")
_SAFE_FILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
CHUNK_BYTES = 1024 * 1024
INPUT_POLICY = "letterbox640_stride32"


class ManifestError(Exception):
    """The manifest is missing, unreadable or does not describe a verifiable model."""


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    file_name: str
    sha256: str
    size_bytes: int
    repository: str
    repository_revision: str
    checkpoint_url: str
    model_format: str
    class_map: dict[int, str] = field(default_factory=dict)
    input_policy: str = INPUT_POLICY
    # Optional publisher checksum (RF-DETR publishes MD5 in its package registry); verified by model_init.
    checkpoint_md5: str | None = None
    # DreamDrive class id = class_id_offset + source label id (RF-DETR: 100 + COCO 2017 category id).
    class_id_offset: int = 0
    source_label_space: str | None = None

    def class_map_wire(self) -> dict[str, str]:
        return {str(k): v for k, v in sorted(self.class_map.items())}


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    reason: str | None  # None | "model_missing" | "hash_mismatch"
    detail: str
    actual_sha256: str | None = None
    actual_size: int | None = None
    actual_md5: str | None = None  # only computed when requested and the manifest records a publisher MD5


def _require_str(entry: dict[str, Any], key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"manifest field '{key}' must be a non-empty string")
    return value


def load_manifest(path: Path, model_id: str) -> ModelManifest:
    """Load and validate one model entry. Raises ManifestError with a non-sensitive message."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest file not found: {path.name}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestError(f"manifest unreadable: {type(exc).__name__}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
        raise ManifestError("manifest must be a mapping with a 'models' list")
    entries = [m for m in raw["models"] if isinstance(m, dict) and m.get("id") == model_id]
    if len(entries) != 1:
        raise ManifestError(f"manifest must contain exactly one model with id '{model_id}'")
    entry = entries[0]

    sha256 = _require_str(entry, "checkpoint_sha256").lower()
    if not _SHA256_RE.match(sha256):
        raise ManifestError("checkpoint_sha256 must be 64 lowercase hex characters")
    size = entry.get("checkpoint_size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ManifestError("checkpoint_size_bytes must be a positive integer")
    file_name = _require_str(entry, "file_name")
    if not _SAFE_FILE_NAME_RE.match(file_name) or ".." in file_name:
        raise ManifestError("file_name must be a plain file name")

    class_map_raw = entry.get("supported_class_map")
    if not isinstance(class_map_raw, dict) or not class_map_raw:
        raise ManifestError("supported_class_map must be a non-empty mapping")
    class_map: dict[int, str] = {}
    for key, name in class_map_raw.items():
        try:
            class_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ManifestError("supported_class_map keys must be integers") from exc
        if class_id < 0 or not isinstance(name, str) or not name:
            raise ManifestError("supported_class_map entries must map ids >= 0 to names")
        class_map[class_id] = name

    md5 = entry.get("checkpoint_md5_publisher")
    if md5 is not None and (not isinstance(md5, str) or not _MD5_RE.match(md5)):
        raise ManifestError("checkpoint_md5_publisher must be 32 lowercase hex characters")
    offset = entry.get("class_id_offset", 0)
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ManifestError("class_id_offset must be a non-negative integer")
    if any(class_id < offset for class_id in class_map):
        raise ManifestError("supported_class_map ids must be >= class_id_offset")
    label_space = entry.get("source_label_space")
    if label_space is not None and (not isinstance(label_space, str) or not label_space):
        raise ManifestError("source_label_space must be a non-empty string")
    input_policy = entry.get("input_policy_id", INPUT_POLICY)
    if not isinstance(input_policy, str) or not input_policy:
        raise ManifestError("input_policy_id must be a non-empty string")

    return ModelManifest(
        model_id=model_id,
        file_name=file_name,
        sha256=sha256,
        size_bytes=size,
        repository=_require_str(entry, "repository"),
        repository_revision=_require_str(entry, "repository_revision"),
        checkpoint_url=_require_str(entry, "checkpoint_url"),
        model_format=_require_str(entry, "format"),
        class_map=class_map,
        input_policy=input_policy,
        checkpoint_md5=md5,
        class_id_offset=offset,
        source_label_space=label_space,
    )


def model_path(model_root: Path, manifest: ModelManifest) -> Path:
    """The single allowlisted checkpoint location: MODEL_ROOT/<file_name>."""
    return model_root / manifest.file_name


def md5_file(path: Path) -> str:
    """Streaming MD5 (only to compare with a publisher-published MD5; integrity is decided by SHA256)."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> tuple[str, int]:
    """Streaming SHA256 and byte count of a file."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def verify_model_file(path: Path, manifest: ModelManifest, check_md5: bool = False) -> VerifyResult:
    """Check existence, exact size and SHA256 without loading anything (plus the publisher MD5 in the same
    streaming pass when ``check_md5`` is set and the manifest records one)."""
    if not path.is_file():
        return VerifyResult(False, "model_missing", f"checkpoint {manifest.file_name} not found in model root")
    size = path.stat().st_size
    if size != manifest.size_bytes:
        return VerifyResult(
            False,
            "hash_mismatch",
            f"size {size} bytes != manifest {manifest.size_bytes} bytes",
            actual_size=size,
        )
    want_md5 = check_md5 and manifest.checkpoint_md5 is not None
    digest = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False) if want_md5 else None
    counted = 0
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_BYTES):
            digest.update(chunk)
            if md5 is not None:
                md5.update(chunk)
            counted += len(chunk)
    actual = digest.hexdigest()
    actual_md5 = md5.hexdigest() if md5 is not None else None
    if counted != manifest.size_bytes or actual != manifest.sha256:
        return VerifyResult(False, "hash_mismatch", f"sha256 {actual} != manifest {manifest.sha256}", actual, counted)
    if want_md5 and actual_md5 != manifest.checkpoint_md5:
        return VerifyResult(
            False, "hash_mismatch", f"md5 {actual_md5} != publisher md5 {manifest.checkpoint_md5}", actual, counted
        )
    return VerifyResult(True, None, "verified", actual, counted, actual_md5)


def read_verified_bytes(path: Path, manifest: ModelManifest) -> tuple[bytes | None, VerifyResult]:
    """Read the checkpoint once (chunked), hashing as it streams, and return the bytes only if they verify.

    Loading from these exact bytes (instead of re-opening the path) removes the window in which the file
    could change between verification and torch.jit.load.
    """
    if not path.is_file():
        return None, VerifyResult(False, "model_missing", f"checkpoint {manifest.file_name} not found in model root")
    size = path.stat().st_size
    if size != manifest.size_bytes:
        return None, VerifyResult(
            False, "hash_mismatch", f"size {size} bytes != manifest {manifest.size_bytes} bytes", actual_size=size
        )
    digest = hashlib.sha256()
    buf = bytearray()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_BYTES):
            digest.update(chunk)
            buf.extend(chunk)
            if len(buf) > manifest.size_bytes:
                break
    actual = digest.hexdigest()
    if len(buf) != manifest.size_bytes or actual != manifest.sha256:
        return None, VerifyResult(
            False, "hash_mismatch", f"sha256 {actual} != manifest {manifest.sha256}", actual, len(buf)
        )
    return bytes(buf), VerifyResult(True, None, "verified", actual, len(buf))
