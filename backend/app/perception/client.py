"""HTTP client for the internal inference service (docs/IMPLEMENTATION_SPEC.md section 2).

The client never fabricates results: every failure raises a typed `InferenceClientError` subclass and every success
is validated into `app.schemas.inference` models before it is returned.

Error mapping
- connection refused / DNS / transport errors, HTTP 503 `model_not_ready`  -> `InferenceUnavailable`
- HTTP 503 `inference_busy` (service waiter limit)                         -> retried with bounded exponential
  backoff, then `InferenceBusy`
- client-side timeout (connect/read/write/pool)                            -> `InferenceTimeout`
- any other non-2xx, invalid JSON or schema mismatch                       -> `InferenceError`

Two-model ensemble (docs/IMPLEMENTATION_SPEC.md section 2.1, YOLOPv2 + RF-DETR-N): the additive wire fields
`InferenceResult.models`, `ensemble_id`, `ensemble_sha256`, `RawDetection.model_id`, `RawDetection.refined_from_model`
and `ModelInfo.models/ensemble_id/ensemble_sha256` are checked for internal consistency (`validate_ensemble`): every
model id is unique, the primary `model` is listed, `ensemble_sha256` equals sha256 of the `"<model_id>:<sha256>\\n"`
lines in `models` order, and every detection names a model that ran. A response with an empty `models` list is a
single-model (legacy) response. `perception_identity()` and `merged_class_map()` are the only places that decide which
identity a frame's perception carries and which class names the service can return.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import ValidationError

from app.schemas.inference import InferenceResult, ModelIdentity, ModelInfo


def ensemble_sha256(models: Sequence[tuple[str, str]]) -> str:
    """sha256 of the UTF-8 text `"<model_id>:<sha256>\\n"` per model, in the given order (spec section 2.1)."""
    text = "".join(f"{model_id}:{sha}\n" for model_id, sha in models)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PerceptionIdentity:
    """Identity a frame's perception is tagged with: the ensemble when >= 2 models ran, else the single model."""

    model_id: str
    sha256: str
    device: str
    precision: str
    models: tuple[ModelIdentity, ...]
    is_ensemble: bool

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(m.model_id for m in self.models)


def perception_identity(result: InferenceResult) -> PerceptionIdentity:
    """Ensemble id/sha256 when the service reports two or more models, else the single-model identity."""
    m = result.model
    if len(result.models) >= 2 and result.ensemble_id and result.ensemble_sha256:
        return PerceptionIdentity(
            result.ensemble_id, result.ensemble_sha256, m.device, m.precision, tuple(result.models), True
        )
    return PerceptionIdentity(m.model_id, m.sha256, m.device, m.precision, (m,), False)


def merged_class_map(info: ModelInfo) -> dict[str, str]:
    """Union of the top-level class map and every `ModelInfo.models[*].class_map` (ids are disjoint by design:
    YOLOPv2 uses its own ids, RF-DETR uses 100 + COCO id). A class id mapped to two different names is rejected."""
    merged: dict[str, str] = dict(info.class_map)
    for entry in info.models:
        for class_id, name in entry.class_map.items():
            if merged.get(class_id, name) != name:
                raise InferenceError("invalid_response", f"class id {class_id} maps to two names across models")
            merged[class_id] = name
    return merged


def validate_ensemble(result: InferenceResult) -> None:
    """Internal consistency of the spec 2.1 fields; raises InferenceError('invalid_response') on any mismatch."""

    def bad(msg: str) -> InferenceError:
        return InferenceError("invalid_response", msg)

    models = result.models
    if not models:
        if result.ensemble_id is not None or result.ensemble_sha256 is not None:
            raise bad("ensemble identity without a models list")
        for det in result.detections:
            if det.model_id is not None and det.model_id != result.model.model_id:
                raise bad("detection model_id names a model that did not run")
            if det.refined_from_model is not None:
                raise bad("refined_from_model without a models list")
        return
    ids = [m.model_id for m in models]
    if len(set(ids)) != len(ids):
        raise bad("duplicate model ids in models")
    if not any(m.model_id == result.model.model_id and m.sha256 == result.model.sha256 for m in models):
        raise bad("primary model identity is not listed in models")
    expected = ensemble_sha256([(m.model_id, m.sha256) for m in models])
    if len(models) >= 2:
        if not result.ensemble_id or not result.ensemble_sha256:
            raise bad("multi-model result without ensemble_id/ensemble_sha256")
        if result.ensemble_sha256 != expected:
            raise bad("ensemble_sha256 does not match the listed models")
    elif result.ensemble_sha256 is not None and result.ensemble_sha256 != expected:
        raise bad("ensemble_sha256 does not match the listed model")
    known = set(ids)
    for det in result.detections:
        if det.model_id is not None and det.model_id not in known:
            raise bad(f"detection model_id {det.model_id!r} did not run")
        if det.refined_from_model is not None and det.refined_from_model not in known:
            raise bad(f"refined_from_model {det.refined_from_model!r} did not run")


class InferenceClientError(Exception):
    """Base class; `code` is a stable machine code (mirrors the service's error code when one was sent)."""

    def __init__(self, code: str, message: str = "", status_code: int | None = None) -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message or code
        self.status_code = status_code


class InferenceUnavailable(InferenceClientError):  # noqa: N818 - names fixed by the implementation spec
    """Service unreachable or model not ready (no verified weights loaded)."""


class InferenceBusy(InferenceClientError):  # noqa: N818
    """Service rejected the request because its bounded waiter queue was full, after client retries."""


class InferenceTimeout(InferenceClientError):  # noqa: N818
    """The request did not complete within the client timeout."""


class InferenceError(InferenceClientError):
    """Any other failure: rejected image, unexpected status, invalid or inconsistent response."""


def _error_code(response: httpx.Response) -> tuple[str | None, str]:
    """Extract (code, message) from an error body.

    Accepts the DreamDrive envelope `{"code","message"}`, FastAPI's `{"detail": {...}|str}` and readiness bodies
    `{"ready": false, "reason": ...}` so the client does not depend on one framework's error rendering.
    """
    try:
        body: Any = response.json()
    except ValueError:
        return None, response.text[:200]
    if not isinstance(body, dict):
        return None, ""
    detail = body.get("detail")
    if isinstance(detail, dict):
        body = {**body, **detail}
    elif isinstance(detail, str) and "code" not in body:
        return detail, detail
    code = body.get("code") or body.get("error") or body.get("reason")
    message = body.get("message") or ""
    return (str(code) if code else None), str(message)[:500]


class InferenceClient:
    """Synchronous client; one instance per worker job/session. Thread-compatible (httpx.Client is)."""

    def __init__(
        self,
        base_url: str,
        timeout_s: float,
        *,
        busy_retries: int = 3,
        busy_backoff_s: float = 0.25,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if busy_retries < 0:
            raise ValueError("busy_retries must be >= 0")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.busy_retries = busy_retries
        self.busy_backoff_s = busy_backoff_s
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout_s, connect=min(5.0, self.timeout_s)),
            transport=transport,
            headers={"User-Agent": "dreamdrive-backend/inference-client"},
        )

    # -- lifecycle -----------------------------------------------------------------------------------------------
    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> InferenceClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- transport -----------------------------------------------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._http.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise InferenceTimeout("inference_timeout", f"{method} {path} exceeded {self.timeout_s:g} s") from exc
        except httpx.TransportError as exc:
            raise InferenceUnavailable("inference_unreachable", f"{type(exc).__name__} for {method} {path}") from exc

    # -- API -----------------------------------------------------------------------------------------------------
    def ready(self) -> bool:
        """True only when `/health/ready` answers 200 with `ready: true`. Never raises for an unreachable service."""
        try:
            response = self._request("GET", "/health/ready")
        except InferenceClientError:
            return False
        if response.status_code != 200:
            return False
        try:
            body = response.json()
        except ValueError:
            return False
        return isinstance(body, dict) and body.get("ready") is True

    def model_info(self) -> ModelInfo:
        response = self._request("GET", "/v1/model")
        if response.status_code != 200:
            code, message = _error_code(response)
            raise InferenceError(code or f"http_{response.status_code}", message, response.status_code)
        try:
            info = ModelInfo.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise InferenceError("invalid_response", "model info failed schema validation") from exc
        merged_class_map(info)  # rejects conflicting class maps early
        if len(info.models) >= 2 and info.ensemble_sha256 is not None and all(m.sha256 for m in info.models):
            entries = [(m.model_id, m.sha256 or "") for m in info.models]
            if info.ensemble_sha256 != ensemble_sha256(entries):
                raise InferenceError("invalid_response", "model info ensemble_sha256 does not match its models")
        return info

    def infer(
        self,
        image_bytes: bytes,
        frame_id: int,
        content_type: str = "image/jpeg",
        conf_threshold: float = 0.1,
        include_masks: bool = True,
    ) -> InferenceResult:
        if content_type not in ("image/jpeg", "image/png"):
            raise ValueError("content_type must be image/jpeg or image/png")
        if frame_id < 0:
            raise ValueError("frame_id must be >= 0")
        params = {
            "frame_id": str(frame_id),
            "conf_threshold": f"{conf_threshold:g}",
            "include_masks": "true" if include_masks else "false",
        }
        attempt = 0
        while True:
            response = self._request(
                "POST", "/v1/infer", params=params, content=image_bytes, headers={"Content-Type": content_type}
            )
            if response.status_code == 200:
                return self._parse_result(response, frame_id, include_masks)
            code, message = _error_code(response)
            if response.status_code == 503 and code == "inference_busy":
                if attempt >= self.busy_retries:
                    raise InferenceBusy("inference_busy", message or "inference service busy", 503)
                # Bounded exponential backoff: 0.25, 0.5, 1.0 s by default.
                self._sleep(self.busy_backoff_s * (2**attempt))
                attempt += 1
                continue
            if response.status_code == 503:
                raise InferenceUnavailable(code or "model_not_ready", message, 503)
            raise InferenceError(code or f"http_{response.status_code}", message, response.status_code)

    @staticmethod
    def _parse_result(response: httpx.Response, frame_id: int, include_masks: bool) -> InferenceResult:
        try:
            result = InferenceResult.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise InferenceError("invalid_response", "inference result failed schema validation") from exc
        validate_ensemble(result)
        if result.frame_id != frame_id:
            raise InferenceError("invalid_response", f"frame_id mismatch: sent {frame_id}, got {result.frame_id}")
        if include_masks and (result.drivable_mask_png is None or result.lane_mask_png is None):
            raise InferenceError("invalid_response", "masks requested but missing in response")
        width, height = result.image.width, result.image.height
        tol = _BOX_BOUNDS_TOLERANCE_PX
        for det in result.detections:
            x1, y1, x2, y2 = det.xyxy_pixels
            if not (-tol <= x1 <= x2 <= width + tol and -tol <= y1 <= y2 <= height + tol):
                raise InferenceError("invalid_response", "detection box outside image bounds or unordered")
        return result


# The service clips boxes to the image; allow float rounding but reject boxes that are clearly in another frame.
_BOX_BOUNDS_TOLERANCE_PX = 1.0
