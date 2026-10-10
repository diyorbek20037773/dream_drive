"""Perception engine: YOLOPv2 (TorchScript) + RF-DETR Nano, verified load, device/precision, warm-up and one
serialised two-model pass per frame (docs/IMPLEMENTATION_SPEC.md sections 2 and 2.1).

There is no fallback or mock path: if the manifest, either checkpoint, the device or a graph check fails the engine
stays ``ready=False`` with a reason, and every inference request is refused. Both models are required; there is no
silent single-model mode.
"""

from __future__ import annotations

import io
import logging
import threading
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from app import metrics
from app.ensemble import ENSEMBLE_ID, refine_vehicle_classes
from app.ensemble import ensemble_sha256 as compute_ensemble_sha256
from app.manifest import INPUT_POLICY, ManifestError, ModelManifest, load_manifest, model_path, read_verified_bytes
from app.postprocess import Detection, decode_detections, invert_drivable, invert_lane, to_detections
from app.preprocess import LetterboxGeometry
from app.rfdetr_detector import Builder, RfDetrDetector
from app.settings import Settings

logger = logging.getLogger(__name__)

WARMUP_SHAPE = (1, 3, 384, 640)  # canvas of a 16:9 frame, the most common input
NOT_READY_REASONS = (
    "loading",
    "manifest_invalid",
    "model_missing",
    "hash_mismatch",
    "load_failed",
    "device_unavailable",
)


class ModelNotReady(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class InferenceBusy(RuntimeError):
    pass


class OutputFormatError(RuntimeError):
    """The loaded graph does not produce the YOLOPv2 output structure this service decodes."""


@dataclass(frozen=True)
class EngineOutput:
    detections: list[Detection]  # both models after class refinement, sorted by score
    drivable: np.ndarray  # bool HxW, original image size
    lane: np.ndarray  # bool HxW, original image size
    inference_ms: float  # sum over models (YOLOPv2 forward; RF-DETR predict incl. its resize/normalise/top-k)
    postprocess_ms: float  # YOLOPv2 decode + NMS + mask inversion, ensemble refinement
    inference_ms_by_model: dict[str, float]


def unpack_outputs(out: Any) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Validate and split ``[pred, anchor_grid], seg, ll`` exactly as the upstream demo unpacks it."""
    if not isinstance(out, tuple | list) or len(out) != 3:
        raise OutputFormatError("expected 3 outputs ([pred, anchor_grid], seg, ll)")
    det, seg, ll = out
    if not isinstance(det, tuple | list) or len(det) != 2:
        raise OutputFormatError("expected detection output [pred, anchor_grid]")
    pred, anchor_grid = det
    if (
        not isinstance(pred, Sequence)
        or not isinstance(anchor_grid, Sequence)
        or len(pred) != 3
        or len(anchor_grid) != 3
    ):
        raise OutputFormatError("expected 3 detection heads and 3 anchor grids")
    for p in pred:
        if not isinstance(p, torch.Tensor) or p.ndim != 4 or p.shape[1] != 255:
            raise OutputFormatError("detection head must be [B, 255, H, W] (3 anchors x 85)")
    for a in anchor_grid:
        if not isinstance(a, torch.Tensor) or tuple(a.shape) != (1, 3, 1, 1, 2):
            raise OutputFormatError("anchor grid must be [1, 3, 1, 1, 2]")
    if not isinstance(seg, torch.Tensor) or seg.ndim != 4 or seg.shape[1] != 2:
        raise OutputFormatError("drivable output must be [B, 2, H, W]")
    if not isinstance(ll, torch.Tensor) or ll.ndim != 4 or ll.shape[1] != 1:
        raise OutputFormatError("lane output must be [B, 1, H, W]")
    return list(pred), list(anchor_grid), seg, ll


class YolopV2Engine:
    """The service engine. Named after its primary model for compatibility; it owns both detectors."""

    def __init__(self, settings: Settings, rfdetr_builder: Builder | None = None) -> None:
        self.settings = settings
        self.rfdetr = RfDetrDetector(settings, rfdetr_builder)
        self._yolo_ready = False
        self._yolo_reason: str | None = "loading"
        self.device_name = settings.inference_device
        self.precision = settings.inference_precision
        self.manifest: ModelManifest | None = None
        self._model: Any = None
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._ready = False
        self._reason: str | None = "loading"
        self._detail = "model load not started"
        self._settled = threading.Event()
        self._load_started = False
        self._forward_lock = threading.Lock()
        self._admission_lock = threading.Lock()
        self._in_flight = 0
        self.verified_sha256: str | None = None
        self.verified_size: int | None = None
        self.timings_s: dict[str, float] = {}
        self.accelerator_name: str | None = None
        metrics.MODEL_READY.labels(settings.model_id, self.device_name).set(0)

    # ------------------------------------------------------------------ state
    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def reason(self) -> str | None:
        return self._reason

    @property
    def detail(self) -> str:
        return self._detail

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def wait_settled(self, timeout: float | None = None) -> bool:
        """Block until the load attempt finished (successfully or not)."""
        return self._settled.wait(timeout)

    def _fail(self, reason: str, detail: str) -> bool:
        self._ready = False
        self._reason = reason
        self._detail = detail
        self._model = None
        # Per-model states keep their own truth; only models that never got a verdict inherit this reason.
        if not self._yolo_ready and self._yolo_reason == "loading":
            self._yolo_reason = reason
        if not self.rfdetr.ready and self.rfdetr.reason == "loading":
            self.rfdetr.reason = reason
        metrics.MODEL_READY.labels(self.settings.model_id, self.device_name).set(0)
        logger.error("model not ready", extra={"code": reason, "detail": detail, "model": self.settings.model_id})
        self._settled.set()
        return False

    # ------------------------------------------------------------------ load
    def ensure_load_started(self) -> None:
        """Start the load in a background thread unless a load already ran or is running."""
        with self._admission_lock:
            if self._load_started:
                return
            self._load_started = True
        threading.Thread(target=self._load_once, name="model-load", daemon=True).start()

    def load(self) -> bool:
        """Synchronous load (used by model_init-style callers and tests)."""
        with self._admission_lock:
            if self._load_started:
                raise RuntimeError("model load already started")
            self._load_started = True
        return self._load_once()

    def _load_once(self) -> bool:
        """Verify and load the checkpoint, then warm up. Never raises; returns readiness."""
        try:
            return self._load()
        except Exception as exc:  # noqa: BLE001 - defensive: an unexpected bug must still settle as not ready
            logger.exception("unexpected model load error")
            return self._fail("load_failed", f"unexpected load error: {type(exc).__name__}")

    def _load(self) -> bool:
        t_start = time.perf_counter()
        self._detail = "loading"
        try:
            self.manifest = load_manifest(self.settings.model_manifest_path, self.settings.model_id)
        except ManifestError as exc:
            self.rfdetr.load_manifest()
            return self._fail("manifest_invalid", f"{self.settings.model_id}: {exc}")
        if not self.rfdetr.load_manifest():
            reason = self.rfdetr.reason or "manifest_invalid"
            return self._fail(reason, f"{self.rfdetr.model_id}: {self.rfdetr.detail}")

        if self.precision == "fp16" and self.device_name != "cuda":
            return self._fail("device_unavailable", "INFERENCE_PRECISION=fp16 requires INFERENCE_DEVICE=cuda")
        if self.device_name == "cuda":
            if not torch.cuda.is_available():
                return self._fail("device_unavailable", "INFERENCE_DEVICE=cuda but torch reports no CUDA device")
            self._device = torch.device("cuda:0")
            self.accelerator_name = torch.cuda.get_device_name(0)
        else:
            self._device = torch.device("cpu")
        self._dtype = torch.float16 if self.precision == "fp16" else torch.float32
        if self.settings.inference_torch_threads > 0:
            torch.set_num_threads(self.settings.inference_torch_threads)

        model = self._load_yolopv2()
        # RF-DETR is loaded even when YOLOPv2 failed, so /v1/model reports the state of both checkpoints at once.
        rf_ok = self.rfdetr.load("cuda" if self._device.type == "cuda" else "cpu")
        if model is None:
            yolo_reason, yolo_detail = self._yolo_reason or "load_failed", self._detail
            return self._fail(yolo_reason, f"{self.settings.model_id}: {yolo_detail}")
        if not rf_ok:
            return self._fail(self.rfdetr.reason or "load_failed", f"{self.rfdetr.model_id}: {self.rfdetr.detail}")
        self.timings_s["rfdetr_total"] = self.rfdetr.timings_s.get("total", 0.0)
        self.timings_s["total"] = time.perf_counter() - t_start

        self._model = model
        self._ready = True
        self._reason = None
        self._detail = "ready"
        metrics.MODEL_READY.labels(self.settings.model_id, self.device_name).set(1)
        metrics.MODEL_LOAD_SECONDS.labels(self.settings.model_id, self.device_name).set(self.timings_s["yolopv2_total"])
        logger.info(
            "models ready",
            extra={
                "model": ENSEMBLE_ID,
                "device": self.device_name,
                "precision": self.precision,
                "rfdetr_precision": self.rfdetr.precision,
                "ensemble_sha256": self.ensemble_sha256(),
                "torch_threads": torch.get_num_threads(),
                "load_seconds": round(self.timings_s["total"], 3),
            },
        )
        self._settled.set()
        return True

    def _yolo_fail(self, reason: str, detail: str) -> None:
        self._yolo_ready = False
        self._yolo_reason = reason
        self._detail = detail
        logger.error("model not ready", extra={"code": reason, "detail": detail, "model": self.settings.model_id})

    def _load_yolopv2(self) -> Any:
        """Verify + jit.load + warm-up YOLOPv2. Returns the module or None (state recorded, never raises)."""
        assert self.manifest is not None
        t_start = time.perf_counter()
        path = model_path(self.settings.model_root, self.manifest)
        t0 = time.perf_counter()
        data, verify = read_verified_bytes(path, self.manifest)
        self.timings_s["verify"] = time.perf_counter() - t0
        if data is None or not verify.ok:
            self._yolo_fail(verify.reason or "hash_mismatch", verify.detail)
            return None
        self.verified_sha256, self.verified_size = verify.actual_sha256, verify.actual_size

        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                # torch 2.14 marks torch.jit.load deprecated (FutureWarning); the published checkpoint is
                # TorchScript only, so it is the supported loader here. Recorded in docs/MODEL_CARD.md.
                warnings.filterwarnings("ignore", message=r".*torch\.jit\.load.*deprecated", category=FutureWarning)
                model = torch.jit.load(io.BytesIO(data), map_location=self._device)
            model.eval()
            if self._dtype == torch.float16:
                model.half()
        except Exception as exc:  # noqa: BLE001 - any loader failure means "not loadable", reported, not raised
            logger.exception("torch.jit.load failed")
            self._yolo_fail("load_failed", f"torch.jit.load failed: {type(exc).__name__}")
            return None
        finally:
            del data
        self.timings_s["jit_load"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                x = torch.zeros(WARMUP_SHAPE, dtype=self._dtype, device=self._device)
                for _ in range(max(1, self.settings.inference_warmup_runs)):
                    _, _, seg, ll = unpack_outputs(model(x))
                    if tuple(seg.shape[-2:]) != WARMUP_SHAPE[-2:] or tuple(ll.shape[-2:]) != WARMUP_SHAPE[-2:]:
                        raise OutputFormatError("segmentation outputs do not match the input canvas size")
                self._synchronize()
        except torch.cuda.OutOfMemoryError:
            logger.exception("warm-up ran out of GPU memory")
            self._yolo_fail("device_unavailable", "CUDA out of memory during warm-up")
            return None
        except Exception as exc:  # noqa: BLE001
            logger.exception("warm-up failed")
            self._yolo_fail("load_failed", f"warm-up forward pass failed: {type(exc).__name__}: {exc}"[:300])
            return None
        self.timings_s["warmup"] = time.perf_counter() - t0
        self.timings_s["yolopv2_total"] = time.perf_counter() - t_start
        self._yolo_ready = True
        self._yolo_reason = None
        return model

    def _synchronize(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    # ------------------------------------------------------------------ inference
    def infer(
        self,
        image_input: np.ndarray,
        geometry: LetterboxGeometry,
        conf_thres: float,
        iou_thres: float,
        rgb_image: np.ndarray,
    ) -> EngineOutput:
        """Run both models on one frame: YOLOPv2 on the preprocessed 1x3xHxW float32 letterbox canvas, RF-DETR on the
        same decoded frame as RGB uint8 HxWx3 (its own resize/normalisation); boxes come back in original pixels."""
        if not self._ready or self._model is None or self.manifest is None:
            raise ModelNotReady(self._reason or "loading")
        with self._admission_lock:
            if self._in_flight >= 1 + self.settings.inference_max_waiters:
                raise InferenceBusy("too many concurrent inference requests")
            self._in_flight += 1
        try:
            if not self._forward_lock.acquire(timeout=self.settings.inference_lock_timeout_s):
                raise InferenceBusy("timed out waiting for the model lock")
            try:
                return self._infer_locked(image_input, geometry, conf_thres, iou_thres, rgb_image)
            finally:
                self._forward_lock.release()
        finally:
            with self._admission_lock:
                self._in_flight -= 1

    def _infer_locked(
        self,
        image_input: np.ndarray,
        geometry: LetterboxGeometry,
        conf_thres: float,
        iou_thres: float,
        rgb_image: np.ndarray,
    ) -> EngineOutput:
        assert self.manifest is not None
        if rgb_image.shape[:2] != (geometry.image_height, geometry.image_width):
            raise ValueError("rgb_image size does not match the letterbox geometry")
        hist = metrics.INFERENCE_DURATION
        yolo_id, rf_id = self.settings.model_id, self.rfdetr.model_id
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                x = torch.from_numpy(image_input).to(self._device, dtype=self._dtype)
                out = self._model(x)
                self._synchronize()
                t1 = time.perf_counter()
                pred, anchor_grid, seg, ll = unpack_outputs(out)
                det = decode_detections(pred, anchor_grid, conf_thres, iou_thres)
                yolo_dets = to_detections(det, geometry, self.manifest.class_map)
                drivable = invert_drivable(seg, geometry)
                lane = invert_lane(ll, geometry)
                self._synchronize()
            t2 = time.perf_counter()
        except Exception:
            hist.labels(yolo_id, self.device_name, "error").observe(time.perf_counter() - t0)
            raise
        hist.labels(yolo_id, self.device_name, "ok").observe(t2 - t0)
        try:
            rf_dets = self.rfdetr.predict(rgb_image, conf_thres)
            self._synchronize()
            t3 = time.perf_counter()
        except Exception:
            hist.labels(rf_id, self.device_name, "error").observe(time.perf_counter() - t2)
            raise
        hist.labels(rf_id, self.device_name, "ok").observe(t3 - t2)
        detections = refine_vehicle_classes(yolo_dets, rf_dets)
        t4 = time.perf_counter()
        yolo_ms, rf_ms = (t1 - t0) * 1000.0, (t3 - t2) * 1000.0
        return EngineOutput(
            detections=detections,
            drivable=drivable,
            lane=lane,
            inference_ms=yolo_ms + rf_ms,
            postprocess_ms=((t2 - t1) + (t4 - t3)) * 1000.0,
            inference_ms_by_model={yolo_id: round(yolo_ms, 3), rf_id: round(rf_ms, 3)},
        )

    # ------------------------------------------------------------------ descriptions
    def identity(self) -> dict[str, str]:
        """Identity of the primary model (YOLOPv2), kept as ``InferenceResult.model`` for compatibility."""
        assert self.verified_sha256 is not None
        return {
            "model_id": self.settings.model_id,
            "sha256": self.verified_sha256,
            "device": self.device_name,
            "precision": self.precision,
        }

    def identities(self) -> list[dict[str, str]]:
        """Every model that runs on each frame, in ensemble order (YOLOPv2, RF-DETR)."""
        assert self.rfdetr.verified_sha256 is not None
        return [
            self.identity(),
            {
                "model_id": self.rfdetr.model_id,
                "sha256": self.rfdetr.verified_sha256,
                "device": self.device_name,
                "precision": self.rfdetr.precision,
            },
        ]

    def ensemble_sha256(self) -> str | None:
        if not (self._ready and self.verified_sha256 and self.rfdetr.verified_sha256):
            return None
        return compute_ensemble_sha256(
            [(self.settings.model_id, self.verified_sha256), (self.rfdetr.model_id, self.rfdetr.verified_sha256)]
        )

    def _yolo_describe(self) -> dict[str, Any]:
        m = self.manifest
        ok = self._ready or self._yolo_ready
        return {
            "model_id": self.settings.model_id,
            "ready": ok,
            "reason": None if ok else self._yolo_reason,
            "sha256": self.verified_sha256 if ok else None,
            "size_bytes": self.verified_size if ok else None,
            "repository": m.repository if m else None,
            "repository_revision": m.repository_revision if m else None,
            "class_map": m.class_map_wire() if (m and ok) else {},
            "device": self.device_name,
            "precision": self.precision if ok else None,
            "input_policy": INPUT_POLICY,
        }

    def model_info(self) -> dict[str, Any]:
        """ModelInfo wire shape. Top-level fields describe the primary model (YOLOPv2) and the overall readiness;
        ``class_map`` is the union of every enabled class of both models (what /v1/infer can return).
        Checkpoint identity/class maps are only reported for verified, loaded models."""
        m = self.manifest
        class_map: dict[str, str] = {}
        if self._ready:
            if m:
                class_map.update(m.class_map_wire())
            if self.rfdetr.manifest:
                class_map.update(self.rfdetr.manifest.class_map_wire())
        return {
            "model_id": self.settings.model_id,
            "ready": self._ready,
            "reason": self._reason,
            "sha256": self.verified_sha256 if self._ready else None,
            "size_bytes": self.verified_size if self._ready else None,
            "repository_revision": m.repository_revision if m else None,
            "class_map": dict(sorted(class_map.items(), key=lambda kv: int(kv[0]))),
            "device": self.device_name,
            "precision": self.precision,
            "torch_version": torch.__version__,
            "input_policy": INPUT_POLICY,
            "ensemble_id": ENSEMBLE_ID,
            "ensemble_sha256": self.ensemble_sha256(),
            "models": [self._yolo_describe(), self.rfdetr.describe(self.device_name)],
        }

    def status(self) -> dict[str, Any]:
        """Readiness payload (superset of the spec's {"ready", "reason"})."""
        body: dict[str, Any] = {
            "ready": self._ready,
            "reason": self._reason,
            "model_id": self.settings.model_id,
            "device": self.device_name,
            "precision": self.precision,
        }
        body["models"] = {
            self.settings.model_id: {"ready": self._ready or self._yolo_ready, "reason": self._yolo_reason},
            self.rfdetr.model_id: {"ready": self.rfdetr.ready, "reason": self.rfdetr.reason},
        }
        if self._ready:
            body.update(
                sha256=self.verified_sha256,
                ensemble_id=ENSEMBLE_ID,
                ensemble_sha256=self.ensemble_sha256(),
                rfdetr_precision=self.rfdetr.precision,
                rfdetr_precision_note=self.rfdetr.precision_note,
                torch_version=torch.__version__,
                torch_threads=torch.get_num_threads(),
                accelerator=self.accelerator_name,
                load_seconds={k: round(v, 3) for k, v in self.timings_s.items()},
                rfdetr_load_seconds={k: round(v, 3) for k, v in self.rfdetr.timings_s.items()},
            )
        else:
            body["detail"] = self._detail
        return body
