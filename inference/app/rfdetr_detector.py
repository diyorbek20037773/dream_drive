"""RF-DETR Nano detector beside YOLOPv2 (docs/IMPLEMENTATION_SPEC.md section 2.1).

Load path: manifest entry ``rfdetr_nano`` -> streaming SHA256 + size verification of ``MODEL_ROOT/<file_name>`` ->
``rfdetr.RFDETRNano(pretrain_weights=<that verified local path>)`` inside :func:`offline_guard` -> label-space check
against rfdetr's own COCO class list -> ``inference(compile=False, dtype=...)`` -> warm-up with a finiteness check.

Offline: RF-DETR-N uses patch size 16 and 24 positional embeddings, so rfdetr builds its windowed DINOv2 backbone
from the JSON config bundled in the package and never calls ``from_pretrained`` (verified in rfdetr 1.11.2
``models/backbone/dinov2.py``); all weights come from the checkpoint. rfdetr's only download hook
(``rfdetr.assets.model_weights._download_file``) is replaced by a refusal while constructing, and the Hugging Face
offline switches are set before rfdetr/transformers are imported. Any attempted download is a ``load_failed``.
rfdetr also re-hashes the file with MD5 up to three times in 4 KiB reads (76 s over a Windows bind mount in the
spike); DreamDrive computes SHA256 and the publisher MD5 in one pass first and, while constructing, answers rfdetr's
MD5 check for exactly that verified path and MD5 from this result (any other path or MD5 goes to rfdetr's own check).

One label per query: RF-DETR scores classes with independent sigmoids and its top-300 selection can return the same
query box under several classes (e.g. car 0.80 and truck 0.15). DreamDrive keeps only the highest-scoring class per
identical box and then drops the box if that class is disabled, so a box RF-DETR considers a car never comes back
as a low-score "truck" (which would also wrongly refine a YOLOPv2 vehicle).

Class ids: for COCO-pretrained checkpoints rfdetr 1.11.2 ``predict()`` returns raw sparse COCO 2017 category ids
(1..90). This is checked at load with the same condition rfdetr uses (``args.num_classes`` > 80 and class names equal
to ``COCO_CLASS_NAMES``); the manifest's enabled classes are checked name-by-name against
``rfdetr.assets.coco_classes.COCO_CLASSES``. DreamDrive id = ``class_id_offset`` (100) + COCO category id.

There is no fallback output: any failure leaves the detector not ready with a reason.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from app import metrics
from app.manifest import ManifestError, ModelManifest, load_manifest, model_path, verify_model_file
from app.postprocess import Detection
from app.settings import Settings

# Set before rfdetr (and through it transformers / huggingface_hub) is imported anywhere in this process.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

logger = logging.getLogger(__name__)

RFDETR_INPUT_POLICY = "rfdetr_resize384_imagenet_norm"
COCO_LABEL_SPACE = "coco2017_category_id"
WARMUP_HW = (720, 1280)


class OfflineViolation(RuntimeError):
    """rfdetr attempted to download something while DreamDrive constructs it from a verified local checkpoint."""


class LabelSpaceError(RuntimeError):
    """The loaded checkpoint does not use the COCO category-id label space this service maps."""


class DetectorRuntime(Protocol):
    """What the service needs from a constructed detector (the real one wraps rfdetr; unit tests inject a stub)."""

    precision: str

    def coco_category_names(self) -> dict[int, str]: ...

    def predict(self, rgb: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """RGB uint8 HxWx3 -> (xyxy float [n,4] original pixels, scores [n], category ids [n])."""
        ...


# (verified checkpoint path, device, precision, verified publisher MD5 or None) -> runtime
Builder = Callable[[Path, str, str, str | None], DetectorRuntime]


@contextmanager
def offline_guard(verified: tuple[Path, str] | None = None) -> Iterator[None]:
    """Make every rfdetr weight download raise :class:`OfflineViolation` for the duration of the block.

    ``verified`` = (path, md5) of a file DreamDrive has just verified (SHA256 + publisher MD5): rfdetr's own MD5
    check of exactly that path against exactly that MD5 is answered from the verification instead of re-reading.
    """
    import rfdetr.assets.model_weights as model_weights

    original_download = model_weights._download_file
    original_md5 = model_weights._validate_file_md5

    def _refuse(*_args: Any, **_kwargs: Any) -> None:
        raise OfflineViolation(
            "rfdetr attempted a weight download; DreamDrive only loads the verified local checkpoint"
        )

    def _md5(filepath: str, expected_md5: str) -> bool:
        if verified is not None and Path(filepath) == verified[0] and expected_md5 == verified[1]:
            return True
        return bool(original_md5(filepath, expected_md5))

    model_weights._download_file = _refuse  # type: ignore[assignment]
    model_weights._validate_file_md5 = _md5  # type: ignore[assignment]
    try:
        yield
    finally:
        model_weights._download_file = original_download  # type: ignore[assignment]
        model_weights._validate_file_md5 = original_md5  # type: ignore[assignment]


class RfDetrNanoRuntime:
    """The real RF-DETR-N built by the rfdetr package from a verified local checkpoint path."""

    def __init__(self, path: Path, device: str, precision: str, verified_md5: str | None = None) -> None:
        with offline_guard((path, verified_md5) if verified_md5 else None):
            from rfdetr import RFDETRNano
            from rfdetr.assets.coco_classes import COCO_CLASS_NAMES, COCO_CLASSES

            model = RFDETRNano(pretrain_weights=str(path), device=device)
        args = getattr(model.model, "args", None)
        slots = getattr(args, "num_classes", None)
        if list(model.class_names) != list(COCO_CLASS_NAMES) or not isinstance(slots, int) or slots <= 80:
            raise LabelSpaceError(
                f"checkpoint label space is not COCO category ids (num_classes={slots}, "
                f"{len(model.class_names)} class names)"
            )
        self._coco = dict(COCO_CLASSES)
        dtype = torch.float16 if precision == "fp16" else torch.float32
        # Export-mode module without torch.jit/compile; in place so only one copy of the weights stays resident.
        model.inference(compile=False, dtype=dtype, inplace=True)
        self._model = model
        self.precision = precision

    def coco_category_names(self) -> dict[int, str]:
        return dict(self._coco)

    def predict(self, rgb: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from supervision import Detections

        det = self._model.predict(rgb, threshold=threshold, include_source_image=False)
        if not isinstance(det, Detections):  # one image in -> one Detections out for a detection checkpoint
            raise TypeError(f"unexpected rfdetr predict() result type {type(det).__name__}")
        xyxy = np.asarray(det.xyxy, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(det.confidence, dtype=np.float64).reshape(-1)
        class_ids = np.asarray(det.class_id, dtype=np.int64).reshape(-1)
        return xyxy, scores, class_ids


def build_rfdetr_nano(path: Path, device: str, precision: str, verified_md5: str | None = None) -> DetectorRuntime:
    return RfDetrNanoRuntime(path, device, precision, verified_md5)


class RfDetrDetector:
    """Verified RF-DETR-N with its own readiness; owned and serialised by the service engine (app.model)."""

    def __init__(self, settings: Settings, builder: Builder | None = None) -> None:
        self.settings = settings
        self.model_id = settings.rfdetr_model_id
        self._builder = builder or build_rfdetr_nano
        self.manifest: ModelManifest | None = None
        self.runtime: DetectorRuntime | None = None
        self.ready = False
        self.reason: str | None = "loading"
        self.detail = "model load not started"
        self.verified_sha256: str | None = None
        self.verified_size: int | None = None
        self.precision: str = settings.inference_precision
        self.precision_note: str | None = None
        self.timings_s: dict[str, float] = {}
        # COCO category id -> (DreamDrive class id, DreamDrive class name); enabled classes only.
        self._enabled: dict[int, tuple[int, str]] = {}
        metrics.MODEL_READY.labels(self.model_id, settings.inference_device).set(0)

    def _fail(self, reason: str, detail: str) -> bool:
        self.ready = False
        self.reason = reason
        self.detail = detail
        self.runtime = None
        metrics.MODEL_READY.labels(self.model_id, self.settings.inference_device).set(0)
        logger.error("model not ready", extra={"code": reason, "detail": detail, "model": self.model_id})
        return False

    def load_manifest(self) -> bool:
        try:
            self.manifest = load_manifest(self.settings.model_manifest_path, self.model_id)
        except ManifestError as exc:
            return self._fail("manifest_invalid", str(exc))
        if self.manifest.source_label_space != COCO_LABEL_SPACE:
            return self._fail("manifest_invalid", f"source_label_space must be '{COCO_LABEL_SPACE}'")
        return True

    def load(self, device: str) -> bool:
        """Verify, construct, check the label space and warm up. Never raises; returns readiness."""
        try:
            return self._load(device)
        except Exception as exc:  # noqa: BLE001 - an unexpected bug must still settle as not ready
            logger.exception("unexpected RF-DETR load error")
            return self._fail("load_failed", f"unexpected load error: {type(exc).__name__}")

    def _load(self, device: str) -> bool:
        t_start = time.perf_counter()
        if self.manifest is None and not self.load_manifest():
            return False
        assert self.manifest is not None
        path = model_path(self.settings.model_root, self.manifest)
        t0 = time.perf_counter()
        verify = verify_model_file(path, self.manifest, check_md5=True)
        self.timings_s["verify"] = time.perf_counter() - t0
        if not verify.ok:
            return self._fail(verify.reason or "hash_mismatch", verify.detail)
        self.verified_sha256, self.verified_size = verify.actual_sha256, verify.actual_size

        requested = self.settings.inference_precision
        attempts = ["fp16", "fp32"] if requested == "fp16" else ["fp32"]
        runtime: DetectorRuntime | None = None
        for precision in attempts:
            t0 = time.perf_counter()
            try:
                candidate = self._builder(path, device, precision, verify.actual_md5)
                self.timings_s["construct"] = time.perf_counter() - t0
                self._check_label_space(candidate)
                t0 = time.perf_counter()
                for _ in range(max(1, self.settings.inference_warmup_runs)):
                    self._warmup(candidate)
                self.timings_s["warmup"] = time.perf_counter() - t0
                runtime = candidate
                self.precision = precision
                break
            except (OfflineViolation, LabelSpaceError) as exc:
                return self._fail("load_failed", f"{type(exc).__name__}: {exc}"[:300])
            except ManifestError as exc:
                return self._fail("manifest_invalid", str(exc))
            except torch.cuda.OutOfMemoryError:
                return self._fail("device_unavailable", "CUDA out of memory while loading RF-DETR")
            except Exception as exc:  # noqa: BLE001
                if precision == "fp16":
                    # Recorded and reported: RF-DETR then runs fp32 on the same device.
                    self.precision_note = f"fp16 failed ({type(exc).__name__}: {exc}"[:200] + "); using fp32"
                    logger.warning("RF-DETR fp16 failed; retrying fp32", extra={"detail": self.precision_note})
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                logger.exception("RF-DETR construction or warm-up failed")
                return self._fail("load_failed", f"RF-DETR load failed: {type(exc).__name__}: {exc}"[:300])
        if runtime is None:
            return self._fail("load_failed", "RF-DETR could not be constructed")
        self.runtime = runtime
        self.ready = True
        self.reason = None
        self.detail = "ready"
        self.timings_s["total"] = time.perf_counter() - t_start
        metrics.MODEL_READY.labels(self.model_id, self.settings.inference_device).set(1)
        metrics.MODEL_LOAD_SECONDS.labels(self.model_id, self.settings.inference_device).set(self.timings_s["total"])
        logger.info(
            "model ready",
            extra={
                "model": self.model_id,
                "device": device,
                "precision": self.precision,
                "sha256": self.verified_sha256,
                "load_seconds": round(self.timings_s["total"], 3),
            },
        )
        return True

    def _check_label_space(self, runtime: DetectorRuntime) -> None:
        """Every enabled manifest class must be the same category in rfdetr's own COCO class list."""
        assert self.manifest is not None
        upstream = runtime.coco_category_names()
        enabled: dict[int, tuple[int, str]] = {}
        for dd_id, name in self.manifest.class_map.items():
            coco_id = dd_id - self.manifest.class_id_offset
            upstream_name = upstream.get(coco_id)
            if upstream_name is None or upstream_name.replace(" ", "_") != name:
                raise ManifestError(
                    f"class {dd_id} '{name}' does not match rfdetr COCO category {coco_id} ({upstream_name!r})"
                )
            enabled[coco_id] = (dd_id, name)
        self._enabled = enabled

    @staticmethod
    def _warmup(runtime: DetectorRuntime) -> None:
        frame = np.full((*WARMUP_HW, 3), 114, dtype=np.uint8)
        xyxy, scores, class_ids = runtime.predict(frame, 0.0)
        if not (np.isfinite(xyxy).all() and np.isfinite(scores).all()):
            raise FloatingPointError("non-finite RF-DETR outputs during warm-up")
        if not (len(xyxy) == len(scores) == len(class_ids)):
            raise ValueError("RF-DETR output arrays have different lengths")

    def predict(self, rgb: np.ndarray, conf_threshold: float) -> list[Detection]:
        """Enabled classes with score >= conf_threshold, boxes clipped to the original image (pixels)."""
        if not self.ready or self.runtime is None:
            raise RuntimeError("RF-DETR not ready")
        height, width = rgb.shape[:2]
        xyxy, scores, class_ids = self.runtime.predict(rgb, 0.0)
        out: list[Detection] = []
        seen_boxes: set[tuple[float, ...]] = set()
        for i in np.argsort(-scores, kind="stable"):
            box, score, coco_id = xyxy[i], float(scores[i]), int(class_ids[i])
            if not np.isfinite(score):
                continue
            key = tuple(float(v) for v in box)
            if key in seen_boxes:
                continue  # a lower-scoring class of a box already decided (one label per query)
            seen_boxes.add(key)
            mapped = self._enabled.get(coco_id)
            if mapped is None or score < conf_threshold:
                continue
            x1, x2 = (float(np.clip(v, 0, width)) for v in (box[0], box[2]))
            y1, y2 = (float(np.clip(v, 0, height)) for v in (box[1], box[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(
                Detection(
                    class_id=mapped[0],
                    class_name=mapped[1],
                    score=float(min(max(score, 0.0), 1.0)),
                    xyxy=(x1, y1, x2, y2),
                    model_id=self.model_id,
                )
            )
        out.sort(key=lambda d: d.score, reverse=True)
        return out

    def describe(self, device: str) -> dict[str, Any]:
        m = self.manifest
        return {
            "model_id": self.model_id,
            "ready": self.ready,
            "reason": self.reason,
            "sha256": self.verified_sha256 if self.ready else None,
            "size_bytes": self.verified_size if self.ready else None,
            "repository": m.repository if m else None,
            "repository_revision": m.repository_revision if m else None,
            "class_map": m.class_map_wire() if (m and self.ready) else {},
            "device": device,
            "precision": self.precision if self.ready else None,
            "input_policy": RFDETR_INPUT_POLICY,
        }
