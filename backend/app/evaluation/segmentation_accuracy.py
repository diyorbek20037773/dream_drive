"""Measured YOLOPv2 drivable-area accuracy on the KITTI Road benchmark training split (docs/EVALUATION_SEGMENTATION.md).

Pipeline: verified ``data_road.zip`` (SHA256) + committed manifest (``datasets/manifests/kitti_road_training_v1.json``,
per-member SHA256) -> each training image (lossless PNG) decoded to RGB and encoded as JPEG q95 exactly like the video
pipeline -> production ``InferenceClient.infer`` -> ``drivable_mask_png`` (YOLOPv2 output, original H x W) compared
with the KITTI ``<cat>_road_<idx>.png`` ground truth in the ORIGINAL PERSPECTIVE image space -> ``Evaluation`` JSON
(kind ``video_ml_accuracy``, source ``learned_model``) readable by ``app.evaluation.registry`` plus a per-image CSV,
the binary prediction masks and best/median/worst visual comparisons as evidence.

Ground-truth encoding (KITTI road devkit ``readme.txt``, update February 2015, and ``python/helper.py``
``getGroundTruth``): colour PNG; the first/R plane marks the valid evaluation area and the third/B plane the road area.
Pixels with R == 0 are "don't care" and are excluded from every count, exactly as the devkit's ``evalExp`` does with
``validArea``. The manifest builder records the colour histogram of every GT file so the encoding is verified on the
actual bytes, not assumed.

The metric is NOT the official KITTI benchmark number: KITTI ranks MaxF (F-measure maximised over a confidence sweep)
in a bird's-eye-view (BEV) metric grid; DreamDrive receives a binary mask, so precision/recall/F1/IoU are computed at
the service's fixed operating point in perspective space. Nothing here is simulated: without a reachable, ready
inference service the run fails instead of producing numbers.

CLI (run from ``backend/`` with the backend venv; complete commands in docs/EVALUATION_SEGMENTATION.md)::

    python -m app.evaluation.segmentation_accuracy build-manifest --zip <data_road.zip> --out <manifest.json>
    python -m app.evaluation.segmentation_accuracy run --zip <data_road.zip> --manifest <manifest.json>
        --inference-url http://127.0.0.1:18041 --out-dir ../artifacts/evaluations
        --evidence-dir ../artifacts/qa/spike/kitti_road
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from numpy.typing import NDArray

from app.evaluation.datasets import KITTI_ROAD_TRAINING
from app.evaluation.registry import write_evaluation
from app.perception.client import InferenceClient, InferenceClientError
from app.schemas.report import Metric
from app.video.encode import encode_jpeg_bytes

ProgressCb = Callable[[int, int, str], None]
BoolMask = NDArray[np.bool_]

MANIFEST_ID = "kitti_road_training_v1"
ARCHIVE_NAME = "data_road.zip"
ARCHIVE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip"
ARCHIVE_SHA256 = "45508c51ba09020235eb714e71c7d3ef15b4551be06d996772ecc8a20fc34a34"
ARCHIVE_SIZE_BYTES = 470_992_343
DEVKIT_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_road.zip"
# SHA256 of devkit_road.zip as downloaded 2026-10-10 (22,283 bytes); only its readme/helper.py were read, not vendored.
DEVKIT_SHA256 = "6f7cba10eeeb82d26cb0fa64ea4771ae92ff818be50098b162a21fcab8e06156"
LICENCE = (
    "All datasets and benchmarks on this page are copyright by us and published under the Creative Commons "
    "Attribution-NonCommercial-ShareAlike 3.0 License"
)
LICENCE_ID = "CC-BY-NC-SA-3.0"
LICENCE_URL = "https://www.cvlibs.net/datasets/kitti/"
ATTRIBUTION = (
    "Jannik Fritsch, Tobias Kuehnl, Andreas Geiger: 'A New Performance Measure and Evaluation Benchmark for Road "
    "Detection Algorithms', International Conference on Intelligent Transportation Systems (ITSC), 2013. KITTI Vision "
    "Benchmark Suite, Karlsruhe Institute of Technology and Toyota Technological Institute at Chicago."
)
IMAGE_DIR = "data_road/training/image_2/"
GT_DIR = "data_road/training/gt_image_2/"
CATEGORIES = ("um", "umm", "uu")
CATEGORY_NAMES = {"um": "urban marked", "umm": "urban multiple marked lanes", "uu": "urban unmarked"}
GT_ENCODING = {
    "source": "KITTI road devkit readme.txt (update February 2015) and python/helper.py getGroundTruth()",
    "valid_area": "R plane > 0 (first plane in RGB order); R == 0 is 'don't care' and excluded from all counts",
    "road": "B plane > 0 (third plane in RGB order); road pixels count only inside the valid area",
    "colours_expected_rgb": {"#ff00ff": "road (valid)", "#ff0000": "non-road (valid)", "#000000": "don't care"},
}
JPEG_QUALITY = 95  # identical to app.video.pipeline
CONF_THRESHOLD = 0.1  # production default; does not affect the drivable mask (argmax of the segmentation head)
YOLOPV2_ID = "yolopv2"
_STEM = re.compile(r"^(um|umm|uu)_(\d{6})$")
_COMMIT = re.compile(r"^[a-f0-9]{40,64}$")
_SHA = re.compile(r"^[a-f0-9]{64}$")
_MAX_FAILED_FRACTION = 0.05

LIMITATIONS_BASE = [
    "Cross-dataset evaluation: YOLOPv2 was trained on BDD100K, whose 'drivable area' label covers the directly "
    "drivable lane plus alternative (adjacent, same-direction or opposite) lanes, while KITTI 'road' is the whole "
    "road surface (composition of all lanes). The labelling conventions differ (e.g. parking strips, road edges, "
    "sidewalk-level surfaces), so part of the measured error is convention mismatch, not model error.",
    "KITTI images are about 1242x375 (aspect 3.3:1). The production letterbox policy (longest side 640, stride-32 "
    "minimal padding) turns them into a 640x224 canvas of 640x193 content; BDD100K training frames are 16:9, so the "
    "model sees an unusually wide, low-resolution canvas it was not trained on.",
    "Metrics are computed in the perspective image space at the service's fixed binary operating point. They are "
    "NOT the official KITTI Road benchmark result, which is MaxF (F-measure maximised over a confidence sweep) "
    "evaluated in a 800x400 bird's-eye-view grid ([-10 m, 10 m] lateral, [6 m, 46 m] longitudinal, 0.05 m cells). "
    "Perspective space weights near-field pixels far more heavily than BEV does. Do not compare these numbers with "
    "the KITTI leaderboard.",
    "Training split only (the only KITTI split with public labels; 289 images, three urban categories um/umm/uu, "
    "Karlsruhe, Germany, daylight). YOLOPv2 never trained on KITTI and nothing was tuned on these images, but the "
    "split is small and geographically narrow; it says nothing about night, rain, highways or other countries.",
    "Licence: KITTI is CC BY-NC-SA 3.0 (non-commercial). The images and labels are used for this local measurement "
    "only and are not redistributed; only file names and SHA256 hashes are committed.",
    "Lane-marking accuracy is NOT evaluated: KITTI um_lane ground truth is the ego-lane AREA, not lane markings, and "
    "YOLOPv2 has no ego-lane output. YOLOPv2 vehicle detection is not evaluated here either (KITTI Road has no boxes).",
    "KITTI training images are lossless PNG; they are encoded once as JPEG q95 before inference (same as the video "
    "pipeline), so results include one lossy compression step.",
    "GT 'don't care' pixels (R plane == 0) are excluded from all counts, following the devkit. Predictions inside "
    "those regions are neither true nor false positives.",
]


# ---------------------------------------------------------------------------------------------------------------------
# Hashing / IO helpers
# ---------------------------------------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Canonical serialisation: identical input -> identical bytes (no timestamps, fixed key order, LF)."""
    return (json.dumps(manifest, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------------------------------------------------
def decode_rgb_png(data: bytes) -> NDArray[np.uint8]:
    arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if arr is None:
        raise ValueError("image could not be decoded")
    return np.asarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def _decode_gt_bgr(data: bytes) -> NDArray[np.uint8]:
    arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("ground-truth PNG could not be decoded")
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"ground truth must be 8-bit colour, got dtype {arr.dtype} shape {arr.shape}")
    return np.asarray(arr[:, :, :3], dtype=np.uint8)


def decode_gt_png(data: bytes) -> tuple[BoolMask, BoolMask]:
    """KITTI road GT -> (road, valid). OpenCV order is BGR: plane 0 = B = road, plane 2 = R = valid (devkit)."""
    bgr = _decode_gt_bgr(data)
    road = bgr[:, :, 0] > 0
    valid = bgr[:, :, 2] > 0
    return road, valid


def gt_colour_histogram(data: bytes) -> dict[str, int]:
    """Exact RGB colour histogram of a GT PNG as ``{"#rrggbb": pixel_count}`` (sorted by colour)."""
    bgr = _decode_gt_bgr(data).astype(np.uint32)
    key = (bgr[:, :, 2] << 16) | (bgr[:, :, 1] << 8) | bgr[:, :, 0]
    values, counts = np.unique(key.ravel(), return_counts=True)
    return {f"#{int(v):06x}": int(c) for v, c in zip(values, counts, strict=True)}


def decode_pred_mask(data: str | None, width: int, height: int) -> tuple[BoolMask, int]:
    """Base64 PNG (uint8 {0,255}, original H x W) -> (bool mask, number of pixels not in {0,255}).

    Unlike the video pipeline's lenient decoder this raises on a missing, undecodable or wrongly sized mask, because a
    silently skipped frame would bias an accuracy measurement.
    """
    if not data:
        raise ValueError("drivable_mask_png missing")
    try:
        raw = np.frombuffer(base64.b64decode(data, validate=True), dtype=np.uint8)
    except (ValueError, TypeError) as exc:
        raise ValueError("drivable_mask_png is not valid base64") from exc
    mask = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError("drivable_mask_png is not a decodable PNG")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape != (height, width):
        raise ValueError(f"drivable mask is {mask.shape[1]}x{mask.shape[0]}, expected {width}x{height}")
    non_binary = int(np.count_nonzero((mask != 0) & (mask != 255)))
    return mask > 127, non_binary


# ---------------------------------------------------------------------------------------------------------------------
# Pixel counting
# ---------------------------------------------------------------------------------------------------------------------
def _ratio(num: int, den: int) -> float | None:
    return None if den <= 0 else num / den


@dataclass(frozen=True)
class PixelCounts:
    """Confusion counts over valid (non-ignored) pixels. Ratios are None when undefined (zero denominator)."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def __add__(self, other: PixelCounts) -> PixelCounts:
        return PixelCounts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn)

    @property
    def union(self) -> int:
        return self.tp + self.fp + self.fn

    @property
    def valid(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def gt_road(self) -> int:
        return self.tp + self.fn

    @property
    def pred_road(self) -> int:
        return self.tp + self.fp

    @property
    def iou(self) -> float | None:
        return _ratio(self.tp, self.union)

    @property
    def precision(self) -> float | None:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float | None:
        return _ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn)


def confusion(pred: BoolMask, road: BoolMask, valid: BoolMask) -> PixelCounts:
    """Pixel confusion of a binary prediction vs KITTI road GT, counting only ``valid`` pixels (devkit ``evalExp``)."""
    if not (pred.shape == road.shape == valid.shape):
        raise ValueError(f"shape mismatch: pred {pred.shape}, road {road.shape}, valid {valid.shape}")
    p = pred & valid
    g = road & valid
    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(g & ~p))
    tn = int(np.count_nonzero(valid)) - tp - fp - fn
    return PixelCounts(tp, fp, fn, tn)


def percentile(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("percentile of an empty list")
    return float(np.percentile(arr, q))


def _r(x: float | None, nd: int = 6) -> float | None:
    return None if x is None else round(float(x), nd)


# ---------------------------------------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------------------------------------
def _gt_counts(colours: Mapping[str, int]) -> dict[str, int]:
    road = non_road = ignore = road_outside_valid = 0
    for hex_colour, count in colours.items():
        v = int(hex_colour[1:], 16)
        r, b = (v >> 16) & 0xFF, v & 0xFF
        if r == 0:
            ignore += count
            if b > 0:
                road_outside_valid += count
        elif b > 0:
            road += count
        else:
            non_road += count
    return {"road_px": road, "non_road_px": non_road, "ignore_px": ignore, "road_outside_valid_px": road_outside_valid}


def build_manifest(
    zip_path: Path, out_path: Path | None = None, *, expected_archive_sha256: str | None = ARCHIVE_SHA256
) -> dict[str, Any]:
    """Deterministic manifest of every KITTI Road training image and its road (and, for um, lane) GT member."""
    zip_path = Path(zip_path)
    archive_sha = sha256_file(zip_path)
    if expected_archive_sha256 is not None and archive_sha != expected_archive_sha256:
        raise ValueError(f"{zip_path.name}: sha256 {archive_sha} != expected {expected_archive_sha256}")
    entries: list[dict[str, Any]] = []
    colour_totals: dict[str, Counter[str]] = {c: Counter() for c in CATEGORIES}
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        images = sorted(n for n in names if n.startswith(IMAGE_DIR) and n.endswith(".png") and n != IMAGE_DIR)
        if not images:
            raise ValueError(f"no training images under {IMAGE_DIR}")
        for member in images:
            stem = member[len(IMAGE_DIR) : -len(".png")]
            m = _STEM.match(stem)
            if not m:
                raise ValueError(f"unexpected image name {member}")
            cat, idx = m.group(1), m.group(2)
            img = zf.read(member)
            rgb = decode_rgb_png(img)
            h, w = rgb.shape[:2]
            gt_member = f"{GT_DIR}{cat}_road_{idx}.png"
            if gt_member not in names:
                raise ValueError(f"missing road ground truth {gt_member}")
            gt = zf.read(gt_member)
            colours = gt_colour_histogram(gt)
            gt_h, gt_w = _decode_gt_bgr(gt).shape[:2]
            if (gt_w, gt_h) != (w, h):
                raise ValueError(f"{gt_member} is {gt_w}x{gt_h}, image is {w}x{h}")
            colour_totals[cat].update(colours)
            lane_member = f"{GT_DIR}{cat}_lane_{idx}.png"
            lane = None
            if lane_member in names:
                lane_bytes = zf.read(lane_member)
                lane = {"member": lane_member, "size_bytes": len(lane_bytes), "sha256": _sha256(lane_bytes)}
            entries.append(
                {
                    "id": stem,
                    "category": cat,
                    "width": w,
                    "height": h,
                    "image": {"member": member, "size_bytes": len(img), "sha256": _sha256(img)},
                    "gt_road": {
                        "member": gt_member,
                        "size_bytes": len(gt),
                        "sha256": _sha256(gt),
                        **_gt_counts(colours),
                        "colours_rgb": colours,
                    },
                    "gt_lane_not_evaluated": lane,
                }
            )
    by_cat = Counter(e["category"] for e in entries)
    sizes = Counter(f"{e['width']}x{e['height']}" for e in entries)
    all_colours: Counter[str] = Counter()
    for c in CATEGORIES:
        all_colours.update(colour_totals[c])
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "manifest_id": MANIFEST_ID,
        "description": "KITTI Road benchmark training split (all images with public ground truth). File names and "
        "SHA256 hashes only; images and labels are not redistributed.",
        "source": {
            "archive_name": ARCHIVE_NAME,
            "archive_url": ARCHIVE_URL,
            "archive_sha256": archive_sha,
            "archive_size_bytes": zip_path.stat().st_size,
            "licence": LICENCE,
            "licence_id": LICENCE_ID,
            "licence_url": LICENCE_URL,
            "attribution": ATTRIBUTION,
            "devkit_url": DEVKIT_URL,
            "devkit_sha256": DEVKIT_SHA256,
        },
        "split": "training",
        "gt_encoding": GT_ENCODING,
        "counts": {
            "images": len(entries),
            "by_category": {c: by_cat.get(c, 0) for c in CATEGORIES},
            "image_sizes": dict(sorted(sizes.items())),
            "gt_lane_files_not_evaluated": sum(1 for e in entries if e["gt_lane_not_evaluated"]),
        },
        "gt_colour_totals_rgb": {
            "all": dict(sorted(all_colours.items())),
            **{c: dict(sorted(colour_totals[c].items())) for c in CATEGORIES},
        },
        "images": entries,
    }
    if out_path is not None:
        _atomic_write_bytes(Path(out_path), manifest_bytes(manifest))
    return manifest


# ---------------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------------
def build_metrics(
    totals: Mapping[str, PixelCounts],
    images_by_scope: Mapping[str, int],
    per_image_iou: list[float],
    iou_undefined: int,
    latency_ms: list[float],
    roundtrip_ms: list[float],
    yolopv2_forward_ms: list[float],
    device: str | None,
) -> dict[str, Metric]:
    """Metric dict for the Evaluation. Dataset-level ratios pool pixels over images (sum TP / sum union)."""
    m: dict[str, Metric] = {}
    for scope in ("overall", *CATEGORIES):
        c = totals.get(scope, PixelCounts())
        n = images_by_scope.get(scope, 0)
        where = "all categories" if scope == "overall" else f"category {scope} ({CATEGORY_NAMES[scope]})"
        px = (
            f"{n} images, {c.valid:,} valid GT pixels ({c.gt_road:,} road, {c.valid - c.gt_road:,} non-road), "
            f"{c.pred_road:,} predicted drivable; sample_count = images"
        )
        defs = {
            "iou": ("pixel IoU = sum TP / sum (TP+FP+FN)", c.iou),
            "precision": ("pixel precision = sum TP / sum (TP+FP)", c.precision),
            "recall": ("pixel recall = sum TP / sum (TP+FN)", c.recall),
            "f1": (
                "pixel F1 = 2TP / (2TP+FP+FN), KITTI's F-measure at the fixed binary operating point but in "
                "perspective space (not the official BEV MaxF)",
                c.f1,
            ),
        }
        for name, (desc, value) in defs.items():
            key = f"drivable_{name}_{scope}"
            if n == 0:
                m[key] = Metric.missing("ratio", f"no evaluated images for {where}")
            elif value is None:
                m[key] = Metric.missing("ratio", f"undefined (zero denominator) for {where}; {px}", sample_count=n)
            else:
                m[key] = Metric.measured(round(value, 4), "ratio", n, f"{desc}, dataset-level over {where}; {px}")
    k = len(per_image_iou)
    if k:
        base = (
            f"over {k} images with a non-empty union (TP+FP+FN > 0); {iou_undefined} images had an empty union and "
            "are excluded; sample_count = images"
        )
        m["drivable_iou_per_image_mean"] = Metric.measured(
            round(float(np.mean(per_image_iou)), 4), "ratio", k, f"mean of per-image pixel IoU {base}"
        )
        for q in (10, 50, 90):
            m[f"drivable_iou_per_image_p{q}"] = Metric.measured(
                round(percentile(per_image_iou, q), 4), "ratio", k, f"{q}th percentile of per-image pixel IoU {base}"
            )
    else:
        for key in ("mean", "p10", "p50", "p90"):
            m[f"drivable_iou_per_image_{key}"] = Metric.missing("ratio", "no image with a non-empty union")
    m["images_evaluated"] = Metric.measured(
        images_by_scope.get("overall", 0),
        "count",
        max(images_by_scope.get("overall", 0), 1),
        "KITTI Road training images evaluated (um/umm/uu)",
    )
    dev = device or "unknown device"
    if latency_ms:
        for q in (50, 95):
            m[f"inference_latency_p{q}_ms"] = Metric.measured(
                round(percentile(latency_ms, q), 1),
                "ms",
                len(latency_ms),
                f"service timing_ms.total per image ({dev}; every model the service runs)",
            )
    if yolopv2_forward_ms:
        for q in (50, 95):
            m[f"yolopv2_forward_p{q}_ms"] = Metric.measured(
                round(percentile(yolopv2_forward_ms, q), 1),
                "ms",
                len(yolopv2_forward_ms),
                f"service timing_ms.inference_by_model.yolopv2 per image ({dev})",
            )
    if roundtrip_ms:
        for q in (50, 95):
            m[f"request_roundtrip_p{q}_ms"] = Metric.measured(
                round(percentile(roundtrip_ms, q), 1),
                "ms",
                len(roundtrip_ms),
                "client HTTP round trip incl. JPEG upload, masks and JSON",
            )
    m["kitti_bev_maxf"] = Metric.missing(
        "ratio",
        "not evaluated: the official KITTI Road metric is MaxF over a confidence sweep in bird's-eye view; the "
        "service returns a binary perspective mask, so only the perspective-space fixed-threshold metrics above exist",
    )
    m["lane_marking_accuracy"] = Metric.missing(
        "ratio",
        "not evaluated: KITTI um_lane ground truth is the ego-lane area, not lane markings; no openly licensed "
        "lane-marking labels are used here",
    )
    m["ego_lane_iou"] = Metric.missing(
        "ratio", "unsupported: YOLOPv2 outputs drivable area and lane lines, not an ego-lane area", status="unsupported"
    )
    return m


# ---------------------------------------------------------------------------------------------------------------------
# Visual comparisons
# ---------------------------------------------------------------------------------------------------------------------
def _label(panel: NDArray[np.uint8], text: str) -> NDArray[np.uint8]:
    bar = np.full((26, panel.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(bar, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def render_comparison(
    rgb: NDArray[np.uint8], road: BoolMask, valid: BoolMask, pred: BoolMask, title: str
) -> NDArray[np.uint8]:
    """Three stacked panels (BGR, for cv2.imwrite): input; GT (road green, don't-care darkened); prediction
    (TP green, FP red, FN blue, don't-care darkened)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)

    def tint(base: NDArray[np.float32], mask: BoolMask, colour: tuple[int, int, int], a: float = 0.55) -> None:
        base[mask] = (1.0 - a) * base[mask] + a * np.asarray(colour, dtype=np.float32)

    gt_panel = bgr.copy()
    tint(gt_panel, road & valid, (0, 255, 0))
    gt_panel[~valid] *= 0.3
    pr_panel = bgr.copy()
    tint(pr_panel, pred & road & valid, (0, 255, 0))
    tint(pr_panel, pred & ~road & valid, (0, 0, 255))
    tint(pr_panel, ~pred & road & valid, (255, 0, 0))
    pr_panel[~valid] *= 0.3
    panels = [
        _label(bgr.astype(np.uint8), f"input  {title}"),
        _label(gt_panel.clip(0, 255).astype(np.uint8), "KITTI GT: green = road, dark = don't care"),
        _label(
            pr_panel.clip(0, 255).astype(np.uint8),
            "YOLOPv2 drivable: green = TP, red = FP, blue = FN, dark = don't care",
        ),
    ]
    return np.vstack(panels)


# ---------------------------------------------------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------------------------------------------------
_CSV_FIELDS = (
    "id",
    "category",
    "width",
    "height",
    "valid_px",
    "gt_road_px",
    "pred_drivable_valid_px",
    "tp",
    "fp",
    "fn",
    "tn",
    "iou",
    "precision",
    "recall",
    "f1",
    "pred_drivable_fraction_service",
    "pred_non_binary_px",
    "canvas_width",
    "canvas_height",
    "scale",
    "pad_top",
    "pad_bottom",
    "service_total_ms",
    "yolopv2_forward_ms",
    "roundtrip_ms",
)


def _model_info(client: InferenceClient) -> dict[str, Any]:
    return client.model_info().model_dump(mode="json")


def run_drivable_accuracy(
    zip_path: Path,
    manifest_path: Path,
    client: InferenceClient,
    out_dir: Path,
    code_commit: str,
    progress_cb: ProgressCb | None = None,
    *,
    eval_id: str | None = None,
    evidence_dir: Path | None = None,
    run_label: str | None = None,
    max_images: int | None = None,
    model_info: Mapping[str, Any] | None = None,
    save_predictions: bool = True,
    visuals: bool = True,
    extra_details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run every manifest image through the inference service, score ``drivable_mask_png`` against KITTI road GT and
    persist an Evaluation (returned with ``details``). Evidence (per-image CSV, binary prediction masks, three visual
    comparisons) goes to ``evidence_dir`` (default ``out_dir``) under the evaluation id."""
    zip_path = Path(zip_path)
    manifest_path = Path(manifest_path)
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    manifest_sha = _sha256(raw_manifest)
    expected_zip_sha = str(manifest["source"]["archive_sha256"])
    zip_sha = sha256_file(zip_path)
    if zip_sha != expected_zip_sha:
        raise ValueError(f"{zip_path.name}: sha256 {zip_sha} != manifest archive_sha256 {expected_zip_sha}")
    entries: list[dict[str, Any]] = sorted(manifest["images"], key=lambda e: str(e["id"]))
    if max_images is not None:
        entries = entries[: int(max_images)]
    if not entries:
        raise ValueError("manifest has no images")

    eval_id = eval_id or uuid.uuid4().hex
    created = datetime.now(UTC)
    if not client.ready():
        raise InferenceClientError("model_not_ready", f"inference service at {client.base_url} is not ready")
    info = dict(model_info) if model_info is not None else _model_info(client)
    ev_dir = Path(evidence_dir) if evidence_dir is not None else Path(out_dir)
    pred_dir = ev_dir / f"{eval_id}_pred_masks"

    rows: list[dict[str, Any]] = []
    counts_by_id: dict[str, PixelCounts] = {}
    packed_preds: dict[str, NDArray[np.uint8]] = {}
    failed: list[dict[str, Any]] = []
    latency: list[float] = []
    roundtrip: list[float] = []
    yolo_fwd: list[float] = []
    identities: list[dict[str, Any]] = []
    ensemble_reported: set[str] = set()
    geometries: Counter[str] = Counter()
    non_binary_total = 0
    total = len(entries)
    with zipfile.ZipFile(zip_path) as zf:
        for i, entry in enumerate(entries):
            eid = str(entry["id"])
            try:
                img = zf.read(entry["image"]["member"])
                if _sha256(img) != entry["image"]["sha256"]:
                    raise ValueError(f"{entry['image']['member']} sha256 does not match the manifest")
                gt = zf.read(entry["gt_road"]["member"])
                if _sha256(gt) != entry["gt_road"]["sha256"]:
                    raise ValueError(f"{entry['gt_road']['member']} sha256 does not match the manifest")
                rgb = decode_rgb_png(img)
                h, w = rgb.shape[:2]
                if (w, h) != (int(entry["width"]), int(entry["height"])):
                    raise ValueError(f"decoded size {w}x{h} != manifest {entry['width']}x{entry['height']}")
                road, valid = decode_gt_png(gt)
                jpeg = encode_jpeg_bytes(rgb, JPEG_QUALITY)
                t0 = time.perf_counter()
                result = client.infer(jpeg, frame_id=i, conf_threshold=CONF_THRESHOLD, include_masks=True)
                rt = (time.perf_counter() - t0) * 1000.0
                if (result.image.width, result.image.height) != (w, h):
                    raise ValueError("service image size differs from the input")
                pred, non_binary = decode_pred_mask(result.drivable_mask_png, w, h)
            except (InferenceClientError, httpx.HTTPError, ValueError, OSError, KeyError) as exc:
                failed.append({"id": eid, "error": type(exc).__name__, "message": str(exc)[:300]})
                if progress_cb is not None:
                    progress_cb(i + 1, total, f"failed:{eid}")
                continue
            c = confusion(pred, road, valid)
            counts_by_id[eid] = c
            packed_preds[eid] = np.packbits(pred)
            non_binary_total += non_binary
            roundtrip.append(rt)
            latency.append(float(result.timing_ms.total))
            fwd = result.timing_ms.inference_by_model.get(YOLOPV2_ID)
            if fwd is not None:
                yolo_fwd.append(float(fwd))
            if not identities:
                identities = [m.model_dump(mode="json") for m in (result.models or [result.model])]
            if result.ensemble_sha256:
                ensemble_reported.add(result.ensemble_sha256)
            g = result.geometry
            geometries[
                f"{w}x{h}->canvas {g.canvas_width}x{g.canvas_height} scale {g.scale:.6f} "
                f"pad l{g.pad_left} t{g.pad_top} r{g.pad_right} b{g.pad_bottom}"
            ] += 1
            if save_predictions:
                ok, buf = cv2.imencode(".png", pred.astype(np.uint8) * 255)
                if not ok:
                    raise RuntimeError("PNG encoding of a prediction mask failed")
                _atomic_write_bytes(pred_dir / f"{eid}.png", buf.tobytes())
            rows.append(
                {
                    "id": eid,
                    "category": entry["category"],
                    "width": w,
                    "height": h,
                    "valid_px": c.valid,
                    "gt_road_px": c.gt_road,
                    "pred_drivable_valid_px": c.pred_road,
                    "tp": c.tp,
                    "fp": c.fp,
                    "fn": c.fn,
                    "tn": c.tn,
                    "iou": _r(c.iou),
                    "precision": _r(c.precision),
                    "recall": _r(c.recall),
                    "f1": _r(c.f1),
                    "pred_drivable_fraction_service": _r(result.mask_stats.drivable_fraction),
                    "pred_non_binary_px": non_binary,
                    "canvas_width": g.canvas_width,
                    "canvas_height": g.canvas_height,
                    "scale": _r(g.scale),
                    "pad_top": g.pad_top,
                    "pad_bottom": g.pad_bottom,
                    "service_total_ms": round(float(result.timing_ms.total), 2),
                    "yolopv2_forward_ms": None if fwd is None else round(float(fwd), 2),
                    "roundtrip_ms": round(rt, 2),
                }
            )
            if progress_cb is not None:
                progress_cb(i + 1, total, eid)

    if not rows:
        raise RuntimeError(f"no image could be evaluated ({len(failed)} failures)")
    if len(failed) > _MAX_FAILED_FRACTION * total:
        raise RuntimeError(f"{len(failed)} of {total} images failed; refusing to report partial accuracy")
    ens = sorted(ensemble_reported)
    if len(ens) > 1:
        raise RuntimeError(f"inconsistent ensemble_sha256 across images: {ens}")

    totals: dict[str, PixelCounts] = {"overall": PixelCounts(), **{c: PixelCounts() for c in CATEGORIES}}
    images_by_scope: Counter[str] = Counter()
    for row in rows:
        c = counts_by_id[row["id"]]
        totals["overall"] = totals["overall"] + c
        totals[row["category"]] = totals[row["category"]] + c
        images_by_scope["overall"] += 1
        images_by_scope[row["category"]] += 1
    per_image_iou = [float(r["iou"]) for r in rows if r["iou"] is not None]
    iou_undefined = sum(1 for r in rows if r["iou"] is None)
    yolo = next((m for m in identities if m.get("model_id") == YOLOPV2_ID), None)
    if yolo is None and str(info.get("model_id")) == YOLOPV2_ID:
        yolo = {"model_id": YOLOPV2_ID, "sha256": info.get("sha256")}
    yolo_sha = str(yolo.get("sha256")) if yolo else ""
    device = identities[0].get("device") if identities else info.get("device")
    metrics = build_metrics(totals, images_by_scope, per_image_iou, iou_undefined, latency, roundtrip, yolo_fwd, device)

    # Visual comparisons: best / median / worst per-image IoU (ties broken by id for determinism).
    ranked = sorted((r for r in rows if r["iou"] is not None), key=lambda r: (float(r["iou"]), r["id"]))
    visual_files: dict[str, dict[str, Any]] = {}
    if visuals and ranked:
        picks = {"worst": ranked[0], "median": ranked[len(ranked) // 2], "best": ranked[-1]}
        by_id = {str(e["id"]): e for e in entries}
        with zipfile.ZipFile(zip_path) as zf:
            for label, row in picks.items():
                entry = by_id[row["id"]]
                rgb = decode_rgb_png(zf.read(entry["image"]["member"]))
                road, valid = decode_gt_png(zf.read(entry["gt_road"]["member"]))
                h, w = rgb.shape[:2]
                pred = np.unpackbits(packed_preds[row["id"]], count=h * w).reshape(h, w).astype(bool)
                title = f"{label}: {row['id']} IoU {row['iou']:.4f} P {row['precision']} R {row['recall']}"
                ok, buf = cv2.imencode(".png", render_comparison(rgb, road, valid, pred, title))
                if not ok:
                    raise RuntimeError("PNG encoding of a visual comparison failed")
                path = ev_dir / f"{eval_id}_visual_{label}_{row['id']}.png"
                _atomic_write_bytes(path, buf.tobytes())
                visual_files[label] = {"id": row["id"], "iou": row["iou"], "file": path.name}

    limitations = list(LIMITATIONS_BASE)
    if run_label:
        limitations.insert(0, f"Run label '{run_label}': validation run, not the release measurement.")
    if max_images is not None:
        limitations.insert(0, f"Partial run: only the first {total} manifest images (max_images={max_images}).")
    if failed:
        limitations.append(f"{len(failed)} of {total} manifest images were excluded (decode/inference failures).")
    if len(identities) > 1:
        limitations.append(
            "The service ran the YOLOPv2 + RF-DETR-N ensemble; the drivable mask is produced by YOLOPv2 alone, so "
            "model_sha256 is the YOLOPv2 checkpoint and the ensemble identity is recorded in details. Service latency "
            "includes both models."
        )

    buf_csv = io.StringIO()
    writer = csv.DictWriter(buf_csv, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    csv_path = ev_dir / f"{eval_id}.per_image.csv"
    _atomic_write_bytes(csv_path, buf_csv.getvalue().encode("utf-8"))

    def counts_dict(c: PixelCounts, n: int) -> dict[str, Any]:
        return {
            "images": n,
            "tp": c.tp,
            "fp": c.fp,
            "fn": c.fn,
            "tn": c.tn,
            "valid_px": c.valid,
            "gt_road_px": c.gt_road,
            "pred_road_px": c.pred_road,
            "iou": _r(c.iou),
            "precision": _r(c.precision),
            "recall": _r(c.recall),
            "f1": _r(c.f1),
        }

    git = _git_state()
    doc: dict[str, Any] = {
        "id": eval_id,
        "kind": "video_ml_accuracy",
        "state": "succeeded",
        "created_utc": created.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "source": "learned_model",
        "sample_count": len(rows),
        "dataset": dict(KITTI_ROAD_TRAINING),
        "metrics": {k: v.model_dump(mode="json") for k, v in metrics.items()},
        "limitations": limitations,
        "code_commit": code_commit if _COMMIT.match(code_commit or "") else None,
        "model_sha256": yolo_sha if _SHA.match(yolo_sha) else None,
        "seeds": None,
        "error_code": None,
        "details": {
            "task": "drivable_area_segmentation",
            "run_label": run_label,
            "dataset": {
                "name": "KITTI Road benchmark, training split",
                "manifest_id": manifest.get("manifest_id"),
                "manifest_file": manifest_path.name,
                "manifest_sha256": manifest_sha,
                "archive_name": manifest["source"].get("archive_name"),
                "archive_url": manifest["source"].get("archive_url"),
                "archive_sha256": zip_sha,
                "licence": manifest["source"].get("licence"),
                "licence_id": manifest["source"].get("licence_id"),
                "attribution": manifest["source"].get("attribution"),
                "images_in_manifest": len(manifest["images"]),
                "images_requested": total,
                "images_evaluated": len(rows),
                "images_failed": failed,
                "gt_encoding": manifest.get("gt_encoding"),
            },
            "protocol": {
                "input": "KITTI image_2 PNG -> RGB -> JPEG q95 (app.video.encode.encode_jpeg_bytes, as the video "
                "pipeline) -> POST /v1/infer via app.perception.client.InferenceClient",
                "prediction": "drivable_mask_png (YOLOPv2 seg head argmax == 1, unletterboxed to the original "
                "frame by the service) > 127",
                "ground_truth": "gt_image_2/<cat>_road_<idx>.png; road = B > 0, valid = R > 0 (devkit)",
                "space": "original perspective image space (no bird's-eye-view transform)",
                "dataset_level": "sum of per-image TP/FP/FN over images, then ratios",
                "per_image": "ratios per image; images with TP+FP+FN == 0 have undefined IoU and are excluded from "
                "the per-image distribution",
                "jpeg_quality": JPEG_QUALITY,
                "conf_threshold": CONF_THRESHOLD,
                "letterbox_geometries": dict(geometries),
                "pred_non_binary_pixels_total": non_binary_total,
            },
            "totals": {s: counts_dict(totals[s], images_by_scope.get(s, 0)) for s in ("overall", *CATEGORIES)},
            "models": identities,
            "yolopv2_sha256": yolo_sha or None,
            "ensemble_sha256_reported": ens[0] if ens else None,
            "service_model_info": {
                k: info.get(k)
                for k in (
                    "model_id",
                    "ready",
                    "sha256",
                    "device",
                    "precision",
                    "torch_version",
                    "input_policy",
                    "ensemble_id",
                    "ensemble_sha256",
                )
            }
            | {
                "models": [
                    {k: m.get(k) for k in ("model_id", "sha256", "size_bytes", "repository_revision", "input_policy")}
                    for m in info.get("models") or []
                    if isinstance(m, dict)
                ]
            },
            "latency_ms": {
                "service_total_p50": _r(percentile(latency, 50), 2) if latency else None,
                "service_total_p95": _r(percentile(latency, 95), 2) if latency else None,
                "yolopv2_forward_p50": _r(percentile(yolo_fwd, 50), 2) if yolo_fwd else None,
                "yolopv2_forward_p95": _r(percentile(yolo_fwd, 95), 2) if yolo_fwd else None,
                "roundtrip_p50": _r(percentile(roundtrip, 50), 2) if roundtrip else None,
                "roundtrip_p95": _r(percentile(roundtrip, 95), 2) if roundtrip else None,
                "samples": len(latency),
            },
            "evidence": {
                "per_image_csv": csv_path.name,
                "prediction_masks_dir": pred_dir.name if save_predictions else None,
                "visuals": visual_files,
            },
            "code": git,
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "cpu_count": os.cpu_count(),
            },
            **(dict(extra_details) if extra_details else {}),
        },
    }
    write_evaluation(Path(out_dir), doc)
    return doc


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, timeout=20, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"head": None, "dirty": None}
    return {"head": head if _COMMIT.match(head) else None, "dirty": bool(dirty)}


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------
def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m app.evaluation.segmentation_accuracy")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-manifest", help="hash every KITTI Road training image and its ground truth")
    b.add_argument("--zip", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run", help="evaluate a running inference service on the KITTI Road training split")
    r.add_argument("--zip", type=Path, required=True)
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--inference-url", required=True)
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--evidence-dir", type=Path, default=None)
    r.add_argument("--code-commit", default="")
    r.add_argument("--timeout-s", type=float, default=120.0)
    r.add_argument("--run-label", default=None)
    r.add_argument("--max-images", type=int, default=None)
    args = p.parse_args(argv)
    if args.cmd == "build-manifest":
        m = build_manifest(args.zip, args.out)
        print(json.dumps({"counts": m["counts"], "gt_colour_totals_rgb": m["gt_colour_totals_rgb"]}, indent=2))
        return 0
    commit = args.code_commit or (_git_state().get("head") or "")

    def progress(done: int, total: int, label: str) -> None:
        if done % 25 == 0 or done == total or label.startswith("failed"):
            print(f"[{done}/{total}] {label}", flush=True)

    with InferenceClient(args.inference_url, args.timeout_s) as client:
        doc = run_drivable_accuracy(
            args.zip,
            args.manifest,
            client,
            args.out_dir,
            commit,
            progress,
            evidence_dir=args.evidence_dir,
            run_label=args.run_label,
            max_images=args.max_images,
        )
    summary = {
        "id": doc["id"],
        "sample_count": doc["sample_count"],
        "model_sha256": doc["model_sha256"],
        "ensemble_sha256_reported": doc["details"]["ensemble_sha256_reported"],
        "metrics": {k: (v["value"], v["sample_count"], v["status"]) for k, v in doc["metrics"].items()},
        "evidence": doc["details"]["evidence"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
