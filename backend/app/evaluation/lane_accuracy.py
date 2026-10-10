"""Measured YOLOPv2 lane-MARKING (lane line) pixel accuracy on a licence-clear Roboflow Universe split
(docs/EVALUATION_LANES.md).

Pipeline: one Roboflow ``coco-segmentation`` split directory (``_annotations.coco.json`` + images) + committed
manifest (``datasets/manifests/roboflow_<project>_v<n>.json``, per-file SHA256) -> each image decoded to RGB at its
original (exported) resolution and encoded as JPEG q95 exactly like the video pipeline -> production
``InferenceClient.infer`` -> ``lane_mask_png`` (YOLOPv2 lane-line head, probability > 0.5, unletterboxed by the service
to the original H x W) compared with the polygon ground truth rasterised at the original resolution -> ``Evaluation``
JSON (kind ``video_ml_accuracy``, source ``learned_model``) readable by ``app.evaluation.registry`` plus a per-image CSV
and best/median/worst visual comparisons.

Two pixel protocols are reported, both pooled over the dataset (sum TP / sum ...) and per image (p10/p50/p90):

* exact: the binary prediction vs the rasterised GT polygons, pixel for pixel;
* tolerant: BOTH masks are dilated with a disk of radius ``k = floor(8 * W / 1280 + 0.5)`` px (8 px at 1280 wide,
  4 px at 640 wide) before counting. This forgives a few pixels of lateral offset and the different line widths of
  YOLOPv2 (BDD100K-style thin lines) and of hand-drawn polygons, without rewarding a missing or invented line.

Additionally a buffered (boundary-style) precision/recall is recorded: precision = predicted pixels within k px of a
GT pixel / predicted pixels; recall = GT pixels within k px of a predicted pixel / GT pixels.

Nothing is simulated: without a reachable, ready inference service the run fails instead of producing numbers. The
dataset images and labels are never committed; the manifest holds file names and hashes only.

CLI (run from ``backend/`` with the backend venv; full commands in docs/EVALUATION_LANES.md)::

    python -m app.evaluation.lane_accuracy build-manifest --dataset <key> --split-dir <dir> --out <manifest.json>
    python -m app.evaluation.lane_accuracy run --split-dir <dir> --manifest <manifest.json>
        --inference-url http://127.0.0.1:18051 --out-dir ../artifacts/evaluations
        --evidence-dir ../artifacts/qa/spike/lanes --max-images 150 --timeout-s 60
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import os
import platform
import sys
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from numpy.typing import NDArray

from app.evaluation.registry import write_evaluation
from app.evaluation.segmentation_accuracy import (
    _COMMIT,
    _SHA,
    PixelCounts,
    _atomic_write_bytes,
    _git_state,
    _label,
    _r,
    _sha256,
    decode_rgb_png,
    manifest_bytes,
    percentile,
    sha256_file,
)
from app.perception.client import InferenceClient, InferenceClientError
from app.schemas.report import Metric
from app.video.encode import encode_jpeg_bytes

ProgressCb = Callable[[int, int, str], None]
BoolMask = NDArray[np.bool_]

ANNOTATIONS_FILE = "_annotations.coco.json"
JPEG_QUALITY = 95  # identical to app.video.pipeline
CONF_THRESHOLD = 0.1  # production default; does not affect the lane mask (lane-line probability > 0.5)
YOLOPV2_ID = "yolopv2"
TOLERANCE_PX_AT_1280 = 8
_IMAGE_EXT = (".jpg", ".jpeg", ".png")
_MAX_FAILED_FRACTION = 0.05

# Audited Roboflow Universe datasets (docs/EVALUATION_LANES.md). Every value was read from the Roboflow API
# (project/version metadata, GET only) or from the export itself; nothing is assumed.
DATASETS: dict[str, dict[str, Any]] = {
    "rtqi-lane-marking-9qzw7-v2": {
        "manifest_id": "roboflow_lane-marking-9qzw7_v2",
        "workspace": "rtqi",
        "workspace_name": "RTQI",
        "project": "lane-marking-9qzw7",
        "project_name": "Lane-Marking",
        "version": 2,
        "url": "https://universe.roboflow.com/rtqi/lane-marking-9qzw7/dataset/2",
        "licence": "CC BY 4.0",
        "licence_id": "CC-BY-4.0",
        "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "RTQI, 'Lane-Marking' dataset (version 2), Roboflow Universe, "
        "https://universe.roboflow.com/rtqi/lane-marking-9qzw7, licensed CC BY 4.0.",
        "export_format": "coco-segmentation",
        "annotation_type": "instance-segmentation polygons",
        "preprocessing": {"auto-orient": True, "resize": {"format": "Stretch to", "width": 640, "height": 640}},
        "augmentation": None,
        "classes": ["LC", "LD", "MD", "RC", "RD"],
    },
}

LIMITATIONS_BASE = [
    "Cross-dataset label convention: YOLOPv2's lane-line head was trained on BDD100K lane annotations (thin "
    "polylines along marking edges/centres, rendered at a fixed width), while this ground truth is hand-drawn "
    "polygons covering the painted marking. Line width, which markings are labelled (e.g. road edges, crosswalks, "
    "far-field dashes) and how dashed gaps are treated differ, so part of the measured error is convention mismatch, "
    "not model error. The exact pixel IoU of thin structures is very sensitive to 1-3 px offsets; read the tolerant "
    "numbers together with the exact ones.",
    "Pixel-level metrics only: there is no per-lane instance matching (no TuSimple accuracy or CULane F1 "
    "with IoU-matched lanes), no lane type (solid/dashed) and no ego-lane assignment. YOLOPv2 outputs one binary "
    "lane-line mask and has no lane instances or types.",
    "The service's fixed binary operating point (lane probability > 0.5) is evaluated; no threshold sweep.",
    "Each exported image (already JPEG) is decoded and re-encoded once as JPEG q95 before inference, exactly as the "
    "video pipeline does, so results include one additional lossy compression step.",
    "Production letterbox policy: the longest side is resized to 640 and the short side padded to a multiple of 32; "
    "the service crops the padding and resizes the lane probabilities back to the original H x W before "
    "thresholding.",
]


# ---------------------------------------------------------------------------------------------------------------------
# Ground-truth rasterisation
# ---------------------------------------------------------------------------------------------------------------------
def _decode_compressed_rle(counts: str) -> list[int]:
    """COCO compressed RLE string -> run lengths (port of pycocotools ``rleFrString``)."""
    out: list[int] = []
    p = 0
    data = counts.encode("ascii")
    while p < len(data):
        x = 0
        k = 0
        more = True
        while more:
            c = data[p] - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(out) > 2:
            x += out[-2]
        out.append(x)
    return out


def _rle_to_mask(seg: Mapping[str, Any], width: int, height: int) -> BoolMask:
    size = seg.get("size")
    if not size or [int(size[0]), int(size[1])] != [height, width]:
        raise ValueError(f"RLE size {size} does not match image {width}x{height}")
    counts = seg["counts"]
    runs = _decode_compressed_rle(counts) if isinstance(counts, str) else [int(c) for c in counts]
    flat = np.zeros(height * width, dtype=bool)
    pos = 0
    val = False
    for run in runs:
        if val:
            flat[pos : pos + run] = True
        pos += run
        val = not val
    if pos != height * width:
        raise ValueError(f"RLE covers {pos} pixels, expected {height * width}")
    return flat.reshape((width, height)).T  # COCO RLE is column-major


def rasterise_annotations(
    annotations: Sequence[Mapping[str, Any]],
    width: int,
    height: int,
    category_ids: set[int] | None = None,
) -> tuple[BoolMask, dict[str, int]]:
    """Union of the annotations' polygons (or RLE) at the original resolution.

    COCO polygon coordinates are continuous with pixel ``(x, y)`` covering ``[x, x+1) x [y, y+1)``; OpenCV fills
    integer pixel centres, so vertices are shifted by -0.5 px and drawn with 4 fractional bits (``shift=4``). Filled
    polygons include their boundary, so a sub-pixel-thin polygon still marks at least its outline pixels.
    Returns ``(mask, stats)`` with stats counting polygon/RLE/missing segmentations.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    stats: Counter[str] = Counter()
    for ann in annotations:
        if category_ids is not None and int(ann.get("category_id", -1)) not in category_ids:
            stats["skipped_category"] += 1
            continue
        seg = ann.get("segmentation")
        if isinstance(seg, dict):
            mask |= _rle_to_mask(seg, width, height).astype(np.uint8)
            stats["rle"] += 1
            continue
        polys = [p for p in (seg or []) if isinstance(p, list | tuple) and len(p) >= 6]
        if not polys:
            stats["no_polygon"] += 1
            continue
        for poly in polys:
            pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2) - 0.5
            fixed = np.round(pts * 16.0).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [fixed], 1, lineType=cv2.LINE_8, shift=4)
            stats["polygons"] += 1
        stats["annotations_with_polygon"] += 1
    return mask.astype(bool), dict(stats)


# ---------------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------------
def tolerance_px(width: int) -> int:
    """k = round-half-up(8 * W / 1280), at least 1."""
    return max(1, int(math.floor(TOLERANCE_PX_AT_1280 * width / 1280.0 + 0.5)))


def dilate(mask: BoolMask, k: int) -> BoolMask:
    if k <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def lane_confusion(pred: BoolMask, gt: BoolMask) -> PixelCounts:
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape}, gt {gt.shape}")
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(gt & ~pred))
    return PixelCounts(tp, fp, fn, int(pred.size) - tp - fp - fn)


def score_image(pred: BoolMask, gt: BoolMask, k: int) -> dict[str, Any]:
    """Exact, tolerant (both dilated by k) and buffered counts for one image."""
    exact = lane_confusion(pred, gt)
    dp, dg = dilate(pred, k), dilate(gt, k)
    tol = lane_confusion(dp, dg)
    return {
        "exact": exact,
        "tol": tol,
        "buf_pred_px": int(np.count_nonzero(pred)),
        "buf_pred_near_gt_px": int(np.count_nonzero(pred & dg)),
        "buf_gt_px": int(np.count_nonzero(gt)),
        "buf_gt_near_pred_px": int(np.count_nonzero(gt & dp)),
    }


def _ratio(num: int, den: int) -> float | None:
    return None if den <= 0 else num / den


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None or p + r <= 0:
        return None if p is None or r is None else 0.0
    return 2 * p * r / (p + r)


def build_metrics(
    exact: PixelCounts,
    tol: PixelCounts,
    buf: Mapping[str, int],
    n_images: int,
    per_image: Mapping[str, list[float]],
    undefined: Mapping[str, int],
    latency_ms: list[float],
    roundtrip_ms: list[float],
    yolopv2_forward_ms: list[float],
    device: str | None,
    k_values: Mapping[int, int],
) -> dict[str, Metric]:
    m: dict[str, Metric] = {}
    ks = ", ".join(f"k={k} px for {n} images" for k, n in sorted(k_values.items()))
    for label, c, desc in (
        ("lane", exact, "exact pixels"),
        ("lane_tol", tol, f"both masks dilated by a disk of radius k = round(8 W / 1280) ({ks})"),
    ):
        px = f"{n_images} images, {c.gt_road:,} GT lane px, {c.pred_road:,} predicted lane px; sample_count = images"
        for name, value, formula in (
            ("iou", c.iou, "sum TP / sum (TP+FP+FN)"),
            ("precision", c.precision, "sum TP / sum (TP+FP)"),
            ("recall", c.recall, "sum TP / sum (TP+FN)"),
            ("f1", c.f1, "2TP / (2TP+FP+FN)"),
        ):
            key = f"{label}_{name}_overall"
            if n_images == 0:
                m[key] = Metric.missing("ratio", "no evaluated images")
            elif value is None:
                m[key] = Metric.missing("ratio", f"undefined (zero denominator); {px}", sample_count=n_images)
            else:
                m[key] = Metric.measured(
                    round(value, 4), "ratio", n_images, f"lane-marking pixel {name} = {formula}, {desc}; {px}"
                )
    bp = _ratio(buf["pred_near_gt"], buf["pred"])
    br = _ratio(buf["gt_near_pred"], buf["gt"])
    for name, value, formula in (
        ("precision", bp, "predicted lane px within k px of a GT lane px / predicted lane px"),
        ("recall", br, "GT lane px within k px of a predicted lane px / GT lane px"),
        ("f1", _f1(bp, br), "harmonic mean of the buffered precision and recall"),
    ):
        key = f"lane_buffered_{name}_overall"
        if value is None:
            m[key] = Metric.missing("ratio", "undefined (zero denominator)", sample_count=n_images)
        else:
            m[key] = Metric.measured(round(value, 4), "ratio", n_images, f"{formula}, pooled over {n_images} images")
    for series, values in per_image.items():
        n_undef = undefined.get(series, 0)
        base = (
            f"over {len(values)} images with a defined value; {n_undef} images excluded (empty union / no GT and no "
            "prediction); sample_count = images"
        )
        for q in (10, 50, 90):
            key = f"{series}_per_image_p{q}"
            if values:
                m[key] = Metric.measured(
                    round(percentile(values, q), 4),
                    "ratio",
                    len(values),
                    f"{q}th percentile of per-image {series} {base}",
                )
            else:
                m[key] = Metric.missing("ratio", "no image with a defined value")
    m["images_evaluated"] = Metric.measured(n_images, "count", max(n_images, 1), "images scored")
    dev = device or "unknown device"
    for name, values, note in (
        ("inference_latency", latency_ms, f"service timing_ms.total per image ({dev}; every model the service runs)"),
        ("yolopv2_forward", yolopv2_forward_ms, f"service timing_ms.inference_by_model.yolopv2 per image ({dev})"),
        ("request_roundtrip", roundtrip_ms, "client HTTP round trip incl. JPEG upload, masks and JSON"),
    ):
        if values:
            for q in (50, 95):
                m[f"{name}_p{q}_ms"] = Metric.measured(round(percentile(values, q), 1), "ms", len(values), note)
    m["lane_instance_f1"] = Metric.missing(
        "ratio",
        "not evaluated: YOLOPv2 returns a binary lane-line mask without lane instances, so TuSimple/CULane-style "
        "per-lane matching is not possible",
        status="unsupported",
    )
    return m


# ---------------------------------------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------------------------------------
def _load_coco(split_dir: Path) -> dict[str, Any]:
    data = json.loads((Path(split_dir) / ANNOTATIONS_FILE).read_bytes())
    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        raise ValueError(f"{ANNOTATIONS_FILE} is not a COCO document")
    return data


def build_manifest(
    split_dir: Path, dataset: Mapping[str, Any], split: str, out_path: Path | None = None
) -> dict[str, Any]:
    """Deterministic manifest: dataset provenance + one entry per image (sorted by file name) with SHA256, size and
    annotation counts. No timestamps, no signed URLs, no API key."""
    split_dir = Path(split_dir)
    ann_path = split_dir / ANNOTATIONS_FILE
    coco = _load_coco(split_dir)
    cats = {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}
    anns_by_img: dict[int, list[dict[str, Any]]] = {}
    for a in coco.get("annotations", []):
        anns_by_img.setdefault(int(a["image_id"]), []).append(a)
    used_cats = sorted({int(a["category_id"]) for a in coco.get("annotations", [])})
    entries: list[dict[str, Any]] = []
    for img in sorted(coco["images"], key=lambda i: str(i["file_name"])):
        name = str(img["file_name"])
        if "/" in name or "\\" in name or not name.lower().endswith(_IMAGE_EXT):
            raise ValueError(f"unexpected image file name {name!r}")
        path = split_dir / name
        data = path.read_bytes()
        rgb = decode_rgb_png(data)
        h, w = rgb.shape[:2]
        if (w, h) != (int(img["width"]), int(img["height"])):
            raise ValueError(f"{name}: decoded {w}x{h} != COCO {img['width']}x{img['height']}")
        anns = anns_by_img.get(int(img["id"]), [])
        gt, stats = rasterise_annotations(anns, w, h)
        entries.append(
            {
                "file": name,
                "coco_image_id": int(img["id"]),
                "width": w,
                "height": h,
                "size_bytes": len(data),
                "sha256": _sha256(data),
                "annotations": len(anns),
                "annotations_by_class": dict(
                    sorted(Counter(cats.get(int(a["category_id"]), "?") for a in anns).items())
                ),
                "polygons": int(stats.get("polygons", 0)),
                "gt_lane_px": int(np.count_nonzero(gt)),
            }
        )
    sizes = Counter(f"{e['width']}x{e['height']}" for e in entries)
    by_class: Counter[str] = Counter()
    for e in entries:
        by_class.update(e["annotations_by_class"])
    extra_files = sorted(
        p.name
        for p in split_dir.iterdir()
        if p.is_file() and p.name != ANNOTATIONS_FILE and not p.name.lower().endswith(_IMAGE_EXT)
    )
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "manifest_id": dataset["manifest_id"],
        "description": "Roboflow Universe lane-marking split used for the YOLOPv2 lane-line accuracy measurement. "
        "File names and SHA256 hashes only; images and labels are not redistributed.",
        "source": {
            "provider": "Roboflow Universe",
            "workspace": dataset["workspace"],
            "workspace_name": dataset["workspace_name"],
            "project": dataset["project"],
            "project_name": dataset["project_name"],
            "version": dataset["version"],
            "url": dataset["url"],
            "licence": dataset["licence"],
            "licence_id": dataset["licence_id"],
            "licence_url": dataset["licence_url"],
            "attribution": dataset["attribution"],
            "export_format": dataset["export_format"],
            "annotation_type": dataset["annotation_type"],
            "preprocessing": dataset["preprocessing"],
            "augmentation": dataset["augmentation"],
            "obtained_via": "Roboflow REST API export (GET), key from the owner's roboflow.login() store; the signed "
            "download link was used once and not stored",
        },
        "split": split,
        "annotations_file": {
            "name": ANNOTATIONS_FILE,
            "size_bytes": ann_path.stat().st_size,
            "sha256": sha256_file(ann_path),
        },
        "other_files_in_split_dir": extra_files,
        "categories": {str(k): v for k, v in sorted(cats.items())},
        "categories_used_by_annotations": [cats.get(c, "?") for c in used_cats],
        "counts": {
            "images": len(entries),
            "images_with_annotations": sum(1 for e in entries if e["annotations"] > 0),
            "annotations": sum(e["annotations"] for e in entries),
            "polygons": sum(e["polygons"] for e in entries),
            "annotations_by_class": dict(sorted(by_class.items())),
            "image_sizes": dict(sorted(sizes.items())),
        },
        "images": entries,
    }
    if out_path is not None:
        _atomic_write_bytes(Path(out_path), manifest_bytes(manifest))
    return manifest


# ---------------------------------------------------------------------------------------------------------------------
# Prediction decoding and visuals
# ---------------------------------------------------------------------------------------------------------------------
def decode_lane_mask(data: str | None, width: int, height: int) -> tuple[BoolMask, int]:
    """Base64 PNG (uint8 {0,255}, original H x W) -> (bool mask, pixels not in {0,255}); raises on any defect."""
    if not data:
        raise ValueError("lane_mask_png missing")
    try:
        raw = np.frombuffer(base64.b64decode(data, validate=True), dtype=np.uint8)
    except (ValueError, TypeError) as exc:
        raise ValueError("lane_mask_png is not valid base64") from exc
    mask = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError("lane_mask_png is not a decodable PNG")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape != (height, width):
        raise ValueError(f"lane mask is {mask.shape[1]}x{mask.shape[0]}, expected {width}x{height}")
    non_binary = int(np.count_nonzero((mask != 0) & (mask != 255)))
    return mask > 127, non_binary


def render_comparison(rgb: NDArray[np.uint8], gt: BoolMask, pred: BoolMask, title: str) -> NDArray[np.uint8]:
    """Three stacked panels (BGR): input; GT polygons (green); exact prediction (TP green, FP red, FN blue)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)

    def tint(base: NDArray[np.float32], mask: BoolMask, colour: tuple[int, int, int], a: float = 0.75) -> None:
        base[mask] = (1.0 - a) * base[mask] + a * np.asarray(colour, dtype=np.float32)

    gt_panel = bgr.copy() * 0.6
    tint(gt_panel, gt, (0, 255, 0))
    pr_panel = bgr.copy() * 0.6
    tint(pr_panel, pred & gt, (0, 255, 0))
    tint(pr_panel, pred & ~gt, (0, 0, 255))
    tint(pr_panel, ~pred & gt, (255, 0, 0))
    return np.vstack(
        [
            _label(bgr.astype(np.uint8), f"input  {title}"),
            _label(gt_panel.clip(0, 255).astype(np.uint8), "GT lane-marking polygons (green)"),
            _label(pr_panel.clip(0, 255).astype(np.uint8), "YOLOPv2 lane: green TP, red FP, blue FN (exact)"),
        ]
    )


# ---------------------------------------------------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------------------------------------------------
_CSV_FIELDS = (
    "file",
    "width",
    "height",
    "k_px",
    "gt_lane_px",
    "pred_lane_px",
    "tp",
    "fp",
    "fn",
    "iou",
    "precision",
    "recall",
    "f1",
    "tol_iou",
    "tol_precision",
    "tol_recall",
    "tol_f1",
    "buffered_precision",
    "buffered_recall",
    "pred_lane_fraction_service",
    "pred_non_binary_px",
    "canvas_width",
    "canvas_height",
    "scale",
    "service_total_ms",
    "yolopv2_forward_ms",
    "roundtrip_ms",
)


def run_lane_accuracy(
    split_dir: Path,
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
    visuals: bool = True,
) -> dict[str, Any]:
    """Score ``lane_mask_png`` of every manifest image (first ``max_images`` by sorted file name) against the polygon
    GT and persist an Evaluation (returned with ``details``)."""
    split_dir = Path(split_dir)
    manifest_path = Path(manifest_path)
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    manifest_sha = _sha256(raw_manifest)
    ann_sha = sha256_file(split_dir / ANNOTATIONS_FILE)
    if ann_sha != manifest["annotations_file"]["sha256"]:
        raise ValueError(f"{ANNOTATIONS_FILE} sha256 {ann_sha} does not match the manifest")
    coco = _load_coco(split_dir)
    img_by_name = {str(i["file_name"]): i for i in coco["images"]}
    anns_by_img: dict[int, list[dict[str, Any]]] = {}
    for a in coco.get("annotations", []):
        anns_by_img.setdefault(int(a["image_id"]), []).append(a)
    entries: list[dict[str, Any]] = sorted(manifest["images"], key=lambda e: str(e["file"]))
    if max_images is not None:
        entries = entries[: int(max_images)]
    if not entries:
        raise ValueError("manifest has no images")

    eval_id = eval_id or uuid.uuid4().hex
    created = datetime.now(UTC)
    if not client.ready():
        raise InferenceClientError("model_not_ready", f"inference service at {client.base_url} is not ready")
    info = dict(model_info) if model_info is not None else client.model_info().model_dump(mode="json")
    ev_dir = Path(evidence_dir) if evidence_dir is not None else Path(out_dir)

    rows: list[dict[str, Any]] = []
    packed: dict[str, NDArray[np.uint8]] = {}
    failed: list[dict[str, Any]] = []
    latency: list[float] = []
    roundtrip: list[float] = []
    yolo_fwd: list[float] = []
    identities: list[dict[str, Any]] = []
    ensemble_reported: set[str] = set()
    geometries: Counter[str] = Counter()
    k_values: Counter[int] = Counter()
    exact_tot = PixelCounts()
    tol_tot = PixelCounts()
    buf = {"pred": 0, "pred_near_gt": 0, "gt": 0, "gt_near_pred": 0}
    non_binary_total = 0
    total = len(entries)
    for i, entry in enumerate(entries):
        name = str(entry["file"])
        try:
            data = (split_dir / name).read_bytes()
            if _sha256(data) != entry["sha256"]:
                raise ValueError(f"{name} sha256 does not match the manifest")
            rgb = decode_rgb_png(data)
            h, w = rgb.shape[:2]
            if (w, h) != (int(entry["width"]), int(entry["height"])):
                raise ValueError(f"decoded size {w}x{h} != manifest {entry['width']}x{entry['height']}")
            img = img_by_name[name]
            gt, _ = rasterise_annotations(anns_by_img.get(int(img["id"]), []), w, h)
            jpeg = encode_jpeg_bytes(rgb, JPEG_QUALITY)
            t0 = time.perf_counter()
            result = client.infer(jpeg, frame_id=i, conf_threshold=CONF_THRESHOLD, include_masks=True)
            rt = (time.perf_counter() - t0) * 1000.0
            if (result.image.width, result.image.height) != (w, h):
                raise ValueError("service image size differs from the input")
            pred, non_binary = decode_lane_mask(result.lane_mask_png, w, h)
        except (InferenceClientError, httpx.HTTPError, ValueError, OSError, KeyError) as exc:
            failed.append({"file": name, "error": type(exc).__name__, "message": str(exc)[:300]})
            if progress_cb is not None:
                progress_cb(i + 1, total, f"failed:{name}")
            continue
        k = tolerance_px(w)
        k_values[k] += 1
        s = score_image(pred, gt, k)
        ex, tl = s["exact"], s["tol"]
        exact_tot = exact_tot + ex
        tol_tot = tol_tot + tl
        buf["pred"] += s["buf_pred_px"]
        buf["pred_near_gt"] += s["buf_pred_near_gt_px"]
        buf["gt"] += s["buf_gt_px"]
        buf["gt_near_pred"] += s["buf_gt_near_pred_px"]
        packed[name] = np.packbits(pred)
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
        bp = _ratio(s["buf_pred_near_gt_px"], s["buf_pred_px"])
        br = _ratio(s["buf_gt_near_pred_px"], s["buf_gt_px"])
        rows.append(
            {
                "file": name,
                "width": w,
                "height": h,
                "k_px": k,
                "gt_lane_px": ex.gt_road,
                "pred_lane_px": ex.pred_road,
                "tp": ex.tp,
                "fp": ex.fp,
                "fn": ex.fn,
                "iou": _r(ex.iou),
                "precision": _r(ex.precision),
                "recall": _r(ex.recall),
                "f1": _r(ex.f1),
                "tol_iou": _r(tl.iou),
                "tol_precision": _r(tl.precision),
                "tol_recall": _r(tl.recall),
                "tol_f1": _r(tl.f1),
                "buffered_precision": _r(bp),
                "buffered_recall": _r(br),
                "pred_lane_fraction_service": _r(result.mask_stats.lane_fraction),
                "pred_non_binary_px": non_binary,
                "canvas_width": g.canvas_width,
                "canvas_height": g.canvas_height,
                "scale": _r(g.scale),
                "service_total_ms": round(float(result.timing_ms.total), 2),
                "yolopv2_forward_ms": None if fwd is None else round(float(fwd), 2),
                "roundtrip_ms": round(rt, 2),
            }
        )
        if progress_cb is not None:
            progress_cb(i + 1, total, name)

    if not rows:
        raise RuntimeError(f"no image could be evaluated ({len(failed)} failures)")
    if len(failed) > _MAX_FAILED_FRACTION * total:
        raise RuntimeError(f"{len(failed)} of {total} images failed; refusing to report partial accuracy")
    ens = sorted(ensemble_reported)
    if len(ens) > 1:
        raise RuntimeError(f"inconsistent ensemble_sha256 across images: {ens}")

    per_image: dict[str, list[float]] = {}
    undefined: dict[str, int] = {}
    for series, col in (("lane_iou", "iou"), ("lane_f1", "f1"), ("lane_tol_iou", "tol_iou"), ("lane_tol_f1", "tol_f1")):
        per_image[series] = [float(r[col]) for r in rows if r[col] is not None]
        undefined[series] = sum(1 for r in rows if r[col] is None)
    yolo = next((m for m in identities if m.get("model_id") == YOLOPV2_ID), None)
    yolo_sha = str(yolo.get("sha256")) if yolo else ""
    device = identities[0].get("device") if identities else info.get("device")
    metrics = build_metrics(
        exact_tot, tol_tot, buf, len(rows), per_image, undefined, latency, roundtrip, yolo_fwd, device, k_values
    )

    # Visuals: best / median / worst by per-image tolerant IoU (ties broken by file name for determinism).
    ranked = sorted((r for r in rows if r["tol_iou"] is not None), key=lambda r: (float(r["tol_iou"]), r["file"]))
    visual_files: dict[str, dict[str, Any]] = {}
    if visuals and ranked:
        picks = {"worst": ranked[0], "median": ranked[len(ranked) // 2], "best": ranked[-1]}
        for label, row in picks.items():
            name = row["file"]
            rgb = decode_rgb_png((split_dir / name).read_bytes())
            h, w = rgb.shape[:2]
            gt, _ = rasterise_annotations(anns_by_img.get(int(img_by_name[name]["id"]), []), w, h)
            pred = np.unpackbits(packed[name], count=h * w).reshape(h, w).astype(bool)
            title = f"{label}: IoU {row['iou']} tolIoU {row['tol_iou']} (k={row['k_px']})"
            ok, png = cv2.imencode(".png", render_comparison(rgb, gt, pred, title))
            if not ok:
                raise RuntimeError("PNG encoding of a visual comparison failed")
            path = ev_dir / f"{eval_id}_visual_{label}.png"
            _atomic_write_bytes(path, png.tobytes())
            visual_files[label] = {
                "file_in_split": name,
                "iou": row["iou"],
                "tol_iou": row["tol_iou"],
                "file": path.name,
            }

    src = manifest["source"]
    limitations = list(LIMITATIONS_BASE)
    pre = src.get("preprocessing") or {}
    rs = pre.get("resize") if isinstance(pre, dict) else None
    if isinstance(rs, dict):
        limitations.insert(
            0,
            f"Roboflow version preprocessing '{rs.get('format')} {rs.get('width')}x{rs.get('height')}' was applied by "
            "the dataset author before export, so the 'original' images evaluated here are the exported "
            f"{rs.get('width')}x{rs.get('height')} images: camera frames of another aspect ratio are geometrically "
            "distorted (non-uniformly stretched) compared with what YOLOPv2 saw in training (BDD100K, 16:9).",
        )
    if max_images is not None and total < len(manifest["images"]):
        limitations.insert(
            0,
            f"Capped run: the first {total} of {len(manifest['images'])} '{manifest['split']}' images by sorted file "
            f"name (max_images={max_images}), to share CPU with a concurrently running stack.",
        )
    if run_label:
        limitations.insert(0, f"Run label '{run_label}'.")
    if failed:
        limitations.append(f"{len(failed)} of {total} images were excluded (decode/inference failures).")
    if len(identities) > 1:
        limitations.append(
            "The service ran the YOLOPv2 + RF-DETR-N ensemble; the lane mask is produced by YOLOPv2 alone, so "
            "model_sha256 is the YOLOPv2 checkpoint and the ensemble identity is recorded in details. Service latency "
            "includes both models."
        )
    limitations.append(
        f"Licence: {src.get('licence')} as declared by the Roboflow Universe dataset author ({src.get('url')}); the "
        "original capture provenance of the images is not documented by the author. Images and labels are used for "
        "this local measurement only and are not redistributed."
    )

    buf_csv = io.StringIO()
    writer = csv.DictWriter(buf_csv, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    csv_path = ev_dir / f"{eval_id}.per_image.csv"
    _atomic_write_bytes(csv_path, buf_csv.getvalue().encode("utf-8"))

    def counts_dict(c: PixelCounts) -> dict[str, Any]:
        return {
            "tp": c.tp,
            "fp": c.fp,
            "fn": c.fn,
            "tn": c.tn,
            "gt_lane_px": c.gt_road,
            "pred_lane_px": c.pred_road,
            "iou": _r(c.iou),
            "precision": _r(c.precision),
            "recall": _r(c.recall),
            "f1": _r(c.f1),
        }

    doc: dict[str, Any] = {
        "id": eval_id,
        "kind": "video_ml_accuracy",
        "state": "succeeded",
        "created_utc": created.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "source": "learned_model",
        "sample_count": len(rows),
        "metrics": {k: v.model_dump(mode="json") for k, v in metrics.items()},
        "limitations": limitations,
        "code_commit": code_commit if _is_commit(code_commit) else None,
        "model_sha256": yolo_sha if _is_sha(yolo_sha) else None,
        "seeds": None,
        "error_code": None,
        "details": {
            "task": "lane_marking_segmentation",
            "run_label": run_label,
            "dataset": {
                "name": f"Roboflow Universe {src.get('workspace')}/{src.get('project')} v{src.get('version')}",
                "manifest_id": manifest.get("manifest_id"),
                "manifest_file": manifest_path.name,
                "manifest_sha256": manifest_sha,
                "url": src.get("url"),
                "version": src.get("version"),
                "split": manifest.get("split"),
                "licence": src.get("licence"),
                "licence_id": src.get("licence_id"),
                "attribution": src.get("attribution"),
                "export_format": src.get("export_format"),
                "preprocessing": src.get("preprocessing"),
                "annotations_sha256": ann_sha,
                "categories_used": manifest.get("categories_used_by_annotations"),
                "images_in_split": len(manifest["images"]),
                "images_requested": total,
                "images_evaluated": len(rows),
                "images_failed": failed,
                "cap": None if max_images is None else f"first {total} by sorted file name",
            },
            "protocol": {
                "input": "exported image -> RGB -> JPEG q95 (app.video.encode.encode_jpeg_bytes, as the video "
                "pipeline) -> POST /v1/infer via app.perception.client.InferenceClient",
                "prediction": "lane_mask_png (YOLOPv2 lane-line probability > 0.5, unletterboxed to the original "
                "frame by the service) > 127",
                "ground_truth": "union of all annotation polygons of the image (all lane-marking classes), "
                "rasterised at the original resolution with cv2.fillPoly (vertices - 0.5 px, 4 fractional bits)",
                "tolerance": "k = max(1, floor(8 * W / 1280 + 0.5)) px; elliptical (disk) structuring element of "
                "size 2k+1; tolerant metrics dilate BOTH masks; buffered precision/recall dilate only the "
                "reference mask",
                "k_px_counts": {str(k): n for k, n in sorted(k_values.items())},
                "dataset_level": "sum of per-image counts over images, then ratios",
                "per_image": "ratios per image; undefined values (zero denominator) excluded and counted",
                "jpeg_quality": JPEG_QUALITY,
                "conf_threshold": CONF_THRESHOLD,
                "letterbox_geometries": dict(geometries),
                "pred_non_binary_pixels_total": non_binary_total,
            },
            "totals": {
                "images": len(rows),
                "exact": counts_dict(exact_tot),
                "tolerant": counts_dict(tol_tot),
                "buffered": dict(buf),
            },
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
            "evidence": {"per_image_csv": csv_path.name, "visuals": visual_files},
            "code": _git_state(),
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "cpu_count": os.cpu_count(),
            },
        },
    }
    write_evaluation(Path(out_dir), doc)
    return doc


def _is_commit(s: str | None) -> bool:
    return bool(_COMMIT.match(s or ""))


def _is_sha(s: str | None) -> bool:
    return bool(_SHA.match(s or ""))


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------
def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m app.evaluation.lane_accuracy")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-manifest", help="hash every image of one exported split")
    b.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    b.add_argument("--split", required=True)
    b.add_argument("--split-dir", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run", help="evaluate a running inference service on one manifest split")
    r.add_argument("--split-dir", type=Path, required=True)
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--inference-url", required=True)
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--evidence-dir", type=Path, default=None)
    r.add_argument("--code-commit", default="")
    r.add_argument("--timeout-s", type=float, default=60.0)
    r.add_argument("--run-label", default=None)
    r.add_argument("--max-images", type=int, default=None)
    args = p.parse_args(argv)
    if args.cmd == "build-manifest":
        m = build_manifest(args.split_dir, DATASETS[args.dataset], args.split, args.out)
        print(json.dumps({"counts": m["counts"], "categories": m["categories"]}, indent=2))
        return 0
    commit = args.code_commit or (_git_state().get("head") or "")

    def progress(done: int, total: int, label: str) -> None:
        if done % 25 == 0 or done == total or label.startswith("failed"):
            print(f"[{done}/{total}] {label}", flush=True)

    with InferenceClient(args.inference_url, args.timeout_s) as client:
        doc = run_lane_accuracy(
            args.split_dir,
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
