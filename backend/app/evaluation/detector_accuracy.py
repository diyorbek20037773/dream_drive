"""Measured object-detector accuracy on a frozen COCO val2017 street-scene subset (docs/EVALUATION.md).

Pipeline: verified COCO annotation archive (SHA256) + committed manifest (``datasets/manifests/``) -> images fetched
on demand from the manifest ``coco_url`` into a local cache (SHA256 pinned in a cache index on first download and
verified afterwards) -> each image decoded to RGB and re-encoded as JPEG q95 exactly like the video pipeline ->
production ``InferenceClient.infer`` -> detections mapped by ``class_name`` (docs/IMPLEMENTATION_SPEC.md 2.1) ->
COCO AP (``app.evaluation.coco_ap``) -> ``Evaluation`` JSON (kind ``video_ml_accuracy``) readable by
``app.evaluation.registry`` plus a raw-prediction file for the pycocotools cross-check in ``tools/eval``.

Only classes the running service actually reports in its class maps are scored; the rest are ``unsupported``. Nothing
here is simulated: without a reachable, ready inference service the run fails instead of producing numbers.

CLI (run from ``backend/`` with the backend venv; complete commands in docs/EVALUATION.md)::

    python -m app.evaluation.detector_accuracy build-manifest --annotations <annotations_trainval2017.zip> --out <json>
    python -m app.evaluation.detector_accuracy run --manifest <json> --annotations <zip> --image-cache <dir>
        --inference-url http://127.0.0.1:18031 --out-dir ../artifacts/evaluations --evidence-dir ../artifacts/qa/spike
"""

from __future__ import annotations

import argparse
import hashlib
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
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np

from app.evaluation.coco_ap import MAX_DETS, ClassResult, ImageSample, evaluate_class, xyxy_to_xywh
from app.evaluation.registry import write_evaluation
from app.perception.client import InferenceClient, InferenceClientError
from app.schemas.report import Metric
from app.video.encode import encode_jpeg_bytes

ProgressCb = Callable[[int, int, str], None]

MANIFEST_ID = "coco_val2017_street_v1"
ANNOTATION_ARCHIVE_NAME = "annotations_trainval2017.zip"
ANNOTATION_ARCHIVE_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATION_ARCHIVE_SHA256 = "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
ANNOTATION_MEMBER = "annotations/instances_val2017.json"
DEFAULT_LIMIT = 300

# Subset rule (docs/EVALUATION.md): at least one non-crowd instance of a SIGNAL category AND of a VEHICLE category.
SIGNAL_CATEGORIES = ("traffic light", "stop sign", "person")
VEHICLE_CATEGORIES = ("car", "bus", "truck", "motorcycle", "bicycle")
COUNTED_CATEGORIES = ("person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign")

JPEG_QUALITY = 95  # identical to app.video.pipeline
ENSEMBLE_MEMBERS = ("yolopv2", "rfdetr_nano")
_COMMIT = re.compile(r"^[a-f0-9]{40,64}$")
_SHA = re.compile(r"^[a-f0-9]{64}$")
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_FAILED_FRACTION = 0.10


@dataclass(frozen=True)
class EvalClass:
    """One scored class: service ``class_name``s that count as predictions vs COCO category names used as GT."""

    name: str
    pred_class_names: tuple[str, ...]
    gt_category_names: tuple[str, ...]
    model_id: str  # which model produces it ("ensemble" = several)
    note: str
    in_enabled_map: bool  # part of map50_enabled_classes (RF-DETR enabled COCO classes, spec 2.1)


EVAL_CLASSES: tuple[EvalClass, ...] = (
    EvalClass("person", ("person",), ("person",), "rfdetr_nano", "RF-DETR-N person vs COCO person", True),
    EvalClass("bicycle", ("bicycle",), ("bicycle",), "rfdetr_nano", "RF-DETR-N bicycle vs COCO bicycle", True),
    EvalClass(
        "motorcycle", ("motorcycle",), ("motorcycle",), "rfdetr_nano", "RF-DETR-N motorcycle vs COCO motorcycle", True
    ),
    EvalClass(
        "bus",
        ("bus",),
        ("bus",),
        "rfdetr_nano",
        "RF-DETR-N bus vs COCO bus (includes boxes that replaced a YOLOPv2 vehicle box in ensemble refinement)",
        True,
    ),
    EvalClass(
        "truck",
        ("truck",),
        ("truck",),
        "rfdetr_nano",
        "RF-DETR-N truck vs COCO truck (includes boxes that replaced a YOLOPv2 vehicle box in ensemble refinement)",
        True,
    ),
    EvalClass(
        "traffic_light",
        ("traffic_light",),
        ("traffic light",),
        "rfdetr_nano",
        "RF-DETR-N traffic_light vs COCO traffic light (box only; the light colour is a separate HSV heuristic and is "
        "not evaluated)",
        True,
    ),
    EvalClass(
        "stop_sign", ("stop_sign",), ("stop sign",), "rfdetr_nano", "RF-DETR-N stop_sign vs COCO stop sign", True
    ),
    EvalClass(
        "vehicle",
        ("vehicle",),
        ("car", "bus", "truck"),
        "yolopv2",
        "cross-dataset check: YOLOPv2 (BDD100K-trained) 'vehicle' boxes vs merged COCO car+bus+truck; labelling "
        "conventions differ and vehicle boxes replaced by an RF-DETR bus/truck box (ensemble refinement) are no "
        "longer 'vehicle'",
        False,
    ),
    EvalClass(
        "vehicle_any",
        ("vehicle", "bus", "truck"),
        ("car", "bus", "truck"),
        "ensemble",
        "ensemble output: any vehicle/bus/truck box vs merged COCO car+bus+truck (what the app draws as a vehicle)",
        False,
    ),
)
EVAL_CLASS_BY_NAME = {c.name: c for c in EVAL_CLASSES}

LIMITATIONS_BASE = [
    "COCO val2017 consists of Flickr photographs, not dash-cam video: viewpoint, object scale, motion blur and "
    "compression differ from DreamDrive's uploaded videos, so these numbers do not transfer to road footage.",
    "RF-DETR-N was trained on COCO train2017; val2017 is its standard held-out split (no overlap with training "
    "images), but the label taxonomy and photographic style are in-domain for it and favour it.",
    "YOLOPv2 'vehicle' (BDD100K-trained, single class) is scored against merged COCO car+bus+truck as a cross-dataset "
    "check; the datasets use different labelling conventions (e.g. what counts as a vehicle, occlusion and truncation "
    "rules), so its AP is indicative only.",
    "Scored at the production confidence threshold through the production /v1/infer path; detections below that "
    "threshold never reach the app, so AP is computed on a truncated precision-recall curve and is not comparable "
    "with leaderboard numbers that use a 0.001 threshold.",
    "Lane-line and drivable-area accuracy are not evaluated: no openly licensed labels are available here (BDD100K "
    "requires owner registration).",
    "Traffic-light colour (red/yellow/green) is an HSV heuristic and is not evaluated; general traffic signs other "
    "than stop signs are unsupported.",
    "Images are JPEG-decoded, then re-encoded at quality 95 before inference (same as the video pipeline), so each "
    "image is compressed twice.",
    "Latency metrics are wall-clock measurements on the evaluation host under whatever concurrent load it had; they "
    "are not throughput guarantees and depend on CPU/GPU, thread count and other containers.",
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


def _atomic_write_json(path: Path, doc: Any, indent: int | None = 2) -> None:
    _atomic_write_bytes(path, (json.dumps(doc, indent=indent, ensure_ascii=False) + "\n").encode("utf-8"))


@dataclass(frozen=True)
class AnnotationSource:
    coco: dict[str, Any]
    archive_sha256: str | None
    member_sha256: str
    path: str


def load_coco_annotations(
    path: Path,
    expected_archive_sha256: str | None = ANNOTATION_ARCHIVE_SHA256,
    expected_member_sha256: str | None = None,
) -> AnnotationSource:
    """Read ``instances_val2017.json`` from the verified archive (or an extracted copy verified by member SHA256)."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        archive_sha = sha256_file(path)
        if expected_archive_sha256 and archive_sha != expected_archive_sha256:
            raise ValueError(f"annotation archive sha256 mismatch: {archive_sha} != {expected_archive_sha256}")
        with zipfile.ZipFile(path) as zf:
            data = zf.read(ANNOTATION_MEMBER)
    else:
        archive_sha = None
        data = path.read_bytes()
    member_sha = hashlib.sha256(data).hexdigest()
    if expected_member_sha256 and member_sha != expected_member_sha256:
        raise ValueError(f"{ANNOTATION_MEMBER} sha256 mismatch: {member_sha} != {expected_member_sha256}")
    if archive_sha is None and not expected_member_sha256:
        raise ValueError("an extracted annotation JSON needs expected_member_sha256 (from the manifest)")
    return AnnotationSource(json.loads(data), archive_sha, member_sha, str(path))


# ---------------------------------------------------------------------------------------------------------------------
# Subset selection (deterministic; no timestamps so re-running reproduces identical bytes)
# ---------------------------------------------------------------------------------------------------------------------
def _category_ids(coco: Mapping[str, Any]) -> dict[str, int]:
    return {str(c["name"]): int(c["id"]) for c in coco["categories"]}


def select_street_subset(coco: Mapping[str, Any], limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    """Manifest body for images with >=1 non-crowd SIGNAL instance and >=1 non-crowd VEHICLE instance."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    cats = _category_ids(coco)
    missing = [n for n in (*SIGNAL_CATEGORIES, *VEHICLE_CATEGORIES, *COUNTED_CATEGORIES) if n not in cats]
    if missing:
        raise ValueError(f"categories missing from the annotation file: {missing}")
    signal_ids = {cats[n] for n in SIGNAL_CATEGORIES}
    vehicle_ids = {cats[n] for n in VEHICLE_CATEGORIES}
    present: dict[int, set[int]] = defaultdict(set)
    for ann in coco["annotations"]:
        if not ann.get("iscrowd", 0):
            present[int(ann["image_id"])].add(int(ann["category_id"]))
    candidates = sorted(i for i, cs in present.items() if cs & signal_ids and cs & vehicle_ids)
    chosen = candidates[:limit]
    chosen_set = set(chosen)
    images_by_id = {int(im["id"]): im for im in coco["images"]}
    licences = {int(lic["id"]): lic for lic in coco.get("licenses", [])}

    counted_ids = {cats[n]: n for n in COUNTED_CATEGORIES}
    non_crowd: Counter[str] = Counter()
    crowd: Counter[str] = Counter()
    with_class: dict[str, set[int]] = defaultdict(set)
    for ann in coco["annotations"]:
        img = int(ann["image_id"])
        name = counted_ids.get(int(ann["category_id"]))
        if img not in chosen_set or name is None:
            continue
        if ann.get("iscrowd", 0):
            crowd[name] += 1
        else:
            non_crowd[name] += 1
            with_class[name].add(img)

    images: list[dict[str, Any]] = []
    licence_use: Counter[int] = Counter()
    for img_id in chosen:
        im = images_by_id[img_id]
        lic_id = int(im["license"])
        lic = licences.get(lic_id, {})
        licence_use[lic_id] += 1
        images.append(
            {
                "id": img_id,
                "file_name": im["file_name"],
                "coco_url": im["coco_url"],
                "flickr_url": im.get("flickr_url"),
                "width": int(im["width"]),
                "height": int(im["height"]),
                "license_id": lic_id,
                "license_name": lic.get("name"),
                "license_url": lic.get("url"),
            }
        )
    return {
        "selection_rule": {
            "text": (
                "COCO 2017 val images that contain at least one non-crowd instance of {traffic light, stop sign, "
                "person} AND at least one non-crowd instance of {car, bus, truck, motorcycle, bicycle}; crowd "
                "(iscrowd=1) annotations never qualify an image (crowd-only images are excluded); qualifying image "
                f"ids sorted ascending, first {limit} taken."
            ),
            "signal_categories": list(SIGNAL_CATEGORIES),
            "vehicle_categories": list(VEHICLE_CATEGORIES),
            "crowd_policy": "iscrowd=1 annotations do not qualify an image; in scoring they are ignore regions",
            "order": "image id ascending",
            "limit": limit,
        },
        "candidate_count": len(candidates),
        "image_count": len(images),
        "class_instance_counts": {
            n: {"non_crowd": non_crowd[n], "crowd": crowd[n], "images": len(with_class[n])} for n in COUNTED_CATEGORIES
        },
        "licenses": [
            {
                "id": lic_id,
                "name": licences.get(lic_id, {}).get("name"),
                "url": licences.get(lic_id, {}).get("url"),
                "image_count": licence_use[lic_id],
            }
            for lic_id in sorted(licence_use)
        ],
        "images": images,
    }


def build_manifest(
    annotations_zip: Path,
    out_path: Path,
    limit: int = DEFAULT_LIMIT,
    expected_archive_sha256: str | None = ANNOTATION_ARCHIVE_SHA256,
) -> dict[str, Any]:
    src = load_coco_annotations(annotations_zip, expected_archive_sha256=expected_archive_sha256)
    body = select_street_subset(src.coco, limit)
    manifest = {
        "schema_version": "1.0",
        "manifest_id": MANIFEST_ID,
        "generated_by": "backend/app/evaluation/detector_accuracy.py build-manifest",
        "source": {
            "dataset": "COCO 2017 (Common Objects in Context)",
            "split": "val2017",
            "archive_name": ANNOTATION_ARCHIVE_NAME,
            "archive_url": ANNOTATION_ARCHIVE_URL,
            "archive_sha256": src.archive_sha256,
            "archive_size_bytes": Path(annotations_zip).stat().st_size,
            "member": ANNOTATION_MEMBER,
            "member_sha256": src.member_sha256,
            "annotation_licence": "CC BY 4.0 (COCO Consortium; https://cocodataset.org/#termsofuse)",
            "image_licence_note": (
                "The COCO Consortium does not own the image copyrights. Each image keeps the Flickr licence listed "
                "per image (license_id/name/url) and use must follow the Flickr Terms of Use. Images are NOT stored "
                "in this repository; they are fetched on demand from coco_url into a local, git-ignored cache."
            ),
        },
        **body,
    }
    _atomic_write_json(Path(out_path), manifest, indent=1)
    return manifest


# ---------------------------------------------------------------------------------------------------------------------
# Image cache (download on demand, SHA256 pinned on first download, verified afterwards)
# ---------------------------------------------------------------------------------------------------------------------
class ImageCache:
    INDEX = "cache_index.json"

    def __init__(self, cache_dir: Path, timeout_s: float = 30.0, retries: int = 3) -> None:
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / self.INDEX
        self.timeout_s = timeout_s
        self.retries = retries
        self.index: dict[str, dict[str, Any]] = {}
        if self.index_path.is_file():
            doc = json.loads(self.index_path.read_text(encoding="utf-8"))
            self.index = dict(doc.get("images", {}))
        self._http: httpx.Client | None = None
        self.downloaded = 0

    def _client(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(
                timeout=httpx.Timeout(self.timeout_s, connect=min(10.0, self.timeout_s)),
                follow_redirects=True,
                headers={"User-Agent": "dreamdrive-eval/1.0"},
            )
        return self._http

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
        self._save()

    def _save(self) -> None:
        _atomic_write_json(self.index_path, {"schema_version": "1.0", "images": dict(sorted(self.index.items()))})

    def get(self, entry: Mapping[str, Any]) -> tuple[bytes, str]:
        """Bytes and sha256 of one manifest image; raises ``ValueError``/``httpx.HTTPError`` on failure."""
        name = str(entry["file_name"])
        if not re.fullmatch(r"[0-9]{12}\.jpg", name):
            raise ValueError(f"unexpected COCO file name {name!r}")
        path = self.dir / name
        pinned = self.index.get(name)
        if path.is_file():
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if pinned and pinned.get("sha256") == digest:
                return data, digest
            if pinned:
                path.unlink()  # corrupted/modified cache entry: re-download and require the pinned hash
        data = self._download(str(entry["coco_url"]))
        digest = hashlib.sha256(data).hexdigest()
        if pinned and pinned.get("sha256") != digest:
            raise ValueError(f"{name}: downloaded sha256 {digest} != pinned {pinned.get('sha256')}")
        _atomic_write_bytes(path, data)
        if not pinned:
            self.index[name] = {
                "sha256": digest,
                "bytes": len(data),
                "url": str(entry["coco_url"]),
                "first_downloaded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            self.downloaded += 1
            if self.downloaded % 25 == 0:
                self._save()
        return data, digest

    def _download(self, url: str) -> bytes:
        if not url.startswith(("http://images.cocodataset.org/", "https://images.cocodataset.org/")):
            raise ValueError(f"refusing non-COCO image url {url!r}")
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                with self._client().stream("GET", url) as resp:
                    resp.raise_for_status()
                    buf = bytearray()
                    for chunk in resp.iter_bytes():
                        buf.extend(chunk)
                        if len(buf) > _MAX_IMAGE_BYTES:
                            raise ValueError(f"{url}: larger than {_MAX_IMAGE_BYTES} bytes")
                    return bytes(buf)
            except (httpx.HTTPError, ValueError) as exc:
                last = exc
                time.sleep(0.5 * (2**attempt))
        assert last is not None
        raise last


# ---------------------------------------------------------------------------------------------------------------------
# Ground truth / prediction mapping
# ---------------------------------------------------------------------------------------------------------------------
def ground_truth_by_class(
    coco: Mapping[str, Any], image_ids: Iterable[int], classes: Iterable[EvalClass] = EVAL_CLASSES
) -> dict[str, dict[int, tuple[list[list[float]], list[bool]]]]:
    """``{eval_class: {image_id: (xywh boxes, iscrowd flags)}}`` for the given images (merged categories allowed)."""
    cats = _category_ids(coco)
    wanted = set(int(i) for i in image_ids)
    classes = list(classes)
    by_cat: dict[int, list[str]] = defaultdict(list)
    for ec in classes:
        for n in ec.gt_category_names:
            by_cat[cats[n]].append(ec.name)
    out: dict[str, dict[int, tuple[list[list[float]], list[bool]]]] = {
        ec.name: {i: ([], []) for i in sorted(wanted)} for ec in classes
    }
    for ann in sorted(coco["annotations"], key=lambda a: int(a["id"])):
        img = int(ann["image_id"])
        if img not in wanted:
            continue
        for ec_name in by_cat.get(int(ann["category_id"]), ()):
            boxes, crowd = out[ec_name][img]
            boxes.append([float(v) for v in ann["bbox"]])
            crowd.append(bool(ann.get("iscrowd", 0)))
    return out


def map_detections(
    detections: Iterable[Mapping[str, Any]], classes: Iterable[EvalClass] = EVAL_CLASSES
) -> dict[str, list[tuple[list[float], float]]]:
    """Detections (``class_name``, ``score``, ``xyxy_pixels``) -> ``{eval_class: [(xywh, score), ...]}`` in input order.

    Unknown names (e.g. ``unverified_class_<id>``) are dropped; one detection may count for several eval classes
    (e.g. ``bus`` for ``bus`` and ``vehicle_any``).
    """
    classes = list(classes)
    out: dict[str, list[tuple[list[float], float]]] = {ec.name: [] for ec in classes}
    for det in detections:
        name = str(det["class_name"])
        for ec in classes:
            if name in ec.pred_class_names:
                out[ec.name].append((xyxy_to_xywh(det["xyxy_pixels"]), float(det["score"])))
    return out


def supported_class_names(model_info: Mapping[str, Any]) -> set[str]:
    """Class names the running service declares (top-level class_map plus every ``models[*].class_map``)."""
    names = {str(v) for v in (model_info.get("class_map") or {}).values()}
    for m in model_info.get("models") or []:
        if isinstance(m, Mapping):
            names |= {str(v) for v in (m.get("class_map") or {}).values()}
    return names


def class_supported(ec: EvalClass, supported: set[str]) -> bool:
    return all(n in supported for n in ec.pred_class_names)


def ensemble_sha256(models: Iterable[Mapping[str, Any]]) -> str | None:
    """sha256 of ``"yolopv2:<sha>\\nrfdetr_nano:<sha>\\n"`` (spec 2.1) when both members are present."""
    shas = {str(m.get("model_id")): str(m.get("sha256")) for m in models if m.get("sha256")}
    if not all(k in shas and _SHA.match(shas[k]) for k in ENSEMBLE_MEMBERS):
        return None
    text = "".join(f"{k}:{shas[k]}\n" for k in ENSEMBLE_MEMBERS)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _decode_rgb(data: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if arr is None:
        raise ValueError("image could not be decoded")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _fetch_model_info(base_url: str, timeout_s: float) -> dict[str, Any]:
    resp = httpx.get(base_url.rstrip("/") + "/v1/model", timeout=timeout_s)
    resp.raise_for_status()
    body = resp.json()
    if not isinstance(body, dict):
        raise ValueError("/v1/model did not return an object")
    return body


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _r(x: float | None, nd: int = 6) -> float | None:
    return None if x is None else round(float(x), nd)


# ---------------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------------
def build_metrics(
    results: Mapping[str, ClassResult],
    supported: Mapping[str, bool],
    images_evaluated: int,
    latency_ms: list[float],
    roundtrip_ms: list[float],
    device: str | None,
) -> dict[str, Metric]:
    m: dict[str, Metric] = {}
    for ec in EVAL_CLASSES:
        res = results.get(ec.name)
        n = res.gt_instances if res else 0
        if not supported.get(ec.name, False):
            note = f"the running inference service does not output {'/'.join(ec.pred_class_names)}"
            for key in (f"ap50_{ec.name}", f"ap50_95_{ec.name}", f"recall50_{ec.name}"):
                m[key] = Metric.missing("ratio", note, status="unsupported", sample_count=n)
            continue
        if res is None or res.ap50 is None or res.ap50_95 is None or res.recall50 is None:
            note = "no non-crowd ground-truth instances of this class in the evaluated images"
            for key in (f"ap50_{ec.name}", f"ap50_95_{ec.name}", f"recall50_{ec.name}"):
                m[key] = Metric.missing("ratio", note, sample_count=0)
            continue
        base = f"{ec.note}; {res.images_with_gt} images with GT, {res.detections_scored} detections scored"
        m[f"ap50_{ec.name}"] = Metric.measured(round(res.ap50, 4), "ratio", n, f"COCO AP@IoU0.50; {base}")
        m[f"ap50_95_{ec.name}"] = Metric.measured(round(res.ap50_95, 4), "ratio", n, f"COCO AP@[.50:.95]; {base}")
        m[f"recall50_{ec.name}"] = Metric.measured(
            round(res.recall50, 4), "ratio", n, f"recall at IoU 0.50, maxDets {MAX_DETS}; {base}"
        )
    enabled = [ec for ec in EVAL_CLASSES if ec.in_enabled_map]
    measured = [results[ec.name] for ec in enabled if supported.get(ec.name) and results.get(ec.name)]
    measured = [r for r in measured if r.ap50 is not None and r.ap50_95 is not None]
    if measured and len(measured) == len(enabled):
        n = sum(r.gt_instances for r in measured)
        names = ", ".join(ec.name for ec in enabled)
        m["map50_enabled_classes"] = Metric.measured(
            round(float(np.mean([r.ap50 for r in measured])), 4),
            "ratio",
            n,
            f"unweighted mean of AP50 over the {len(enabled)} RF-DETR classes enabled in DreamDrive ({names}); "
            "sample_count = their GT instances",
        )
        m["map50_95_enabled_classes"] = Metric.measured(
            round(float(np.mean([r.ap50_95 for r in measured])), 4),
            "ratio",
            n,
            f"unweighted mean of AP@[.50:.95] over the same {len(enabled)} classes",
        )
    else:
        note = f"needs all {len(enabled)} enabled RF-DETR classes; the running service supports {len(measured)} of them"
        m["map50_enabled_classes"] = Metric.missing("ratio", note, status="unsupported")
        m["map50_95_enabled_classes"] = Metric.missing("ratio", note, status="unsupported")
    dev = device or "unknown device"
    if latency_ms:
        m["inference_latency_p50_ms"] = Metric.measured(
            round(_pct(latency_ms, 50), 1), "ms", len(latency_ms), f"service timing_ms.total per image ({dev})"
        )
        m["inference_latency_p95_ms"] = Metric.measured(
            round(_pct(latency_ms, 95), 1), "ms", len(latency_ms), f"service timing_ms.total per image ({dev})"
        )
    if roundtrip_ms:
        m["request_roundtrip_p50_ms"] = Metric.measured(
            round(_pct(roundtrip_ms, 50), 1), "ms", len(roundtrip_ms), "client HTTP round trip incl. masks + JSON"
        )
        m["request_roundtrip_p95_ms"] = Metric.measured(
            round(_pct(roundtrip_ms, 95), 1), "ms", len(roundtrip_ms), "client HTTP round trip incl. masks + JSON"
        )
    m["images_evaluated"] = Metric.measured(images_evaluated, "count", images_evaluated, "COCO val2017 street subset")
    m["lane_accuracy"] = Metric.missing("ratio", "not evaluated: no openly licensed lane labels available")
    m["drivable_area_iou"] = Metric.missing("ratio", "not evaluated: no openly licensed drivable-area labels available")
    m["traffic_light_colour_accuracy"] = Metric.missing(
        "ratio", "not evaluated: COCO has no light-state labels; the colour is an HSV heuristic"
    )
    return m


# ---------------------------------------------------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------------------------------------------------
def run_detector_accuracy(
    manifest_path: Path,
    annotations_path_or_cache: Path,
    image_cache_dir: Path,
    client: InferenceClient,
    out_dir: Path,
    code_commit: str,
    progress_cb: ProgressCb | None = None,
    *,
    eval_id: str | None = None,
    conf_threshold: float = 0.1,
    evidence_dir: Path | None = None,
    run_label: str | None = None,
    max_images: int | None = None,
    extra_details: Mapping[str, Any] | None = None,
    model_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the frozen COCO subset through the inference service and persist an Evaluation (returned with details).

    ``run_label`` (e.g. ``"dry_run_single_model"``) is recorded in the limitations and details so a validation run is
    never mistaken for the release measurement. Raw predictions go to ``evidence_dir/<id>.predictions.json``.
    ``model_info`` (the raw ``/v1/model`` body) is fetched from the service when not given (tests inject it).
    """
    manifest_path = Path(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    src_info = manifest["source"]
    ann = load_coco_annotations(
        Path(annotations_path_or_cache),
        expected_archive_sha256=src_info.get("archive_sha256"),
        expected_member_sha256=src_info.get("member_sha256"),
    )
    entries: list[dict[str, Any]] = sorted(manifest["images"], key=lambda e: int(e["id"]))
    if max_images is not None:
        entries = entries[: int(max_images)]
    if not entries:
        raise ValueError("manifest has no images")

    eval_id = eval_id or uuid.uuid4().hex
    created = datetime.now(UTC)
    if not client.ready():
        raise InferenceClientError("model_not_ready", f"inference service at {client.base_url} is not ready")
    info = dict(model_info) if model_info is not None else _fetch_model_info(client.base_url, client.timeout_s)
    supported_names = supported_class_names(info)
    supported = {ec.name: class_supported(ec, supported_names) for ec in EVAL_CLASSES}

    cache = ImageCache(image_cache_dir)
    preds_by_image: dict[int, list[dict[str, Any]]] = {}
    failed: list[dict[str, Any]] = []
    image_hashes: dict[int, str] = {}
    latency: list[float] = []
    roundtrip: list[float] = []
    identities: list[dict[str, Any]] = []
    ensemble_reported: set[str] = set()
    unknown_names: Counter[str] = Counter()
    class_ids_seen: dict[str, set[int]] = defaultdict(set)
    known_pred_names = {n for ec in EVAL_CLASSES for n in ec.pred_class_names}
    total = len(entries)
    try:
        for i, entry in enumerate(entries):
            img_id = int(entry["id"])
            try:
                data, digest = cache.get(entry)
                rgb = _decode_rgb(data)
                h, w = rgb.shape[:2]
                if (w, h) != (int(entry["width"]), int(entry["height"])):
                    raise ValueError(f"decoded size {w}x{h} != annotated {entry['width']}x{entry['height']}")
                jpeg = encode_jpeg_bytes(rgb, JPEG_QUALITY)
                t0 = time.perf_counter()
                result = client.infer(jpeg, frame_id=i, conf_threshold=conf_threshold, include_masks=True)
                roundtrip.append((time.perf_counter() - t0) * 1000.0)
            except (InferenceClientError, httpx.HTTPError, ValueError, OSError) as exc:
                failed.append({"image_id": img_id, "error": type(exc).__name__, "message": str(exc)[:300]})
                if progress_cb is not None:
                    progress_cb(i + 1, total, f"failed:{img_id}")
                continue
            if (result.image.width, result.image.height) != (w, h):
                failed.append({"image_id": img_id, "error": "size_mismatch", "message": "service image size differs"})
                continue
            image_hashes[img_id] = digest
            latency.append(float(result.timing_ms.total))
            if not identities:
                identities = [m.model_dump(mode="json") for m in getattr(result, "models", None) or [result.model]]
            ens = getattr(result, "ensemble_sha256", None)
            if ens:
                ensemble_reported.add(str(ens))
            dets: list[dict[str, Any]] = []
            for d in result.detections:
                det = d.model_dump(mode="json")
                name = str(det["class_name"])
                if name not in known_pred_names:
                    unknown_names[name] += 1
                class_ids_seen[name].add(int(det["class_id"]))
                dets.append(
                    {
                        "class_name": name,
                        "class_id": int(det["class_id"]),
                        "model_id": det.get("model_id"),
                        "refined_from_model": det.get("refined_from_model"),
                        "score": float(det["score"]),
                        "xyxy_pixels": [float(v) for v in det["xyxy_pixels"]],
                    }
                )
            preds_by_image[img_id] = dets
            if progress_cb is not None:
                progress_cb(i + 1, total, str(img_id))
    finally:
        cache.close()

    evaluated_ids = sorted(preds_by_image)
    if not evaluated_ids:
        raise RuntimeError(f"no image could be evaluated ({len(failed)} failures)")
    if len(failed) > _MAX_FAILED_FRACTION * total:
        raise RuntimeError(f"{len(failed)} of {total} images failed; refusing to report partial accuracy")

    gt = ground_truth_by_class(ann.coco, evaluated_ids)
    results: dict[str, ClassResult] = {}
    for ec in EVAL_CLASSES:
        samples = []
        for img_id in evaluated_ids:
            boxes, crowd = gt[ec.name][img_id]
            mapped = map_detections(preds_by_image[img_id], [ec])[ec.name]
            samples.append(ImageSample.build(img_id, boxes, crowd, [b for b, _ in mapped], [s for _, s in mapped]))
        results[ec.name] = evaluate_class(samples)

    # Identity: the per-result model list first; models only listed by /v1/model (with a sha256) are added.
    known_ids = {str(m.get("model_id")) for m in identities}
    for m in info.get("models") or []:
        if isinstance(m, dict) and m.get("sha256") and str(m.get("model_id")) not in known_ids:
            identities.append({k: m.get(k) for k in ("model_id", "sha256", "device", "precision")})
            known_ids.add(str(m.get("model_id")))
    ens_computed = ensemble_sha256(identities)
    ens_reported = sorted(ensemble_reported)
    if len(ens_reported) > 1:
        raise RuntimeError(f"inconsistent ensemble_sha256 across images: {ens_reported}")
    model_sha = ens_reported[0] if ens_reported else ens_computed
    if model_sha is None and len(identities) == 1:
        model_sha = identities[0].get("sha256")
    device = identities[0].get("device") if identities else info.get("device")

    metrics = build_metrics(results, supported, len(evaluated_ids), latency, roundtrip, device)
    limitations = list(LIMITATIONS_BASE)
    if run_label:
        limitations.insert(0, f"Run label '{run_label}': validation run, not the release measurement.")
    unsupported = [ec.name for ec in EVAL_CLASSES if not supported[ec.name]]
    if unsupported:
        limitations.append(
            "Unsupported by the running inference service (scored as unsupported, not as zero): "
            + ", ".join(unsupported)
        )
    if failed:
        limitations.append(f"{len(failed)} of {total} manifest images were excluded (download/inference failures).")

    hashes_text = "".join(f"{i}:{image_hashes[i]}\n" for i in evaluated_ids)
    git = _git_state()
    doc: dict[str, Any] = {
        "id": eval_id,
        "kind": "video_ml_accuracy",
        "state": "succeeded",
        "created_utc": created.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "source": "learned_model",
        "sample_count": len(evaluated_ids),
        "metrics": {k: v.model_dump(mode="json") for k, v in metrics.items()},
        "limitations": limitations,
        "code_commit": code_commit if _COMMIT.match(code_commit or "") else None,
        "model_sha256": model_sha if model_sha and _SHA.match(model_sha) else None,
        "seeds": None,
        "error_code": None,
        "details": {
            "run_label": run_label,
            "dataset": {
                "manifest_id": manifest.get("manifest_id"),
                "manifest_file": manifest_path.name,
                "manifest_sha256": manifest_sha,
                "annotation_archive_sha256": ann.archive_sha256,
                "annotation_member": ANNOTATION_MEMBER,
                "annotation_member_sha256": ann.member_sha256,
                "images_in_manifest": len(manifest["images"]),
                "images_requested": total,
                "images_evaluated": len(evaluated_ids),
                "images_failed": failed,
                "image_sha256_list_sha256": hashlib.sha256(hashes_text.encode()).hexdigest(),
                "images_downloaded_this_run": cache.downloaded,
            },
            "protocol": {
                "ap": "COCO bbox, IoU .50:.05:.95, 101-point interpolation, maxDets 100 per image and class, "
                "area all, iscrowd = ignore region (app/evaluation/coco_ap.py)",
                "conf_threshold": conf_threshold,
                "jpeg_quality": JPEG_QUALITY,
                "include_masks": True,
                "class_mapping": [
                    {
                        "eval_class": ec.name,
                        "pred_class_names": list(ec.pred_class_names),
                        "gt_coco_categories": list(ec.gt_category_names),
                        "model_id": ec.model_id,
                        "supported_by_service": supported[ec.name],
                        "in_map50_enabled_classes": ec.in_enabled_map,
                    }
                    for ec in EVAL_CLASSES
                ],
            },
            "per_class": {ec.name: results[ec.name].as_dict(include_curves=True) for ec in EVAL_CLASSES},
            "models": identities,
            "ensemble_sha256_reported": ens_reported[0] if ens_reported else None,
            "ensemble_sha256_computed": ens_computed,
            "service_model_info": {
                k: info.get(k) for k in ("model_id", "ready", "sha256", "device", "precision", "torch_version")
            }
            | {
                "class_map": info.get("class_map"),
                "models": [
                    {k: m.get(k) for k in ("model_id", "sha256", "size_bytes", "repository_revision", "class_map")}
                    for m in info.get("models") or []
                    if isinstance(m, dict)
                ],
            },
            "class_ids_seen": {k: sorted(v) for k, v in sorted(class_ids_seen.items())},
            "unscored_class_names": dict(unknown_names),
            "latency_ms": {
                "service_total_p50": _r(_pct(latency, 50), 2) if latency else None,
                "service_total_p95": _r(_pct(latency, 95), 2) if latency else None,
                "roundtrip_p50": _r(_pct(roundtrip, 50), 2) if roundtrip else None,
                "roundtrip_p95": _r(_pct(roundtrip, 95), 2) if roundtrip else None,
                "samples": len(latency),
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
    ev_dir = Path(evidence_dir) if evidence_dir is not None else Path(out_dir)
    _atomic_write_json(
        ev_dir / f"{eval_id}.predictions.json",
        {
            "schema_version": "1.0",
            "evaluation_id": eval_id,
            "run_label": run_label,
            "manifest_sha256": manifest_sha,
            "annotation_member_sha256": ann.member_sha256,
            "conf_threshold": conf_threshold,
            "eval_classes": [
                {
                    "name": ec.name,
                    "pred_class_names": list(ec.pred_class_names),
                    "gt_coco_categories": list(ec.gt_category_names),
                    "supported_by_service": supported[ec.name],
                }
                for ec in EVAL_CLASSES
            ],
            "image_ids": evaluated_ids,
            "predictions": [{"image_id": i, "detections": preds_by_image[i]} for i in evaluated_ids],
            "numpy_results": {ec.name: results[ec.name].as_dict() for ec in EVAL_CLASSES},
        },
        indent=None,
    )
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
    p = argparse.ArgumentParser(prog="python -m app.evaluation.detector_accuracy")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-manifest", help="select the deterministic COCO val2017 street subset")
    b.add_argument("--annotations", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    r = sub.add_parser("run", help="evaluate a running inference service on the subset")
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--annotations", type=Path, required=True)
    r.add_argument("--image-cache", type=Path, required=True)
    r.add_argument("--inference-url", required=True)
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--evidence-dir", type=Path, default=None)
    r.add_argument("--code-commit", default="")
    r.add_argument("--conf-threshold", type=float, default=0.1)
    r.add_argument("--timeout-s", type=float, default=120.0)
    r.add_argument("--run-label", default=None)
    r.add_argument("--max-images", type=int, default=None)
    r.add_argument("--inference-image", default=None, help="image ref/id of the service, recorded in details")
    args = p.parse_args(argv)
    if args.cmd == "build-manifest":
        m = build_manifest(args.annotations, args.out, args.limit)
        print(json.dumps({k: m[k] for k in ("candidate_count", "image_count", "class_instance_counts")}, indent=2))
        return 0
    commit = args.code_commit or (_git_state().get("head") or "")

    def progress(done: int, total: int, label: str) -> None:
        if done % 25 == 0 or done == total or label.startswith("failed"):
            print(f"[{done}/{total}] {label}", flush=True)

    with InferenceClient(args.inference_url, args.timeout_s) as client:
        doc = run_detector_accuracy(
            args.manifest,
            args.annotations,
            args.image_cache,
            client,
            args.out_dir,
            commit,
            progress,
            conf_threshold=args.conf_threshold,
            evidence_dir=args.evidence_dir,
            run_label=args.run_label,
            max_images=args.max_images,
            extra_details={"inference_image": args.inference_image} if args.inference_image else None,
        )
    summary = {
        "id": doc["id"],
        "sample_count": doc["sample_count"],
        "model_sha256": doc["model_sha256"],
        "metrics": {k: (v["value"], v["sample_count"], v["status"]) for k, v in doc["metrics"].items()},
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
