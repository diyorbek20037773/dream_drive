"""Two-model ensemble rules (docs/IMPLEMENTATION_SPEC.md section 2.1). Pure functions, unit-tested.

- Both models run on every frame; the response lists every model that ran and an ensemble identity.
- ``ensemble_sha256`` = SHA256 of the UTF-8 text ``"yolopv2:<sha256>\\nrfdetr_nano:<sha256>\\n"`` (members in this
  fixed order), so one value identifies the exact pair of verified checkpoints.
- Class refinement: an RF-DETR ``bus``/``truck`` box with IoU >= 0.5 to a YOLOPv2 ``vehicle`` box replaces that
  vehicle box. The replacing detection keeps RF-DETR's class, score and box and records
  ``refined_from_model="yolopv2"``. Matching is greedy by RF-DETR score; each vehicle box is replaced at most once.
  Nothing is invented: every output box is one a model produced.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import replace

from app.postprocess import Detection

YOLOPV2_ID = "yolopv2"
RFDETR_ID = "rfdetr_nano"
ENSEMBLE_ID = f"{YOLOPV2_ID}+{RFDETR_ID}"
ENSEMBLE_MEMBERS = (YOLOPV2_ID, RFDETR_ID)
YOLOPV2_VEHICLE_NAME = "vehicle"
REFINING_CLASS_NAMES = frozenset({"bus", "truck"})
REFINE_IOU_THRESHOLD = 0.5


def ensemble_sha256(members: Iterable[tuple[str, str]]) -> str:
    """SHA256 of ``"<model_id>:<sha256>\\n"`` lines in the given order."""
    text = "".join(f"{model_id}:{sha}\n" for model_id, sha in members)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def refine_vehicle_classes(
    yolo: Sequence[Detection], rfdetr: Sequence[Detection], iou_threshold: float = REFINE_IOU_THRESHOLD
) -> list[Detection]:
    """Merge both models' detections, replacing YOLOPv2 vehicle boxes matched by an RF-DETR bus/truck box.

    Returns all detections sorted by score (descending). Inputs are not modified.
    """
    replaced: set[int] = set()
    merged_rf: list[Detection] = []
    order = sorted(range(len(rfdetr)), key=lambda i: rfdetr[i].score, reverse=True)
    refined: dict[int, Detection] = {}
    for i in order:
        det = rfdetr[i]
        if det.class_name not in REFINING_CLASS_NAMES:
            continue
        best_iou, best_j = 0.0, -1
        for j, v in enumerate(yolo):
            if j in replaced or v.class_name != YOLOPV2_VEHICLE_NAME:
                continue
            iou = box_iou(det.xyxy, v.xyxy)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_threshold:
            replaced.add(best_j)
            refined[i] = replace(det, refined_from_model=yolo[best_j].model_id)
    for i, det in enumerate(rfdetr):
        merged_rf.append(refined.get(i, det))
    kept_yolo = [d for j, d in enumerate(yolo) if j not in replaced]
    out = kept_yolo + merged_rf
    out.sort(key=lambda d: d.score, reverse=True)
    return out
