"""RF-DETR Nano real-weight smoke: checkpoint identity and the COCO class-id mapping on a real image.

Fixture: tests/fixtures/images/coco_val2017_000000190923.jpg (CC BY 2.0, "IMG_0118" by ttarasiuk) with its COCO
ground-truth boxes (CC BY 4.0). A returned person (101) and traffic_light (110) must each overlap a ground-truth box
of the same COCO category at IoU >= 0.5. With a wrong mapping (e.g. 0-based list positions, where position 9 is
"traffic light" but category 9 is "boat") the traffic light would come back under another id and this fails.
One image is a mapping/plumbing check, not an accuracy measurement.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import cv2
import numpy as np
import pytest
from conftest import COCO_ANNOTATIONS, COCO_IMAGE
from fastapi.testclient import TestClient

from app.ensemble import box_iou
from app.main import create_app
from app.manifest import load_manifest, md5_file, model_path, verify_model_file

CHECKED = {101: 1, 110: 10}  # DreamDrive id -> COCO category id that must be found


@pytest.fixture(scope="module")
def client(real_engine: Any, smoke_settings: Any) -> Any:
    with TestClient(create_app(smoke_settings, real_engine)) as c:
        yield c


def _gt_boxes() -> dict[int, list[list[float]]]:
    raw = json.loads(COCO_ANNOTATIONS.read_text(encoding="utf-8"))
    out: dict[int, list[list[float]]] = {}
    for a in raw["annotations"]:
        if a["iscrowd"]:
            continue
        x, y, w, h = a["bbox_xywh"]
        out.setdefault(a["category_id"], []).append([x, y, x + w, y + h])
    return out


def test_rfdetr_checkpoint_identity(real_engine: Any, smoke_settings: Any, evidence: dict) -> None:
    m = load_manifest(smoke_settings.model_manifest_path, smoke_settings.rfdetr_model_id)
    path = model_path(smoke_settings.model_root, m)
    res = verify_model_file(path, m)
    assert res.ok, res.detail
    assert res.actual_sha256 == m.sha256 == real_engine.rfdetr.verified_sha256
    assert res.actual_size == m.size_bytes == 366287238
    md5 = md5_file(path)
    assert md5 == m.checkpoint_md5 == "fb6504cce7fbdc783f7a46991f07639f"
    evidence["rfdetr_model"]["publisher_md5_verified"] = md5
    assert real_engine.rfdetr.ready and real_engine.ready


def test_rfdetr_person_and_traffic_light_on_coco_fixture(client: Any, evidence: dict) -> None:
    data = COCO_IMAGE.read_bytes()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    r = client.post("/v1/infer?frame_id=190923", content=data, headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image"] == {"width": w, "height": h}
    dets = body["detections"]
    for d in dets:
        x1, y1, x2, y2 = d["xyxy_pixels"]
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h, d
        assert 0.1 <= d["score"] <= 1.0
    rf = [d for d in dets if d["model_id"] == "rfdetr_nano"]
    assert {d["class_id"] for d in rf} <= {101, 102, 104, 106, 108, 110, 113}

    gt = _gt_boxes()
    matches: dict[str, Any] = {}
    for dd_id, coco_id in CHECKED.items():
        best = (0.0, None)
        for d in rf:
            if d["class_id"] != dd_id:
                continue
            for g in gt[coco_id]:
                iou = box_iou(d["xyxy_pixels"], g)
                if iou > best[0]:
                    best = (iou, d)
        matches[str(dd_id)] = {
            "best_iou_with_same_category_gt": round(best[0], 4),
            "score": None if best[1] is None else round(best[1]["score"], 4),
            "box": None if best[1] is None else best[1]["xyxy_pixels"],
        }
        assert best[0] >= 0.5, f"no class {dd_id} box with IoU>=0.5 to a COCO category {coco_id} box: {rf}"
    # No traffic-light ground-truth box is reported under any other enabled id.
    for g in gt[10]:
        for d in rf:
            if box_iou(d["xyxy_pixels"], g) >= 0.5:
                assert d["class_id"] == 110, d

    evidence["coco_fixture"] = {
        "file": "tests/fixtures/images/coco_val2017_000000190923.jpg",
        "sha256": hashlib.sha256(data).hexdigest(),
        "licence": "CC BY 2.0, 'IMG_0118' by ttarasiuk (Flickr 3706332805); annotations CC BY 4.0 (COCO)",
        "width": w,
        "height": h,
        "conf_threshold": 0.1,
        "matches_iou_ge_0_5": matches,
        "rfdetr_detections": [
            {
                "class_id": d["class_id"],
                "class_name": d["class_name"],
                "score": round(d["score"], 4),
                "xyxy_pixels": d["xyxy_pixels"],
                "refined_from_model": d["refined_from_model"],
            }
            for d in rf
        ],
        "yolopv2_detections": sum(1 for d in dets if d["model_id"] == "yolopv2"),
        "timing_ms": body["timing_ms"],
        "note": "Mapping/plumbing check on one labelled image; not an accuracy measurement.",
    }


def test_rfdetr_rgb_order_matters(real_engine: Any) -> None:
    """The service feeds RF-DETR RGB: the same frame with channels swapped (BGR) gives a different result."""
    img = cv2.imdecode(np.frombuffer(COCO_IMAGE.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    a = real_engine.rfdetr.predict(rgb, 0.1)
    b = real_engine.rfdetr.predict(np.ascontiguousarray(img), 0.1)
    assert [(d.class_id, round(d.score, 3)) for d in a] != [(d.class_id, round(d.score, 3)) for d in b]
