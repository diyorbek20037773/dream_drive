"""RF-DETR integration rules (docs/IMPLEMENTATION_SPEC.md section 2.1), no real weights needed.

- class-id mapping: manifest == spec list, checked against rfdetr's own COCO class list; raw COCO category ids
  (1-based, sparse) are mapped, never 0-based list positions;
- class refinement of YOLOPv2 vehicle boxes by RF-DETR bus/truck at IoU >= 0.5;
- ensemble SHA256 rule;
- offline construction guard: RF-DETR-N builds without any network access and rfdetr downloads are refused.
"""

from __future__ import annotations

import hashlib
import socket
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import REPO_MANIFEST, SPEC_RFDETR_CLASS_MAP

from app.ensemble import ENSEMBLE_ID, box_iou, ensemble_sha256, refine_vehicle_classes
from app.manifest import load_manifest
from app.postprocess import Detection
from app.rfdetr_detector import OfflineViolation, RfDetrDetector, offline_guard

YOLO_SHA = "f2a8c8374203ae3e67ff9c184e931f763957de92a993b23269e4e721627f1f8c"
RF_SHA = "d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d"


# ----------------------------------------------------------------------------------------------- class ids
def test_spec_class_ids_are_100_plus_rfdetr_coco_category_ids() -> None:
    """Every enabled id is 100 + the COCO category id whose name in rfdetr's own list matches (space -> _)."""
    from rfdetr.assets.coco_classes import COCO_CLASS_NAMES, COCO_CLASSES

    m = load_manifest(REPO_MANIFEST, "rfdetr_nano")
    assert m.class_map_wire() == SPEC_RFDETR_CLASS_MAP
    for dd_id, name in m.class_map.items():
        assert COCO_CLASSES[dd_id - 100].replace(" ", "_") == name
    # The 0-based flat name list is NOT the id space: position 9 is "traffic light" but category 9 is "boat".
    assert COCO_CLASS_NAMES[9] == "traffic light" and COCO_CLASSES[9] == "boat"
    assert COCO_CLASSES[10] == "traffic light" and COCO_CLASSES[13] == "stop sign"
    assert 3 in COCO_CLASSES and "103" not in m.class_map_wire()  # COCO car exists but is disabled


class _FixedRuntime:
    """Synthetic stand-in returning given raw rows (test double, not a model)."""

    precision = "fp32"

    def __init__(self, rows: list[tuple[tuple[float, float, float, float], float, int]]) -> None:
        self.rows = rows

    def coco_category_names(self) -> dict[int, str]:
        from rfdetr.assets.coco_classes import COCO_CLASSES

        return dict(COCO_CLASSES)

    def predict(self, rgb: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xyxy = np.array([r[0] for r in self.rows], dtype=np.float64).reshape(-1, 4)
        scores = np.array([r[1] for r in self.rows], dtype=np.float64)
        cls = np.array([r[2] for r in self.rows], dtype=np.int64)
        keep = scores > threshold
        return xyxy[keep], scores[keep], cls[keep]


def _ready_detector(tmp_path: Path, rows: list[Any], make_settings: Any) -> RfDetrDetector:
    payload = b"stand-in" * 10
    (tmp_path / "models").mkdir(exist_ok=True)
    (tmp_path / "models" / "rf-detr-nano.pth").write_bytes(payload)
    from conftest import rfdetr_entry, write_manifest

    manifest = write_manifest(
        tmp_path / "m.yaml",
        file_name="unused.pt",
        sha256="1" * 64,
        size=1,
        rfdetr=rfdetr_entry(sha256=hashlib.sha256(payload).hexdigest(), size=len(payload)),
    )
    settings = make_settings(model_manifest_path=manifest)
    det = RfDetrDetector(settings, builder=lambda path, device, precision, md5: _FixedRuntime(rows))
    assert det.load("cpu"), det.detail
    return det


def test_detector_maps_raw_coco_ids_and_filters(tmp_path: Path, make_settings: Any) -> None:
    rows = [
        ((10.0, 20.0, 50.0, 120.0), 0.91, 1),  # person -> 101
        ((60.0, 5.0, 70.0, 30.0), 0.55, 10),  # traffic light -> 110 (0-based position 9 would be wrong)
        ((80.0, 5.0, 95.0, 20.0), 0.40, 13),  # stop sign -> 113
        ((1.0, 1.0, 5.0, 5.0), 0.99, 9),  # boat (category 9): disabled, dropped
        ((2.0, 1.0, 5.0, 5.0), 0.99, 3),  # car: disabled, dropped
        ((100.0, 50.0, 140.0, 90.0), 0.30, 2),  # bicycle -> 102
        ((150.0, 50.0, 190.0, 90.0), 0.20, 4),  # motorcycle -> 104
        ((-30.0, -10.0, 400.0, 500.0), 0.60, 6),  # bus, clipped to the 320x240 frame -> 106
        ((20.0, 0.0, 30.0, 10.0), 0.099, 8),  # truck below 0.1 -> dropped
        ((0.0, 0.0, 10.0, 10.0), 0.10, 8),  # truck exactly at threshold -> kept (>=)
        ((330.0, 10.0, 360.0, 30.0), 0.80, 1),  # entirely right of the frame -> dropped
    ]
    det = _ready_detector(tmp_path, rows, make_settings)
    out = det.predict(np.zeros((240, 320, 3), np.uint8), 0.1)
    got = [(d.class_id, d.class_name, round(d.score, 3)) for d in out]
    assert got == [
        (101, "person", 0.91),
        (106, "bus", 0.6),
        (110, "traffic_light", 0.55),
        (113, "stop_sign", 0.4),
        (102, "bicycle", 0.3),
        (104, "motorcycle", 0.2),
        (108, "truck", 0.1),
    ]
    assert all(d.model_id == "rfdetr_nano" and d.refined_from_model is None for d in out)
    bus = out[1]
    assert bus.xyxy == (0.0, 0.0, 320.0, 240.0)
    assert out[0].xyxy == (10.0, 20.0, 50.0, 120.0)  # boxes stay in original pixels (no rescaling here)


def test_one_label_per_query_box(tmp_path: Path, make_settings: Any) -> None:
    """The same box under several classes keeps only its top class; a top class that is disabled drops the box."""
    car_box = (10.0, 10.0, 60.0, 40.0)
    truck_box = (100.0, 10.0, 180.0, 90.0)
    rows = [
        (car_box, 0.80, 3),  # car (disabled) is the top class of this box ...
        (car_box, 0.15, 8),  # ... so its truck hypothesis must not be returned
        (truck_box, 0.60, 8),  # truck is the top class here ...
        (truck_box, 0.30, 6),  # ... bus hypothesis of the same box dropped
        (truck_box, 0.20, 3),  # ... car hypothesis dropped
        ((200.0, 10.0, 220.0, 60.0), 0.50, 1),
        ((200.0, 10.0, 220.0, 60.0), 0.12, 2),  # bicycle hypothesis of the person box dropped
    ]
    det = _ready_detector(tmp_path, rows, make_settings)
    out = det.predict(np.zeros((240, 320, 3), np.uint8), 0.1)
    assert [(d.class_id, round(d.score, 2), d.xyxy) for d in out] == [
        (108, 0.6, truck_box),
        (101, 0.5, (200.0, 10.0, 220.0, 60.0)),
    ]


def test_offline_guard_md5_shortcut_is_exact(tmp_path: Path) -> None:
    """Only the verified (path, md5) pair skips rfdetr's re-hash; anything else is checked by rfdetr itself."""
    import rfdetr.assets.model_weights as mw

    f = tmp_path / "rf-detr-nano.pth"
    f.write_bytes(b"abc")
    real = hashlib.md5(b"abc", usedforsecurity=False).hexdigest()
    other = tmp_path / "other.pth"
    other.write_bytes(b"abc")
    original = mw._validate_file_md5
    with offline_guard((f, real)):
        assert mw._validate_file_md5(str(f), real) is True
        assert mw._validate_file_md5(str(f), "0" * 32) is False  # falls through to rfdetr's real check
        assert mw._validate_file_md5(str(other), real) is True  # real check, really matches
    assert mw._validate_file_md5 is original


# ----------------------------------------------------------------------------------------------- refinement
def _d(cls: int, name: str, score: float, box: tuple[float, float, float, float], model: str) -> Detection:
    return Detection(class_id=cls, class_name=name, score=score, xyxy=box, model_id=model)


def test_box_iou() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (10, 10, 20, 20)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert box_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_bus_truck_refine_matching_vehicle_boxes() -> None:
    v1 = _d(3, "vehicle", 0.9, (100, 100, 200, 200), "yolopv2")
    v2 = _d(3, "vehicle", 0.8, (300, 100, 400, 200), "yolopv2")
    v3 = _d(3, "vehicle", 0.7, (500, 100, 600, 200), "yolopv2")
    bus = _d(106, "bus", 0.6, (105, 100, 205, 200), "rfdetr_nano")  # IoU ~0.905 with v1
    truck = _d(108, "truck", 0.5, (300, 100, 380, 200), "rfdetr_nano")  # IoU 0.8 with v2
    far_truck = _d(108, "truck", 0.4, (530, 100, 630, 200), "rfdetr_nano")  # IoU 70/130 = 0.538 with v3
    person = _d(101, "person", 0.95, (120, 120, 140, 190), "rfdetr_nano")  # never refines
    out = refine_vehicle_classes([v1, v2, v3], [bus, truck, far_truck, person])
    assert [(d.class_id, d.model_id, d.refined_from_model) for d in out] == [
        (101, "rfdetr_nano", None),
        (106, "rfdetr_nano", "yolopv2"),
        (108, "rfdetr_nano", "yolopv2"),
        (108, "rfdetr_nano", "yolopv2"),
    ]
    assert out[1].xyxy == bus.xyxy and out[1].score == 0.6  # RF-DETR's own box/score are kept


def test_refinement_threshold_and_one_to_one() -> None:
    v = _d(3, "vehicle", 0.9, (0, 0, 100, 100), "yolopv2")
    below = _d(108, "truck", 0.9, (0, 0, 100, 49), "rfdetr_nano")  # IoU 0.49 -> no refinement
    out = refine_vehicle_classes([v], [below])
    assert [(d.class_id, d.refined_from_model) for d in out] == [(3, None), (108, None)]
    at = _d(108, "truck", 0.9, (0, 0, 100, 50), "rfdetr_nano")  # IoU exactly 0.5 -> refinement
    assert [(d.class_id, d.refined_from_model) for d in refine_vehicle_classes([v], [at])] == [(108, "yolopv2")]
    # Two RF-DETR boxes on one vehicle: the higher score replaces it; the other stays a plain RF-DETR detection.
    bus = _d(106, "bus", 0.7, (0, 0, 100, 100), "rfdetr_nano")
    truck = _d(108, "truck", 0.6, (0, 0, 100, 100), "rfdetr_nano")
    out = refine_vehicle_classes([v], [truck, bus])
    assert [(d.class_id, d.refined_from_model) for d in out] == [(106, "yolopv2"), (108, None)]
    # Unverified YOLOPv2 classes are never refined; inputs are not mutated.
    odd = _d(7, "unverified_class_7", 0.9, (0, 0, 100, 100), "yolopv2")
    out = refine_vehicle_classes([odd], [bus])
    assert [(d.class_id, d.refined_from_model) for d in out] == [(7, None), (106, None)]
    assert bus.refined_from_model is None


def test_refinement_without_rfdetr_boxes_is_identity() -> None:
    v = [_d(3, "vehicle", s, (0, 0, 10 * s, 10), "yolopv2") for s in (0.3, 0.9, 0.5)]
    assert [d.score for d in refine_vehicle_classes(v, [])] == [0.9, 0.5, 0.3]


# ----------------------------------------------------------------------------------------------- ensemble sha
def test_ensemble_sha256_rule() -> None:
    text = f"yolopv2:{YOLO_SHA}\nrfdetr_nano:{RF_SHA}\n"
    assert (
        ensemble_sha256([("yolopv2", YOLO_SHA), ("rfdetr_nano", RF_SHA)])
        == hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    assert ensemble_sha256([("rfdetr_nano", RF_SHA), ("yolopv2", YOLO_SHA)]) != ensemble_sha256(
        [("yolopv2", YOLO_SHA), ("rfdetr_nano", RF_SHA)]
    )
    assert ENSEMBLE_ID == "yolopv2+rfdetr_nano"


# ----------------------------------------------------------------------------------------------- offline
@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every outbound socket connection raises (and is recorded)."""
    attempts: list[Any] = []

    def refuse(self: socket.socket, address: Any) -> None:
        attempts.append(address)
        raise OSError("network disabled in this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    return attempts


def test_offline_switches_are_set_before_rfdetr_import() -> None:
    import os

    import huggingface_hub.constants as hf_constants

    import app.rfdetr_detector  # noqa: F401 - sets the switches at import

    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert hf_constants.HF_HUB_OFFLINE is True


def test_rfdetr_nano_architecture_builds_without_network(no_network: list[Any]) -> None:
    """RF-DETR-N (patch 16) builds its DINOv2 backbone from the bundled config: no hub download is needed.

    pretrain_weights=None so nothing is loaded here; the real checkpoint is covered by tests/model_smoke.
    """
    import torch
    from rfdetr import RFDETRNano

    with offline_guard():
        model = RFDETRNano(pretrain_weights=None, device="cpu")
    assert no_network == []
    n_params = sum(p.numel() for p in model.model.model.parameters())
    assert n_params > 1_000_000
    assert model.model.resolution == 384
    assert isinstance(next(model.model.model.parameters()), torch.Tensor)


def test_rfdetr_download_attempt_is_refused(tmp_path: Path, no_network: list[Any]) -> None:
    """If a checkpoint path were missing, rfdetr would download it; under the guard that raises instead."""
    from rfdetr import RFDETRNano

    missing = tmp_path / "rf-detr-nano.pth"
    with offline_guard(), pytest.raises(OfflineViolation):
        RFDETRNano(pretrain_weights=str(missing), device="cpu")
    assert not missing.exists()
    assert no_network == []
