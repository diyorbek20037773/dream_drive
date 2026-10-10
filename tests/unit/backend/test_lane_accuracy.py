"""Unit tests for the lane-marking accuracy evaluator (no network, synthetic data only)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.evaluation import lane_accuracy as la


def _line_mask(h: int, w: int, x: int, width: int = 3) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[:, x : x + width] = True
    return m


def test_tolerance_px_scales_with_width_and_rounds_half_up() -> None:
    assert la.tolerance_px(1280) == 8
    assert la.tolerance_px(640) == 4
    assert la.tolerance_px(1920) == 12
    assert la.tolerance_px(720) == 5  # 4.5 rounds up (not banker's rounding)
    assert la.tolerance_px(10) == 1  # floor at 1 px


def test_identical_masks_are_perfect_exact_and_tolerant() -> None:
    gt = _line_mask(64, 128, 40)
    s = la.score_image(gt.copy(), gt, 4)
    assert s["exact"].iou == 1.0 and s["exact"].f1 == 1.0
    assert s["tol"].iou == 1.0
    assert s["buf_pred_near_gt_px"] == s["buf_pred_px"] == s["buf_gt_px"] == s["buf_gt_near_pred_px"]


@pytest.mark.parametrize("shift", [1, 2, 3])
def test_small_shift_hurts_exact_but_not_buffered(shift: int) -> None:
    gt = _line_mask(64, 128, 40, width=3)
    pred = _line_mask(64, 128, 40 + shift, width=3)
    s = la.score_image(pred, gt, 4)
    # exact: overlap 3 - shift columns of 3 -> IoU = (3-s)/(3+s)
    assert s["exact"].iou == pytest.approx((3 - shift) / (3 + shift))
    # tolerant (both dilated by 4): overlap much larger, IoU strictly better than exact
    assert s["tol"].iou is not None and s["tol"].iou > s["exact"].iou
    assert s["tol"].iou == pytest.approx((11 - shift) / (11 + shift))  # 3 + 2k columns each
    # buffered: every predicted pixel is within 4 px of GT and vice versa
    assert s["buf_pred_near_gt_px"] == s["buf_pred_px"]
    assert s["buf_gt_near_pred_px"] == s["buf_gt_px"]


def test_shift_beyond_tolerance_is_not_forgiven_by_buffered_metrics() -> None:
    gt = _line_mask(64, 128, 20)
    pred = _line_mask(64, 128, 80)
    s = la.score_image(pred, gt, 4)
    assert s["exact"].tp == 0
    assert s["tol"].tp == 0
    assert s["buf_pred_near_gt_px"] == 0 and s["buf_gt_near_pred_px"] == 0


def test_empty_prediction_and_empty_gt() -> None:
    gt = _line_mask(32, 32, 10)
    empty = np.zeros((32, 32), dtype=bool)
    s = la.score_image(empty, gt, 2)
    assert s["exact"].recall == 0.0 and s["exact"].precision is None and s["exact"].iou == 0.0
    both = la.score_image(empty, empty, 2)
    assert both["exact"].iou is None and both["tol"].iou is None  # undefined, excluded per image
    fp_only = la.score_image(gt, empty, 2)
    assert fp_only["exact"].precision == 0.0 and fp_only["exact"].recall is None


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        la.lane_confusion(np.zeros((4, 4), bool), np.zeros((4, 5), bool))


def test_rasterise_rectangle_polygon_matches_coco_pixel_convention() -> None:
    # COCO polygon covering x in [2, 6), y in [1, 4): exactly pixels x=2..5, y=1..3 (4 x 3 = 12 px)
    ann = {"category_id": 1, "segmentation": [[2, 1, 6, 1, 6, 4, 2, 4]]}
    mask, stats = la.rasterise_annotations([ann], 10, 8)
    assert stats["polygons"] == 1
    ys, xs = np.nonzero(mask)
    assert xs.min() == 2 and xs.max() <= 6 and ys.min() == 1 and ys.max() <= 4
    assert 12 <= int(mask.sum()) <= 20  # boundary pixels are included by fillPoly


def test_rasterise_thin_polygon_is_not_lost_and_union_of_multiple() -> None:
    thin = {"category_id": 1, "segmentation": [[10.2, 0, 10.4, 0, 10.4, 30, 10.2, 30]]}
    other = {"category_id": 2, "segmentation": [[20, 5, 25, 5, 25, 10, 20, 10]]}
    mask, stats = la.rasterise_annotations([thin, other], 40, 32)
    assert mask[:, 9:12].sum() >= 25
    assert mask[6:9, 21:24].all()
    only_cat2, _ = la.rasterise_annotations([thin, other], 40, 32, category_ids={2})
    assert only_cat2[:, 9:12].sum() == 0
    assert stats["annotations_with_polygon"] == 2


def test_rasterise_handles_missing_segmentation_and_rle() -> None:
    no_seg = {"category_id": 1, "segmentation": []}
    mask, stats = la.rasterise_annotations([no_seg], 8, 8)
    assert not mask.any() and stats["no_polygon"] == 1
    # Uncompressed column-major RLE, h=2 w=3: 2 zeros (column 0), 2 ones (column 1), 2 zeros (column 2)
    rle = {"category_id": 1, "segmentation": {"size": [2, 3], "counts": [2, 2, 2]}}
    m, _ = la.rasterise_annotations([rle], 3, 2)
    assert m.tolist() == [[False, True, False], [False, True, False]]


def test_decode_lane_mask_validates_size_and_binarity() -> None:
    raw = np.zeros((4, 6), dtype=np.uint8)
    raw[1, 2] = 255
    raw[2, 3] = 100
    ok, buf = cv2.imencode(".png", raw)
    assert ok
    b64 = base64.b64encode(buf.tobytes()).decode()
    mask, non_binary = la.decode_lane_mask(b64, 6, 4)
    assert mask[1, 2] and not mask[2, 3] and non_binary == 1
    with pytest.raises(ValueError):
        la.decode_lane_mask(b64, 7, 4)
    with pytest.raises(ValueError):
        la.decode_lane_mask(None, 6, 4)


def _make_split(tmp: Path) -> Path:
    split = tmp / "test"
    split.mkdir()
    rng = np.random.default_rng(0)
    images, anns = [], []
    for i, name in enumerate(["b.jpg", "a.jpg"]):
        img = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(split / name), img)
        images.append({"id": i, "file_name": name, "width": 64, "height": 48})
        anns.append({"id": i, "image_id": i, "category_id": 1, "segmentation": [[10, 0, 13, 0, 13, 48, 10, 48]]})
    coco = {"images": images, "annotations": anns, "categories": [{"id": 0, "name": "lanes"}, {"id": 1, "name": "LC"}]}
    (split / la.ANNOTATIONS_FILE).write_text(json.dumps(coco), encoding="utf-8")
    return split


def test_manifest_is_deterministic_and_secret_free(tmp_path: Path) -> None:
    split = _make_split(tmp_path)
    ds = la.DATASETS["rtqi-lane-marking-9qzw7-v2"]
    a = la.build_manifest(split, ds, "test", tmp_path / "m1.json")
    b = la.build_manifest(split, ds, "test", tmp_path / "m2.json")
    assert (tmp_path / "m1.json").read_bytes() == (tmp_path / "m2.json").read_bytes()
    assert a == b
    assert [e["file"] for e in a["images"]] == ["a.jpg", "b.jpg"]  # sorted by file name
    assert a["counts"]["images"] == 2 and a["counts"]["annotations_by_class"] == {"LC": 2}
    text = (tmp_path / "m1.json").read_text(encoding="utf-8")
    assert "api_key" not in text and "created" not in text and "?" not in a["source"]["url"]
    assert a["source"]["licence"] == "CC BY 4.0"
    assert all(len(e["sha256"]) == 64 for e in a["images"])


def test_build_metrics_reports_tolerant_and_per_image_percentiles() -> None:
    exact = la.PixelCounts(tp=10, fp=10, fn=10, tn=70)
    tol = la.PixelCounts(tp=30, fp=5, fn=5, tn=60)
    buf = {"pred": 20, "pred_near_gt": 18, "gt": 20, "gt_near_pred": 16}
    per = {"lane_iou": [0.1, 0.3, 0.5], "lane_tol_iou": [0.5, 0.7, 0.9]}
    m = la.build_metrics(exact, tol, buf, 3, per, {"lane_iou": 1}, [10.0], [12.0], [5.0], "cpu", {4: 3})
    assert m["lane_iou_overall"].value == pytest.approx(10 / 30, abs=1e-4)
    assert m["lane_tol_iou_overall"].value == pytest.approx(30 / 40, abs=1e-4)
    assert m["lane_buffered_precision_overall"].value == pytest.approx(0.9)
    assert m["lane_buffered_recall_overall"].value == pytest.approx(0.8)
    assert m["lane_iou_per_image_p50"].value == pytest.approx(0.3)
    assert m["lane_iou_per_image_p50"].sample_count == 3
    assert m["lane_instance_f1"].status == "unsupported" and m["lane_instance_f1"].value is None
    assert m["images_evaluated"].value == 3
