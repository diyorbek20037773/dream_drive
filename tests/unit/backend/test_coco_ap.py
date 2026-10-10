"""COCO-style AP on hand-built cases (app/evaluation/coco_ap.py). Agreement with pycocotools on random and real
predictions is checked separately by tools/eval (docs/EVALUATION.md)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from app.evaluation.coco_ap import (
    IOU_THRESHOLDS,
    MAX_DETS,
    RECALL_THRESHOLDS,
    ImageSample,
    bbox_iou_xywh,
    evaluate_class,
    xyxy_to_xywh,
)


def _img(image_id, gt=(), crowd=(), det=(), scores=()):
    return ImageSample.build(image_id, list(gt), list(crowd), list(det), list(scores))


def test_protocol_constants() -> None:
    assert len(IOU_THRESHOLDS) == 10 and IOU_THRESHOLDS[0] == 0.5 and abs(IOU_THRESHOLDS[-1] - 0.95) < 1e-12
    assert len(RECALL_THRESHOLDS) == 101 and RECALL_THRESHOLDS[0] == 0.0 and RECALL_THRESHOLDS[-1] == 1.0
    assert MAX_DETS == 100


def test_perfect_detections_score_one() -> None:
    samples = [
        _img(
            1, gt=[[10, 10, 50, 40], [100, 100, 20, 20]], det=[[10, 10, 50, 40], [100, 100, 20, 20]], scores=[0.9, 0.3]
        ),
        _img(2, gt=[[0, 0, 5, 5]], det=[[0, 0, 5, 5]], scores=[0.5]),
    ]
    r = evaluate_class(samples)
    assert r.ap50 == pytest.approx(1.0)
    assert r.ap50_95 == pytest.approx(1.0)
    assert r.recall50 == pytest.approx(1.0)
    assert r.gt_instances == 3 and r.images_with_gt == 2 and r.detections_scored == 3


def test_no_detections_score_zero() -> None:
    r = evaluate_class([_img(1, gt=[[10, 10, 50, 40]])])
    assert r.ap50 == 0.0 and r.ap50_95 == 0.0 and r.recall50 == 0.0
    assert r.gt_instances == 1 and r.detections_scored == 0


def test_no_ground_truth_is_undefined_not_zero() -> None:
    r = evaluate_class([_img(1, det=[[0, 0, 10, 10]], scores=[0.9])])
    assert r.ap50 is None and r.ap50_95 is None and r.recall50 is None
    assert r.gt_instances == 0 and r.detections_scored == 1


def test_higher_scored_false_positive_halves_precision() -> None:
    gt = [[10, 10, 50, 50]]
    fp_first = evaluate_class([_img(1, gt=gt, det=[[200, 200, 30, 30], [10, 10, 50, 50]], scores=[0.9, 0.8])])
    assert fp_first.ap50 == pytest.approx(0.5)
    assert fp_first.recall50 == pytest.approx(1.0)
    fp_last = evaluate_class([_img(1, gt=gt, det=[[200, 200, 30, 30], [10, 10, 50, 50]], scores=[0.7, 0.8])])
    assert fp_last.ap50 == pytest.approx(1.0)


def test_iou_threshold_sweep() -> None:
    # IoU = 62/100 = 0.62: a match at 0.50, 0.55 and 0.60 only -> AP50:95 = 3/10, AP75 = 0.
    r = evaluate_class([_img(1, gt=[[0, 0, 10, 10]], det=[[0, 0, 10, 6.2]], scores=[0.9])])
    assert r.ap50 == pytest.approx(1.0)
    assert r.ap75 == pytest.approx(0.0)
    assert r.ap50_95 == pytest.approx(0.3)
    assert r.recall50_95 == pytest.approx(0.3)


def test_duplicate_detection_is_a_false_positive() -> None:
    r = evaluate_class([_img(1, gt=[[0, 0, 10, 10]], det=[[0, 0, 10, 10], [0, 0, 10, 10]], scores=[0.9, 0.8])])
    # TP then FP: the precision envelope is 1.0 up to recall 1.0, so AP stays 1 but the FP is counted.
    assert r.ap50 == pytest.approx(1.0)
    r2 = evaluate_class([_img(1, gt=[[0, 0, 10, 10], [50, 50, 10, 10]], det=[[0, 0, 10, 10]] * 2, scores=[0.9, 0.8])])
    assert r2.recall50 == pytest.approx(0.5)
    assert r2.ap50 == pytest.approx(51 / 101)  # precision 1 at recall points 0..0.5, 0 beyond


def test_crowd_region_absorbs_detections_without_penalty() -> None:
    gt = [[0, 0, 10, 10], [100, 100, 80, 80]]
    dets = [[110, 110, 10, 10], [130, 130, 10, 10], [0, 0, 10, 10]]  # two inside the crowd box, scored highest
    with_crowd = evaluate_class([_img(1, gt=gt, crowd=[False, True], det=dets, scores=[0.95, 0.9, 0.5])])
    assert with_crowd.ap50 == pytest.approx(1.0)
    assert with_crowd.gt_instances == 1 and with_crowd.gt_crowd_regions == 1
    as_regular = evaluate_class([_img(1, gt=gt, crowd=[False, False], det=dets, scores=[0.95, 0.9, 0.5])])
    assert as_regular.ap50 is not None and as_regular.ap50 < 1.0
    assert as_regular.gt_instances == 2


def test_crowd_only_image_has_no_positives() -> None:
    r = evaluate_class([_img(1, gt=[[0, 0, 50, 50]], crowd=[True], det=[[0, 0, 10, 10]], scores=[0.9])])
    assert r.ap50 is None and r.gt_instances == 0 and r.gt_crowd_regions == 1


def test_crowd_iou_uses_detection_area() -> None:
    gt = np.array([[0.0, 0.0, 100.0, 100.0]])
    det = np.array([[10.0, 10.0, 10.0, 10.0]])
    assert bbox_iou_xywh(det, gt, np.array([True]))[0, 0] == pytest.approx(1.0)
    assert bbox_iou_xywh(det, gt, np.array([False]))[0, 0] == pytest.approx(100 / 10000)
    assert bbox_iou_xywh(np.array([[200.0, 200.0, 5.0, 5.0]]), gt, np.array([False]))[0, 0] == 0.0


def test_max_dets_truncates_lowest_scores_per_image() -> None:
    fps = [[500 + i, 500, 5, 5] for i in range(MAX_DETS)]
    det = [*fps, [0, 0, 10, 10]]
    scores = [0.9] * MAX_DETS + [0.1]  # the only true positive is the 101st by score
    r = evaluate_class([_img(1, gt=[[0, 0, 10, 10]], det=det, scores=scores)])
    assert r.recall50 == 0.0 and r.detections_scored == MAX_DETS


def test_image_order_does_not_change_result() -> None:
    rng = random.Random(7)  # noqa: S311 - deterministic test data, not cryptography
    samples = []
    for i in range(1, 15):
        gt = [[rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(5, 40), rng.uniform(5, 40)] for _ in range(3)]
        det = [[b[0] + rng.uniform(-3, 3), b[1] + rng.uniform(-3, 3), b[2], b[3]] for b in gt[:2]]
        det.append([300, 300, 10, 10])
        samples.append(_img(i, gt=gt, det=det, scores=[rng.random() for _ in det]))
    a = evaluate_class(samples)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    b = evaluate_class(shuffled)
    assert (a.ap50, a.ap50_95, a.recall50) == (b.ap50, b.ap50_95, b.recall50)
    assert 0.0 < a.ap50_95 < 1.0


def test_xyxy_to_xywh_and_validation() -> None:
    assert xyxy_to_xywh([10, 20, 30, 60]) == [10.0, 20.0, 20.0, 40.0]
    with pytest.raises(ValueError):
        ImageSample.build(1, [[0, 0, 1, 1]], [True, False])
    with pytest.raises(ValueError):
        ImageSample.build(1, det_xywh=[[0, 0, 1, 1]], det_scores=[0.1, 0.2])
