"""COCO-style bounding-box average precision in plain NumPy (no pycocotools dependency in the backend image).

Protocol (mirrors ``pycocotools.cocoeval.COCOeval`` for ``iouType="bbox"``, ``areaRng="all"``, ``useCats=1``):

- IoU thresholds 0.50:0.05:0.95 (10 values), 101 recall points 0.00:0.01:1.00, ``maxDets=100`` per image *and class*.
- Detections of one image/class are sorted by descending score (stable) and truncated to ``maxDets``.
- Ground truth with ``iscrowd=1`` is an *ignore region*: a detection matched to it is neither a TP nor an FP, a crowd
  box may absorb several detections, and its IoU uses the detection area as the denominator (COCO convention).
- Greedy matching per IoU threshold in score order: a detection takes the best still-unmatched non-crowd box with
  IoU >= threshold, and only falls back to a crowd box when no regular box qualifies.
- Precision is made monotonically non-increasing from the right and sampled at the 101 recall points
  (``searchsorted(side="left")``); recall points beyond the achieved recall get precision 0.
- AP50:95 is the mean over all thresholds and recall points; AP50 / AP75 use one threshold; recall is the final
  recall at each threshold (AR@100 is its mean over thresholds).

A class with no non-crowd ground truth has undefined AP (``None``), exactly where pycocotools reports -1.
The implementation is cross-checked against pycocotools on real predictions by ``tools/eval`` (docs/EVALUATION.md).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

IOU_THRESHOLDS: NDArray[np.float64] = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1)
RECALL_THRESHOLDS: NDArray[np.float64] = np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01)) + 1)
MAX_DETS = 100
_AREA_ALL = (0.0, 1e5**2)
_EPS = float(np.spacing(1))


@dataclass
class ImageSample:
    """Ground truth and detections of one class in one image. Boxes are COCO ``[x, y, w, h]`` in pixels."""

    image_id: int
    gt_xywh: NDArray[np.float64]
    gt_iscrowd: NDArray[np.bool_]
    det_xywh: NDArray[np.float64]
    det_scores: NDArray[np.float64]

    @classmethod
    def build(
        cls,
        image_id: int,
        gt_xywh: ArrayLike = (),
        gt_iscrowd: ArrayLike = (),
        det_xywh: ArrayLike = (),
        det_scores: ArrayLike = (),
    ) -> ImageSample:
        g = np.asarray(gt_xywh, dtype=np.float64).reshape(-1, 4)
        c = np.asarray(gt_iscrowd, dtype=bool).reshape(-1)
        d = np.asarray(det_xywh, dtype=np.float64).reshape(-1, 4)
        s = np.asarray(det_scores, dtype=np.float64).reshape(-1)
        if c.size == 0 and g.shape[0]:
            c = np.zeros(g.shape[0], dtype=bool)
        if c.shape[0] != g.shape[0]:
            raise ValueError("gt_iscrowd must have one flag per ground-truth box")
        if s.shape[0] != d.shape[0]:
            raise ValueError("det_scores must have one score per detection")
        return cls(int(image_id), g, c, d, s)


@dataclass
class ClassResult:
    """AP/recall of one class; ``None`` means undefined (no non-crowd ground truth)."""

    ap50_95: float | None
    ap50: float | None
    ap75: float | None
    recall50: float | None
    recall50_95: float | None  # AR@100 for this class (mean recall over the 10 IoU thresholds)
    gt_instances: int  # non-crowd ground-truth boxes (the denominator of recall)
    gt_crowd_regions: int
    images_with_gt: int
    detections_scored: int  # detections after per-image maxDets truncation
    precision_at_recall: list[list[float]] = field(default_factory=list)  # [T][R], for plots / regression checks
    recall_per_threshold: list[float] = field(default_factory=list)

    def as_dict(self, include_curves: bool = False) -> dict[str, object]:
        out: dict[str, object] = {
            "ap50_95": self.ap50_95,
            "ap50": self.ap50,
            "ap75": self.ap75,
            "recall50": self.recall50,
            "recall50_95": self.recall50_95,
            "gt_instances": self.gt_instances,
            "gt_crowd_regions": self.gt_crowd_regions,
            "images_with_gt": self.images_with_gt,
            "detections_scored": self.detections_scored,
        }
        if include_curves:
            out["recall_per_threshold"] = self.recall_per_threshold
        return out


def bbox_iou_xywh(det: NDArray[np.float64], gt: NDArray[np.float64], gt_iscrowd: NDArray[np.bool_]) -> NDArray:
    """IoU matrix ``[len(det), len(gt)]``; for crowd ground truth the union is the detection area (pycocotools)."""
    if det.shape[0] == 0 or gt.shape[0] == 0:
        return np.zeros((det.shape[0], gt.shape[0]), dtype=np.float64)
    dx, dy, dw, dh = (det[:, i : i + 1] for i in range(4))
    gx, gy, gw, gh = (gt[None, :, i] for i in range(4))
    iw = np.minimum(dx + dw, gx + gw) - np.maximum(dx, gx)
    ih = np.minimum(dy + dh, gy + gh) - np.maximum(dy, gy)
    valid = (iw > 0) & (ih > 0)
    inter = np.where(valid, iw * ih, 0.0)
    det_area = dw * dh
    union = np.where(gt_iscrowd[None, :], det_area, det_area + gw * gh - inter)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(valid, inter / np.where(union > 0, union, 1.0), 0.0)
    return iou


@dataclass
class _ImageEval:
    scores: NDArray[np.float64]  # [D] sorted, truncated
    matched: NDArray[np.bool_]  # [T, D]
    det_ignored: NDArray[np.bool_]  # [T, D]
    gt_ignored: NDArray[np.bool_]  # [G]


def _evaluate_image(sample: ImageSample, max_dets: int) -> _ImageEval | None:
    n_gt, n_det = sample.gt_xywh.shape[0], sample.det_xywh.shape[0]
    if n_gt == 0 and n_det == 0:
        return None
    order = np.argsort(-sample.det_scores, kind="mergesort")[:max_dets]
    det = sample.det_xywh[order]
    scores = sample.det_scores[order]
    # areaRng "all": ignore = iscrowd (COCO GT areas are always inside [0, 1e10]).
    gt_ignore_raw = sample.gt_iscrowd.astype(bool)
    gorder = np.argsort(gt_ignore_raw, kind="mergesort")  # regular boxes first, crowd last (stable)
    gt = sample.gt_xywh[gorder]
    gt_crowd = sample.gt_iscrowd[gorder].astype(bool)
    gt_ignored = gt_ignore_raw[gorder]
    ious = bbox_iou_xywh(det, gt, gt_crowd)
    n_t, d_count, g_count = len(IOU_THRESHOLDS), det.shape[0], gt.shape[0]
    gt_taken = np.zeros((n_t, g_count), dtype=bool)
    matched = np.zeros((n_t, d_count), dtype=bool)
    det_ignored = np.zeros((n_t, d_count), dtype=bool)
    if d_count and g_count:
        for ti, thr in enumerate(IOU_THRESHOLDS):
            for di in range(d_count):
                best = min(float(thr), 1 - 1e-10)
                m = -1
                for gi in range(g_count):
                    if gt_taken[ti, gi] and not gt_crowd[gi]:
                        continue
                    if m > -1 and not gt_ignored[m] and gt_ignored[gi]:
                        break  # a regular match exists; do not fall through to crowd regions
                    if ious[di, gi] < best:
                        continue
                    best = float(ious[di, gi])
                    m = gi
                if m == -1:
                    continue
                det_ignored[ti, di] = gt_ignored[m]
                matched[ti, di] = True
                gt_taken[ti, m] = True
    # Unmatched detections outside the area range are ignored (never happens for "all", kept for fidelity).
    area = det[:, 2] * det[:, 3]
    out_of_range = (area < _AREA_ALL[0]) | (area > _AREA_ALL[1])
    det_ignored = det_ignored | (~matched & out_of_range[None, :])
    return _ImageEval(scores, matched, det_ignored, gt_ignored)


def evaluate_class(samples: Iterable[ImageSample], max_dets: int = MAX_DETS) -> ClassResult:
    """COCO AP for one class over images (processed in ascending image id, as pycocotools does)."""
    ordered = sorted(samples, key=lambda s: s.image_id)
    evals = [e for e in (_evaluate_image(s, max_dets) for s in ordered) if e is not None]
    gt_instances = int(sum(int((~s.gt_iscrowd).sum()) for s in ordered))
    gt_crowd = int(sum(int(s.gt_iscrowd.sum()) for s in ordered))
    images_with_gt = int(sum(1 for s in ordered if (~s.gt_iscrowd).any()))
    n_t, n_r = len(IOU_THRESHOLDS), len(RECALL_THRESHOLDS)
    if not evals:
        return ClassResult(None, None, None, None, None, gt_instances, gt_crowd, images_with_gt, 0)
    scores = np.concatenate([e.scores for e in evals])
    matched = np.concatenate([e.matched for e in evals], axis=1)
    det_ignored = np.concatenate([e.det_ignored for e in evals], axis=1)
    gt_ignored = np.concatenate([e.gt_ignored for e in evals])
    n_pos = int(np.count_nonzero(~gt_ignored))
    if n_pos == 0:
        return ClassResult(None, None, None, None, None, gt_instances, gt_crowd, images_with_gt, int(scores.size))
    inds = np.argsort(-scores, kind="mergesort")
    matched = matched[:, inds]
    det_ignored = det_ignored[:, inds]
    tps = matched & ~det_ignored
    fps = ~matched & ~det_ignored
    tp_sum = np.cumsum(tps, axis=1).astype(np.float64)
    fp_sum = np.cumsum(fps, axis=1).astype(np.float64)
    precision = np.zeros((n_t, n_r))
    recall = np.zeros(n_t)
    for t in range(n_t):
        tp, fp = tp_sum[t], fp_sum[t]
        nd = tp.shape[0]
        rc = tp / n_pos
        pr = tp / (fp + tp + _EPS)
        recall[t] = rc[-1] if nd else 0.0
        if nd:
            pr = np.maximum.accumulate(pr[::-1])[::-1]  # precision envelope (non-increasing)
        q = np.zeros(n_r)
        idx = np.searchsorted(rc, RECALL_THRESHOLDS, side="left")
        ok = idx < nd
        q[ok] = pr[idx[ok]]
        precision[t] = q
    t50 = int(np.argmin(np.abs(IOU_THRESHOLDS - 0.5)))
    t75 = int(np.argmin(np.abs(IOU_THRESHOLDS - 0.75)))
    return ClassResult(
        ap50_95=float(precision.mean()),
        ap50=float(precision[t50].mean()),
        ap75=float(precision[t75].mean()),
        recall50=float(recall[t50]),
        recall50_95=float(recall.mean()),
        gt_instances=gt_instances,
        gt_crowd_regions=gt_crowd,
        images_with_gt=images_with_gt,
        detections_scored=int(scores.size),
        precision_at_recall=precision.tolist(),
        recall_per_threshold=recall.tolist(),
    )


def xyxy_to_xywh(box: ArrayLike) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in np.asarray(box, dtype=np.float64).reshape(4))
    return [x1, y1, x2 - x1, y2 - y1]
