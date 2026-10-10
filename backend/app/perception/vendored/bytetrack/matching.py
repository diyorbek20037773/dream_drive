# Ported from ByteTrack yolox/tracker/matching.py @ d1bf0191adff59bc8fcfeaa0b33d3d1642552a99 (MIT, (c) 2021 Yifu Zhang).
# DreamDrive modifications (see NOTICE.md): lap.lapjv replaced by scipy.optimize.linear_sum_assignment on the same
# extended cost matrix; cython_bbox.bbox_overlaps replaced by a numpy port with the same +1 area convention;
# unused re-ID/gating helpers removed; type hints added.
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class _HasTlbrScore(Protocol):
    @property
    def tlbr(self) -> FloatArray: ...

    score: float


def linear_assignment(cost_matrix: FloatArray, thresh: float) -> tuple[IntArray, IntArray, IntArray]:
    """Minimum-cost matching that leaves pairs with cost > thresh unmatched.

    Equivalent formulation to `lap.lapjv(cost, extend_cost=True, cost_limit=thresh)`: the cost matrix is embedded in
    an (n+m) x (n+m) matrix whose dummy entries cost thresh/2 (dummy-dummy 0), so matching a real pair only wins when
    its cost is below the thresh/2 + thresh/2 cost of leaving both unmatched.
    """
    n_rows, n_cols = cost_matrix.shape
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.arange(n_rows, dtype=np.int64), np.arange(n_cols, dtype=np.int64)
    size = n_rows + n_cols
    extended = np.full((size, size), thresh / 2.0, dtype=np.float64)
    extended[n_rows:, n_cols:] = 0.0
    extended[:n_rows, :n_cols] = cost_matrix
    rows, cols = linear_sum_assignment(extended)
    matches = [(r, c) for r, c in zip(rows, cols, strict=True) if r < n_rows and c < n_cols]
    matched_rows = {r for r, _ in matches}
    matched_cols = {c for _, c in matches}
    unmatched_a = np.asarray([r for r in range(n_rows) if r not in matched_rows], dtype=np.int64)
    unmatched_b = np.asarray([c for c in range(n_cols) if c not in matched_cols], dtype=np.int64)
    match_arr = np.asarray(matches, dtype=np.int64).reshape(-1, 2)
    return match_arr, unmatched_a, unmatched_b


def ious(atlbrs: FloatArray | Sequence[FloatArray], btlbrs: FloatArray | Sequence[FloatArray]) -> FloatArray:
    """Pairwise IoU of (x1, y1, x2, y2) boxes with cython_bbox's +1 pixel convention."""
    a = np.asarray(atlbrs, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(btlbrs, dtype=np.float64).reshape(-1, 4)
    out = np.zeros((len(a), len(b)), dtype=np.float64)
    if out.size == 0:
        return out
    area_b = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    area_a = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]) + 1
    ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]) + 1
    inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)
    union = area_a[:, None] + area_b[None, :] - inter
    np.divide(inter, union, out=out, where=union > 0)
    return out


def iou_distance(atracks: Sequence[_HasTlbrScore], btracks: Sequence[_HasTlbrScore]) -> FloatArray:
    atlbrs = [track.tlbr for track in atracks]
    btlbrs = [track.tlbr for track in btracks]
    return 1.0 - ious(atlbrs, btlbrs)


def fuse_score(cost_matrix: FloatArray, detections: Sequence[_HasTlbrScore]) -> FloatArray:
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1.0 - cost_matrix
    det_scores = np.array([det.score for det in detections], dtype=np.float64)
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1.0 - fuse_sim
