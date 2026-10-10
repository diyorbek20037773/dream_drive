"""Vendored ByteTrack + per-job adapter: isolation, low-score second association, timestamp gaps, reset."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from app.perception.tracking import TrackerConfig, VideoTracker, tracked_class_names
from app.perception.vendored.bytetrack import BYTETracker, IdCounter
from app.perception.vendored.bytetrack.matching import ious, linear_assignment
from app.perception.world_state import TrackedDetection

VENDORED = Path(__file__).resolve().parents[3] / "backend" / "app" / "perception" / "vendored" / "bytetrack"


def _det(x1: float, y1: float, x2: float, y2: float, score: float = 0.9, cls: str = "vehicle") -> TrackedDetection:
    return TrackedDetection(3 if cls == "vehicle" else 0, cls, score, [x1, y1, x2, y2])


def _box(cx: float, cy: float = 200.0, w: float = 60.0, h: float = 40.0, score: float = 0.9) -> TrackedDetection:
    return _det(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score)


def _tracker(**kw: float) -> VideoTracker:
    return VideoTracker(frozenset({"vehicle"}), TrackerConfig(frame_rate=5.0, **kw))


def test_vendored_provenance_files_present() -> None:
    notice = (VENDORED / "NOTICE.md").read_text(encoding="utf-8")
    assert "d1bf0191adff59bc8fcfeaa0b33d3d1642552a99" in notice
    license_text = (VENDORED / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in license_text and "Yifu Zhang" in license_text


def test_track_ids_persist_for_a_moving_object() -> None:
    tr = _tracker()
    ids = [tr.update([_box(100 + 10 * k)], 0.2 * k)[0] for k in range(10)]
    assert ids[0] == 1  # first frame births are confirmed immediately (upstream frame_id == 1 rule)
    assert set(ids) == {1}
    summary = tr.track_summaries()
    assert summary == [
        {
            "track_id": 1,
            "class_name": "vehicle",
            "first_t": 0.0,
            "last_t": 1.8,
            "frames": 10,
            "max_score": 0.9,
            "group": "vehicle",
            "class_counts": {"vehicle": 10},
        }
    ]


def test_new_object_after_first_frame_needs_confirmation() -> None:
    tr = _tracker()
    tr.update([_box(100)], 0.0)
    out = tr.update([_box(100.0 + 2), _box(400)], 0.2)
    assert out == [1, None]  # newly born track is unconfirmed on its first frame
    out = tr.update([_box(100.0 + 4), _box(402)], 0.4)
    assert out == [1, 2]


def test_ids_are_isolated_per_job() -> None:
    job_a, job_b = _tracker(), _tracker()
    for k in range(5):
        job_a.update([_box(100 + 5 * k), _box(400 + 5 * k, cy=100)], 0.2 * k)
    assert job_a.ids_issued == 2
    # A second job (e.g. the next upload processed by the same worker) starts at 1, not 3.
    assert job_b.update([_box(250)], 0.0) == [1]
    # The vendored tracker on its own also has no process-global counter.
    t1, t2 = BYTETracker(frame_rate=5.0), BYTETracker(frame_rate=5.0)
    rows = np.array([[10, 10, 50, 50, 0.9]], dtype=float)
    assert t1.update(rows, 0.0)[0].track_id == 1
    assert t2.update(rows, 0.0)[0].track_id == 1


def test_shared_counter_gives_unique_ids_across_classes() -> None:
    tr = VideoTracker(frozenset({"vehicle", "bus"}), TrackerConfig(frame_rate=5.0))
    out = tr.update([_box(100), _det(300, 100, 360, 160, 0.9, cls="bus")], 0.0)
    assert sorted(out) == [1, 2]  # type: ignore[type-var]


def test_second_association_keeps_low_score_continuation() -> None:
    tr = _tracker()
    for k in range(4):
        assert tr.update([_box(100 + 4 * k)], 0.2 * k) == [1]
    # Occlusion-like frames: the detector still sees the car but only with score 0.3 (< high 0.5, > low 0.1).
    for k in range(4, 7):
        assert tr.update([_box(100 + 4 * k, score=0.3)], 0.2 * k) == [1]
    # Back to high confidence: still the same identity.
    assert tr.update([_box(100 + 4 * 7)], 1.4) == [1]
    assert tr.ids_issued == 1


def test_low_score_boxes_never_start_tracks() -> None:
    tr = _tracker()
    for k in range(5):
        assert tr.update([_box(300, score=0.45)], 0.2 * k) == [None]
    assert tr.ids_issued == 0


def test_scores_at_or_below_low_threshold_are_ignored() -> None:
    tr = _tracker()
    tr.update([_box(100)], 0.0)
    assert tr.update([_box(102, score=0.1)], 0.2) == [None]  # strictly greater than low (0.1) is required


def test_without_second_stage_the_track_would_break() -> None:
    """Control experiment: dropping low-score boxes before the tracker loses the identity."""
    tr = _tracker()
    for k in range(4):
        tr.update([_box(100 + 4 * k)], 0.2 * k)
    for k in range(4, 10):
        tr.update([], 0.2 * k)  # low-score boxes filtered away upstream of the tracker
    assert tr.update([_box(100 + 4 * 10)], 2.0) == [None]  # lost > 1 s ago -> expired -> new (unconfirmed) track


def _moving(tr: VideoTracker, times: list[float], speed_px_s: float) -> list[int | None]:
    return [tr.update([_box(100 + speed_px_s * t)], t)[0] for t in times]


def test_timestamp_gap_uses_real_elapsed_time_for_prediction() -> None:
    speed = 150.0  # px/s -> 30 px per 0.2 s step; box width 60 px
    regular = [0.2 * k for k in range(8)]
    gap_t = regular[-1] + 0.8  # four samples missing (VFR gap), still inside the 1.0 s buffer
    tr = _tracker()
    ids = _moving(tr, [*regular, gap_t], speed)
    assert ids == [1] * len(ids), "Kalman prediction over the real 0.8 s gap must re-associate the same object"


def test_frame_count_based_prediction_would_fail_the_same_gap() -> None:
    """Control: feeding the same boxes with timestamps that hide the gap (one nominal step) breaks association."""
    speed = 150.0
    regular = [0.2 * k for k in range(8)]
    tr = _tracker()
    _moving(tr, regular, speed)
    far_box = _box(100 + speed * (regular[-1] + 0.8))
    assert tr.update([far_box], regular[-1] + 0.2) == [None]


def test_lost_track_expires_after_buffer_seconds_even_without_intermediate_frames() -> None:
    tr = _tracker()
    for k in range(3):
        tr.update([_box(200)], 0.2 * k)
    assert tr.update([_box(200)], 0.4 + 0.9) == [1]  # 0.9 s gap < 1.0 s buffer -> same id
    tr2 = _tracker()
    for k in range(3):
        tr2.update([_box(200)], 0.2 * k)
    assert tr2.update([_box(200)], 0.4 + 1.2) == [None]  # 1.2 s gap > buffer -> old track expired
    assert tr2.update([_box(200)], 0.4 + 1.4) == [2]


def test_reset_restarts_ids() -> None:
    tr = _tracker()
    tr.update([_box(100), _box(400)], 0.0)
    assert tr.ids_issued == 2
    tr.reset()
    assert tr.ids_issued == 0 and tr.track_summaries() == []
    assert tr.update([_box(100)], 0.0) == [1]  # time may restart after a reset


def test_non_increasing_timestamps_are_rejected() -> None:
    tr = _tracker()
    tr.update([_box(100)], 0.4)
    with pytest.raises(ValueError):
        tr.update([_box(100)], 0.4)
    vendored = BYTETracker()
    vendored.update(np.zeros((0, 5)), 1.0)
    with pytest.raises(ValueError):
        vendored.update(np.zeros((0, 5)), 0.5)


def test_unverified_classes_are_not_tracked() -> None:
    assert tracked_class_names({"3": "vehicle", "5": "unverified_class_5"}) == frozenset({"vehicle"})
    tr = _tracker()
    assert tr.update([_det(10, 10, 60, 60, 0.95, cls="unverified_class_0")], 0.0) == [None]


def test_id_counter_is_monotonic() -> None:
    c = IdCounter()
    assert [c.next_id() for _ in range(3)] == [1, 2, 3] and c.issued == 3


# ---------------------------------------------------------------------------------------------------------------
# Matching helpers vs. reference semantics
# ---------------------------------------------------------------------------------------------------------------
def test_ious_match_cython_bbox_plus_one_convention() -> None:
    a = np.array([[0, 0, 9, 9]], dtype=float)  # 10x10 under the +1 convention
    b = np.array([[5, 5, 14, 14], [20, 20, 30, 30]], dtype=float)
    out = ious(a, b)
    assert out[0, 0] == pytest.approx(25 / (100 + 100 - 25))
    assert out[0, 1] == 0.0
    assert ious(np.zeros((0, 4)), b).shape == (0, 2)


def _brute_force(cost: np.ndarray, thresh: float) -> float:
    """Optimal value of the lapjv cost_limit formulation by enumeration (each unmatched row/col costs thresh/2)."""
    n, m = cost.shape
    best = float("inf")
    for k in range(min(n, m) + 1):
        for rows in itertools.combinations(range(n), k):
            for cols in itertools.permutations(range(m), k):
                total = sum(cost[r, c] for r, c in zip(rows, cols, strict=True)) + (n - k + m - k) * thresh / 2
                best = min(best, total)
    return best


@pytest.mark.parametrize("seed", range(25))
def test_linear_assignment_is_optimal_under_cost_limit(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n, m = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    cost = rng.random((n, m))
    thresh = 0.6
    matches, ua, ub = linear_assignment(cost, thresh)
    assert all(cost[r, c] <= thresh + 1e-12 for r, c in matches)
    value = sum(cost[r, c] for r, c in matches) + (len(ua) + len(ub)) * thresh / 2
    assert value == pytest.approx(_brute_force(cost, thresh))
    assert len(matches) + len(ua) == n and len(matches) + len(ub) == m


def test_linear_assignment_empty() -> None:
    matches, ua, ub = linear_assignment(np.zeros((0, 3)), 0.8)
    assert matches.shape == (0, 2) and list(ua) == [] and list(ub) == [0, 1, 2]
