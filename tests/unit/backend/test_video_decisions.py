"""Deterministic analytical video decisions (image-space cues, documented thresholds, reason codes)."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from app.perception.video_decisions import (
    DEFAULT_THRESHOLDS,
    decide,
    frame_quality,
    inference_failed_decision,
    lane_analysis,
    obstacle_cue,
)
from app.perception.world_state import TrackedDetection
from app.schemas.reason_codes import ReasonCode

W, H = 640, 360


def _image(seed: int = 0, low: int = 70, high: int = 190) -> np.ndarray:
    return np.random.default_rng(seed).integers(low, high, size=(H, W, 3), dtype=np.uint8)


def _lanes(left_x: int | None = 200, right_x: int | None = 440, top: int = 200) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    if left_x is not None:
        mask[top:H, left_x : left_x + 4] = True
    if right_x is not None:
        mask[top:H, right_x : right_x + 4] = True
    return mask


def _drivable(top: int = 200) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[top:, 100:540] = True
    return mask


def _vehicle(x1: float, y1: float, x2: float, y2: float, score: float = 0.9, track: int | None = 1) -> TrackedDetection:
    return TrackedDetection(3, "vehicle", score, [x1, y1, x2, y2], track)


def test_clear_road_is_observe_with_valid_lane() -> None:
    d = decide(_image(), _drivable(), _lanes(), [])
    assert d.state == "observe"
    assert d.reason_codes == [ReasonCode.NO_VEHICLE_IN_CORRIDOR, ReasonCode.LANE_VALID]
    assert d.lane_status == "valid"
    assert d.image_lane_offset_normalized == pytest.approx(0.0, abs=0.02)
    assert d.image_obstacle_cue is None
    assert d.ttc_seconds is None
    assert d.quality.status == "valid"
    assert d.lane_points and all(200 <= y < H for _, y in d.lane_points)


def test_lane_offset_sign_and_normalisation() -> None:
    # Lines occupy x 100..103 and 340..343; innermost pixels 103 and 340 -> centre 221.5, half-width 118.5 px.
    status, offset, half, *_ = lane_analysis(_lanes(100, 340))
    assert status == "valid"
    assert half == pytest.approx(118.5)
    assert offset == pytest.approx((320 - 221.5) / 118.5, abs=1e-3)
    assert offset > 0  # positive = camera right of the lane centre
    _, offset_left, *_ = lane_analysis(_lanes(300, 600))
    assert offset_left < 0


def test_one_sided_lane_is_uncertain_and_slows() -> None:
    d = decide(_image(), _drivable(), _lanes(right_x=None), [])
    assert d.lane_status == "uncertain"
    assert d.state == "slow"
    assert d.reason_codes[0] == ReasonCode.LANE_UNCERTAIN
    assert d.image_lane_offset_normalized is not None  # falls back to the drivable-area centre
    assert "lane_offset_from_drivable_area" in d.uncertainty_reasons


def test_missing_lane_mask_is_unavailable() -> None:
    d = decide(_image(), None, None, [])
    assert d.lane_status == "unavailable"
    assert d.image_lane_offset_normalized is None
    assert {"lane_mask_missing", "drivable_mask_missing", "lane_offset_unavailable"} <= set(d.uncertainty_reasons)
    assert ReasonCode.LANE_UNAVAILABLE in d.reason_codes


def test_near_vehicle_in_corridor_requests_controlled_stop() -> None:
    near = _vehicle(220, 200, 420, 350)  # bottom near the image bottom, tall box
    d = decide(_image(), _drivable(), _lanes(), [near])
    assert d.state == "controlled_stop"
    assert d.reason_codes[0] == ReasonCode.VEHICLE_IN_CORRIDOR_NEAR
    assert d.image_obstacle_cue is not None and d.image_obstacle_cue >= DEFAULT_THRESHOLDS.cue_near
    assert d.obstacle_detection_index == 0


def test_mid_range_vehicle_slows_and_far_vehicle_is_informational() -> None:
    mid = _vehicle(270, 210, 370, 290)  # y2=0.81H, h=0.22H -> cue ~0.51
    d = decide(_image(), _drivable(), _lanes(), [mid])
    assert d.state == "slow"
    assert d.reason_codes[0] == ReasonCode.VEHICLE_IN_CORRIDOR
    far = _vehicle(305, 190, 335, 215)  # small box just below the corridor top
    d_far = decide(_image(), _drivable(), _lanes(), [far])
    assert d_far.state == "observe"
    assert ReasonCode.VEHICLE_IN_CORRIDOR in d_far.reason_codes
    assert d_far.image_obstacle_cue is not None and d_far.image_obstacle_cue < DEFAULT_THRESHOLDS.cue_slow


def test_cue_increases_as_vehicle_approaches() -> None:
    cues = []
    for k in range(6):
        y2 = 220 + 25 * k
        h = 30 + 25 * k
        cue, _ = obstacle_cue([_vehicle(320 - h, y2 - h, 320 + h, y2)], W, H)
        cues.append(cue)
    assert all(c is not None for c in cues)
    assert cues == sorted(cues) and cues[-1] > cues[0]
    assert all(0.0 <= c <= 1.0 for c in cues)  # type: ignore[operator]


def test_vehicle_outside_corridor_is_ignored() -> None:
    side = _vehicle(0, 220, 120, 340)  # far left, outside the trapezoid
    d = decide(_image(), _drivable(), _lanes(), [side])
    assert d.image_obstacle_cue is None
    assert ReasonCode.NO_VEHICLE_IN_CORRIDOR in d.reason_codes


def test_low_score_counts_only_when_tracked() -> None:
    untracked = _vehicle(220, 200, 420, 350, score=0.45, track=None)
    tracked = _vehicle(220, 200, 420, 350, score=0.45, track=7)
    assert decide(_image(), _drivable(), _lanes(), [untracked]).image_obstacle_cue is None
    assert decide(_image(), _drivable(), _lanes(), [tracked]).state == "controlled_stop"


@pytest.mark.parametrize("score", [0.1, 0.22, 0.39])
def test_tracked_continuation_below_floor_never_drives_a_decision(score: float) -> None:
    """ByteTrack keeps 0.1-0.4 boxes on confirmed tracks; they must not become decision evidence."""
    assert DEFAULT_THRESHOLDS.tracked_min_score == 0.4
    tracked = _vehicle(220, 200, 420, 350, score=score, track=7)
    d = decide(_image(), _drivable(), _lanes(), [tracked])
    assert d.image_obstacle_cue is None
    assert d.state != "controlled_stop"


def test_unverified_classes_are_excluded() -> None:
    other = TrackedDetection(0, "unverified_class_0", 0.95, [220, 200, 420, 350], None)
    d = decide(_image(), _drivable(), _lanes(), [other])
    assert d.image_obstacle_cue is None
    assert "unverified_class_excluded" in d.uncertainty_reasons


def test_largest_corridor_vehicle_drives_the_cue() -> None:
    small = _vehicle(300, 200, 340, 230, track=1)
    big = _vehicle(240, 220, 400, 340, track=2)
    cue, idx = obstacle_cue([small, big], W, H)
    assert idx == 1 and cue is not None


def test_dark_and_blurry_frames() -> None:
    dark = _image(low=0, high=30)
    qd = frame_quality(dark)
    assert qd.dark and qd.status == "degraded"
    d = decide(dark, _drivable(), _lanes(), [])
    assert d.state == "slow" and ReasonCode.FRAME_DARK in d.reason_codes
    flat = np.full((H, W, 3), 128, dtype=np.uint8)
    qb = frame_quality(flat)
    assert qb.blurry and not qb.dark
    assert ReasonCode.FRAME_BLURRY in decide(flat, _drivable(), _lanes(), []).reason_codes
    sharp = frame_quality(_image())
    assert not sharp.blurry and not sharp.dark


def test_blur_threshold_flags_defocus_not_texture() -> None:
    sharp = _image()
    assert not frame_quality(sharp).blurry
    defocused = cv2.GaussianBlur(sharp, (0, 0), 6)
    q = frame_quality(defocused)
    assert q.blurry and q.laplacian_var < DEFAULT_THRESHOLDS.blur_laplacian_var


def test_drivable_area_low() -> None:
    d = decide(_image(), np.zeros((H, W), dtype=bool), _lanes(), [])
    assert ReasonCode.DRIVABLE_AREA_LOW in d.reason_codes
    assert d.state == "slow"
    assert d.corridor_drivable_fraction == 0.0


def test_mask_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        decide(_image(), np.zeros((10, 10), dtype=bool), None, [])


def test_inference_failed_decision() -> None:
    d = inference_failed_decision(_image())
    assert d.state == "observe"
    assert d.reason_codes == [ReasonCode.INFERENCE_FAILED]
    assert d.lane_status == "unavailable" and d.image_obstacle_cue is None


def test_reason_codes_are_known_and_deterministic() -> None:
    args = (_image(), _drivable(), _lanes(right_x=None), [_vehicle(270, 210, 370, 290)])
    a, b = decide(*args), decide(*args)
    assert a.reason_codes == b.reason_codes and a.state == b.state
    assert a.image_obstacle_cue == b.image_obstacle_cue
    known = {c.value for c in ReasonCode}
    assert set(a.reason_codes) <= known
    assert len(a.reason_codes) == len(set(a.reason_codes))


def test_thresholds_are_configurable() -> None:
    strict = replace(DEFAULT_THRESHOLDS, cue_near=0.3)
    mid = _vehicle(270, 210, 370, 290)
    assert decide(_image(), _drivable(), _lanes(), [mid], strict).state == "controlled_stop"
