"""Deterministic analytical decisions for uploaded video (no actuator commands, no learned policy, no LLM).

Inputs are the learned models' masks/detections for one frame plus the frame pixels (for quality checks and the
traffic-light colour heuristic only). Everything here is image-space geometry with documented thresholds; nothing is
a metric distance or a calibrated probability. Speed and depth of the recording vehicle are unknown, so `ttc_seconds`
is always None.

Geometry (fractions of image width W / height H, camera assumed roughly centred and forward-facing):
- Bottom band: rows [0.55 H, 0.95 H). Lane evidence and the lane/drivable centre are measured here; the last 5 % is
  skipped because dash-cams often see their own bonnet there.
- Ego corridor: trapezoid with bottom edge x in [0.25 W, 0.75 W] at y = H and top edge x in [0.45 W, 0.55 W] at
  y = 0.55 H. A box is "in the corridor" when its bottom edge lies below the corridor top and its horizontal overlap
  with the corridor at that row covers >= 30 % of the narrower of the two widths.

Class families (only these class names take part in decisions; anything else, e.g. `unverified_class_<id>` or a
class the service does not enable, is ignored and flagged `unverified_class_excluded`):
- vehicle: `vehicle` (YOLOPv2), `bus`, `truck` (RF-DETR-N; the service relabels a YOLOPv2 vehicle box as bus/truck
  when both models agree, IoU >= 0.5)
- person: `person`;  two_wheeler: `bicycle`, `motorcycle`;  traffic_light: `traffic_light`;  stop_sign: `stop_sign`
Evidence rule for every family: score >= the family minimum (0.5), or attached to a confirmed ByteTrack track AND
score >= `tracked_min_score` (0.4). The inference service returns boxes down to 0.1 so that ByteTrack's second
association can keep tracks alive; those 0.1-0.4 continuation boxes keep the track id but never drive a decision, so
no decision path consumes a box below 0.4.

Signals:
- lane_status: rows of the band containing lane-mask pixels left / right of the image centre. Both sides >= 10 % of
  band rows -> valid; one side -> uncertain; neither (or no mask) -> unavailable.
- image_lane_offset_normalized: (image centre x - reference centre x) / reference half-width, clipped to [-1, 1];
  positive = the camera is right of the reference centre. Reference = lane centre (median midpoint of the innermost
  lane pixels on each side, rows where both exist) when lane_status is valid, else the drivable-area centre (median
  midpoint of the drivable extent per band row with >= 5 % drivable pixels), else None.
- corridor cue in [0, 1] for the largest in-corridor box of a family: mean of
  y_term = clip((y2/H - 0.55) / 0.45) (bottom edge lower in the image = closer) and
  h_term = clip((box height / H) / 0.5). An image-space cue, NOT metres and NOT a collision probability.
  `image_obstacle_cue` reports the largest cue over the vehicle, person and two-wheeler families.
- Person / two-wheeler in the corridor: near when its cue >= 0.50 (lower than the 0.60 vehicle threshold because
  people are smaller and more vulnerable); any in-corridor person or two-wheeler slows.
- Traffic light: boxes with evidence and height >= 1.2 % of H get an HSV colour HEURISTIC
  (app.perception.traffic_light, method `hsv_heuristic`, not model output). A light is "relevant" when its box centre
  lies in the upper central region x in [0.20 W, 0.80 W], y < 0.55 H (above the corridor top). The relevant state is
  red if any relevant light is red, else yellow, else green, else unknown.
- Stop sign: evidence and box height >= 2 % of H anywhere in the frame -> STOP_SIGN_VISIBLE.
- DRIVABLE_AREA_LOW: drivable pixels cover < 25 % of the corridor.
- FRAME_DARK: mean BT.601 luma < 40 / 255. FRAME_BLURRY: variance of the 3x3 Laplacian of the grey image resized to
  640 px width < 100. Calibration on the CC0 daylight dash-cam fixture (docs/DATA_SOURCES.md): sharp frames measured
  14,355-19,067; the same frames with Gaussian defocus sigma 3 / 4 / 6 px (at 1280x720) measured 299-409 / 124-165 /
  38-47, so the threshold flags roughly sigma >= 4.5 px. Mean luma of those frames was 75-107. These are image-quality
  heuristics, not calibrated detectors; low-texture scenes (fog, night) can also fall below the blur threshold.

State precedence (all analytical recommendations, never commands):
controlled_stop: VEHICLE_IN_CORRIDOR_NEAR (vehicle cue >= 0.60), PERSON_IN_CORRIDOR_NEAR (person cue >= 0.50),
  TWO_WHEELER_IN_CORRIDOR with two-wheeler cue >= 0.50, TRAFFIC_LIGHT_RED_HEURISTIC (relevant light red - heuristic)
> slow: VEHICLE_IN_CORRIDOR (cue >= 0.35), PERSON_IN_CORRIDOR, TWO_WHEELER_IN_CORRIDOR, TRAFFIC_LIGHT_YELLOW_HEURISTIC,
  STOP_SIGN_VISIBLE, lane not valid, DRIVABLE_AREA_LOW, FRAME_DARK, FRAME_BLURRY
> observe.
Reason codes come from app.schemas.reason_codes; the state-driving codes are listed first in the order above, then
informational codes (corridor/lane status, TRAFFIC_LIGHT_VISIBLE, TRAFFIC_LIGHT_GREEN_HEURISTIC, other flags).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from app.perception.traffic_light import (
    DEFAULT_LIGHT_THRESHOLDS,
    LightState,
    LightStateResult,
    LightThresholds,
    classify_light,
)
from app.schemas.reason_codes import ReasonCode

BoolMask = NDArray[np.bool_]
RgbImage = NDArray[np.uint8]
LaneStatus = Literal["valid", "uncertain", "unavailable"]
DecisionStateName = Literal["observe", "slow", "controlled_stop"]
QualityStatus = Literal["valid", "degraded"]
ClassFamily = Literal["vehicle", "person", "two_wheeler", "traffic_light", "stop_sign"]

# Enabled class names (docs/IMPLEMENTATION_SPEC.md section 2.1) -> decision family. Nothing else is used.
CLASS_FAMILIES: dict[str, ClassFamily] = {
    "vehicle": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "person": "person",
    "bicycle": "two_wheeler",
    "motorcycle": "two_wheeler",
    "traffic_light": "traffic_light",
    "stop_sign": "stop_sign",
}


def class_family(class_name: str) -> ClassFamily | None:
    return CLASS_FAMILIES.get(class_name)


@dataclass(frozen=True)
class DecisionThresholds:
    band_top: float = 0.55
    band_bottom: float = 0.95
    corridor_top: float = 0.55
    corridor_bottom_half_width: float = 0.25
    corridor_top_half_width: float = 0.05
    corridor_min_overlap: float = 0.30
    lane_side_min_rows: float = 0.10
    drivable_row_min_fraction: float = 0.05
    drivable_corridor_low: float = 0.25
    vehicle_min_score: float = 0.5
    cue_height_full: float = 0.5
    cue_near: float = 0.60
    cue_slow: float = 0.35
    dark_luma: float = 40.0
    blur_laplacian_var: float = 100.0
    blur_eval_width: int = 640
    # Second detector families (RF-DETR-N)
    person_min_score: float = 0.5
    two_wheeler_min_score: float = 0.5
    vru_cue_near: float = 0.50
    light_min_score: float = 0.5
    light_min_height: float = 0.012
    light_region_x_min: float = 0.20
    light_region_x_max: float = 0.80
    light_region_y_max: float = 0.55
    stop_sign_min_score: float = 0.5
    stop_sign_min_height: float = 0.02
    # Floor for tracked low-score continuation boxes (every family); see the module docstring evidence rule.
    tracked_min_score: float = 0.4

    def as_dict(self) -> dict[str, float | int]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


DEFAULT_THRESHOLDS = DecisionThresholds()


class DecisionDetection(Protocol):
    @property
    def class_name(self) -> str: ...

    @property
    def score(self) -> float: ...

    @property
    def xyxy_pixels(self) -> list[float]: ...

    @property
    def track_id(self) -> int | None: ...


@dataclass(frozen=True)
class FrameQuality:
    mean_luma: float
    laplacian_var: float
    dark: bool
    blurry: bool

    @property
    def status(self) -> QualityStatus:
        return "degraded" if (self.dark or self.blurry) else "valid"


@dataclass
class VideoDecision:
    state: DecisionStateName
    reason_codes: list[str]
    lane_status: LaneStatus
    image_lane_offset_normalized: float | None
    image_obstacle_cue: float | None
    quality: FrameQuality
    uncertainty_reasons: list[str]
    lane_points: list[list[float]] = field(default_factory=list)
    obstacle_detection_index: int | None = None
    corridor_drivable_fraction: float | None = None
    lane_rows_left: float | None = None
    lane_rows_right: float | None = None
    ttc_seconds: None = None
    vehicle_cue: float | None = None
    person_cue: float | None = None
    person_detection_index: int | None = None
    two_wheeler_cue: float | None = None
    two_wheeler_detection_index: int | None = None
    # Aligned with the input detections: an HSV heuristic result for evaluated traffic lights, None otherwise.
    light_states: list[LightStateResult | None] = field(default_factory=list)
    relevant_light_indices: list[int] = field(default_factory=list)
    light_state: LightState | None = None  # aggregated over relevant lights; None = no relevant light
    stop_sign_indices: list[int] = field(default_factory=list)

    def measurements(self) -> dict[str, float | None]:
        def r(v: float | None) -> float | None:
            return None if v is None else round(float(v), 4)

        return {
            "mean_luma": r(self.quality.mean_luma),
            "laplacian_var": r(self.quality.laplacian_var),
            "corridor_drivable_fraction": r(self.corridor_drivable_fraction),
            "lane_rows_left": r(self.lane_rows_left),
            "lane_rows_right": r(self.lane_rows_right),
            "vehicle_cue": r(self.vehicle_cue),
            "person_cue": r(self.person_cue),
            "two_wheeler_cue": r(self.two_wheeler_cue),
        }

    def highlight_indices(self) -> set[int]:
        """Detections that drive a non-informational code (drawn emphasised in the overlay)."""
        out: set[int] = set()
        if self.obstacle_detection_index is not None:
            out.add(self.obstacle_detection_index)
        for idx in (self.person_detection_index, self.two_wheeler_detection_index):
            if idx is not None:
                out.add(idx)
        if self.light_state in ("red", "yellow"):
            out.update(i for i in self.relevant_light_indices if self._light(i) == self.light_state)
        out.update(self.stop_sign_indices)
        return out

    def _light(self, index: int) -> LightState | None:
        if 0 <= index < len(self.light_states):
            res = self.light_states[index]
            return None if res is None else res.state
        return None


def frame_quality(image: RgbImage, th: DecisionThresholds = DEFAULT_THRESHOLDS) -> FrameQuality:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_luma = float(gray.mean())
    height, width = gray.shape
    if width > th.blur_eval_width:
        new_h = max(1, round(height * th.blur_eval_width / width))
        gray = cv2.resize(gray, (th.blur_eval_width, new_h), interpolation=cv2.INTER_AREA)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var())
    return FrameQuality(
        mean_luma=mean_luma,
        laplacian_var=lap_var,
        dark=mean_luma < th.dark_luma,
        blurry=lap_var < th.blur_laplacian_var,
    )


def corridor_polygon(width: int, height: int, th: DecisionThresholds = DEFAULT_THRESHOLDS) -> NDArray[np.int32]:
    """Corridor trapezoid vertices (x, y) in pixels: bottom-left, bottom-right, top-right, top-left."""
    cx = width / 2.0
    top_y = th.corridor_top * height
    pts = [
        (cx - th.corridor_bottom_half_width * width, height - 1),
        (cx + th.corridor_bottom_half_width * width, height - 1),
        (cx + th.corridor_top_half_width * width, top_y),
        (cx - th.corridor_top_half_width * width, top_y),
    ]
    return np.round(np.asarray(pts)).astype(np.int32)


@lru_cache(maxsize=8)
def _corridor_mask(width: int, height: int, th: DecisionThresholds) -> BoolMask:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [corridor_polygon(width, height, th)], 1)
    return mask.astype(bool)


def _corridor_x_extent(y: float, width: int, height: int, th: DecisionThresholds) -> tuple[float, float] | None:
    top_y = th.corridor_top * height
    if y < top_y:
        return None
    frac = min(1.0, (y - top_y) / max(1.0, (height - 1) - top_y))
    half = (th.corridor_top_half_width + frac * (th.corridor_bottom_half_width - th.corridor_top_half_width)) * width
    cx = width / 2.0
    return cx - half, cx + half


def _band_rows(height: int, th: DecisionThresholds) -> tuple[int, int]:
    return int(th.band_top * height), max(int(th.band_top * height) + 1, int(th.band_bottom * height))


def lane_analysis(
    lane_mask: BoolMask | None, th: DecisionThresholds = DEFAULT_THRESHOLDS
) -> tuple[LaneStatus, float | None, float | None, float | None, float | None, list[list[float]]]:
    """Returns (status, offset, half_width_px, rows_left, rows_right, lane_centre_points)."""
    if lane_mask is None:
        return "unavailable", None, None, None, None, []
    height, width = lane_mask.shape
    cx = width // 2
    y0, y1 = _band_rows(height, th)
    band = lane_mask[y0:y1]
    left, right = band[:, :cx], band[:, cx:]
    has_left, has_right = left.any(axis=1), right.any(axis=1)
    rows = max(1, band.shape[0])
    rows_left, rows_right = float(has_left.sum()) / rows, float(has_right.sum()) / rows
    ok_left, ok_right = rows_left >= th.lane_side_min_rows, rows_right >= th.lane_side_min_rows
    status: LaneStatus = (
        "valid" if (ok_left and ok_right) else ("uncertain" if (ok_left or ok_right) else "unavailable")
    )
    if status != "valid":
        return status, None, None, rows_left, rows_right, []
    both = np.nonzero(has_left & has_right)[0]
    if both.size == 0:
        return "uncertain", None, None, rows_left, rows_right, []
    # Innermost lane pixel on each side of the image centre, per row.
    left_inner = (cx - 1) - np.argmax(left[both, ::-1], axis=1)
    right_inner = cx + np.argmax(right[both], axis=1)
    mids = (left_inner + right_inner) / 2.0
    halves = (right_inner - left_inner) / 2.0
    centre, half = float(np.median(mids)), float(np.median(halves))
    offset = None if half < 1 else float(np.clip((width / 2.0 - centre) / half, -1.0, 1.0))
    step = max(1, both.size // 12)
    points = [[round(float(mids[i]), 1), float(y0 + both[i])] for i in range(0, both.size, step)]
    return status, offset, half, rows_left, rows_right, points


def drivable_offset(drivable_mask: BoolMask, th: DecisionThresholds = DEFAULT_THRESHOLDS) -> float | None:
    height, width = drivable_mask.shape
    y0, y1 = _band_rows(height, th)
    band = drivable_mask[y0:y1]
    counts = band.sum(axis=1)
    rows = np.nonzero(counts >= th.drivable_row_min_fraction * width)[0]
    if rows.size == 0:
        return None
    sub = band[rows]
    first = np.argmax(sub, axis=1)
    last = width - 1 - np.argmax(sub[:, ::-1], axis=1)
    mids, halves = (first + last) / 2.0, (last - first) / 2.0
    half = float(np.median(halves))
    if half < 1:
        return None
    return float(np.clip((width / 2.0 - float(np.median(mids))) / half, -1.0, 1.0))


def _has_evidence(det: DecisionDetection, min_score: float, tracked_min_score: float) -> bool:
    """Untracked boxes need the family minimum; tracked continuation boxes still need `tracked_min_score`."""
    if det.score >= min_score:
        return True
    return det.track_id is not None and det.score >= tracked_min_score


def _family_min_score(family: ClassFamily, th: DecisionThresholds) -> float:
    return {
        "vehicle": th.vehicle_min_score,
        "person": th.person_min_score,
        "two_wheeler": th.two_wheeler_min_score,
        "traffic_light": th.light_min_score,
        "stop_sign": th.stop_sign_min_score,
    }[family]


def _is_family_evidence(det: DecisionDetection, family: ClassFamily, th: DecisionThresholds) -> bool:
    return class_family(det.class_name) == family and _has_evidence(
        det, _family_min_score(family, th), th.tracked_min_score
    )


def _is_vehicle_evidence(det: DecisionDetection, th: DecisionThresholds) -> bool:
    return _is_family_evidence(det, "vehicle", th)


def _corridor_cue(
    detections: Sequence[DecisionDetection],
    accept: Callable[[DecisionDetection], bool],
    width: int,
    height: int,
    th: DecisionThresholds,
) -> tuple[float | None, int | None]:
    best: tuple[float, int] | None = None
    for i, det in enumerate(detections):
        if not accept(det):
            continue
        x1, y1, x2, y2 = det.xyxy_pixels
        extent = _corridor_x_extent(min(y2, height - 1.0), width, height, th)
        if extent is None:
            continue
        cl, cr = extent
        overlap = max(0.0, min(x2, cr) - max(x1, cl))
        denom = max(1.0, min(x2 - x1, cr - cl))
        if overlap / denom < th.corridor_min_overlap:
            continue
        area = (x2 - x1) * (y2 - y1)
        if best is None or area > best[0]:
            best = (area, i)
    if best is None:
        return None, None
    _, idx = best
    x1, y1, x2, y2 = detections[idx].xyxy_pixels
    y_term = float(np.clip((y2 / height - th.corridor_top) / (1.0 - th.corridor_top), 0.0, 1.0))
    h_term = float(np.clip(((y2 - y1) / height) / th.cue_height_full, 0.0, 1.0))
    return round(0.5 * y_term + 0.5 * h_term, 4), idx


def obstacle_cue(
    detections: Sequence[DecisionDetection], width: int, height: int, th: DecisionThresholds = DEFAULT_THRESHOLDS
) -> tuple[float | None, int | None]:
    """(cue, detection index) for the largest vehicle-family box in the ego corridor; (None, None) when none."""
    return _corridor_cue(detections, lambda d: _is_vehicle_evidence(d, th), width, height, th)


def family_cue(
    detections: Sequence[DecisionDetection],
    family: ClassFamily,
    width: int,
    height: int,
    th: DecisionThresholds = DEFAULT_THRESHOLDS,
) -> tuple[float | None, int | None]:
    """Same corridor cue as `obstacle_cue` for any decision family (person, two_wheeler, ...)."""
    return _corridor_cue(detections, lambda d: _is_family_evidence(d, family, th), width, height, th)


def light_is_relevant(
    xyxy: Sequence[float], width: int, height: int, th: DecisionThresholds = DEFAULT_THRESHOLDS
) -> bool:
    """Upper central region test for a traffic-light box (where lights governing the ego lane usually appear)."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return th.light_region_x_min * width <= cx <= th.light_region_x_max * width and cy < th.light_region_y_max * height


def analyze_lights(
    image: RgbImage,
    detections: Sequence[DecisionDetection],
    th: DecisionThresholds = DEFAULT_THRESHOLDS,
    light_th: LightThresholds = DEFAULT_LIGHT_THRESHOLDS,
) -> tuple[list[LightStateResult | None], list[int], LightState | None]:
    """(per-detection HSV results, relevant light indices, aggregated relevant state)."""
    height, width = image.shape[:2]
    results: list[LightStateResult | None] = [None] * len(detections)
    relevant: list[int] = []
    for i, det in enumerate(detections):
        if not _is_family_evidence(det, "traffic_light", th):
            continue
        x1, y1, x2, y2 = det.xyxy_pixels
        if (y2 - y1) < th.light_min_height * height:
            continue
        results[i] = classify_light(image, det.xyxy_pixels, light_th)
        if light_is_relevant(det.xyxy_pixels, width, height, th):
            relevant.append(i)
    if not relevant:
        return results, relevant, None
    states = {results[i].state for i in relevant if results[i] is not None}  # type: ignore[union-attr]
    for state in ("red", "yellow", "green"):
        if state in states:
            return results, relevant, state  # type: ignore[return-value]
    return results, relevant, "unknown"


def decide(
    image: RgbImage,
    drivable_mask: BoolMask | None,
    lane_mask: BoolMask | None,
    detections: Sequence[DecisionDetection],
    th: DecisionThresholds = DEFAULT_THRESHOLDS,
    light_th: LightThresholds = DEFAULT_LIGHT_THRESHOLDS,
) -> VideoDecision:
    height, width = image.shape[:2]
    for name, mask in (("drivable", drivable_mask), ("lane", lane_mask)):
        if mask is not None and mask.shape != (height, width):
            raise ValueError(f"{name} mask shape {mask.shape} does not match image {(height, width)}")
    quality = frame_quality(image, th)
    uncertainty: list[str] = []

    lane_status, offset, _half, rows_left, rows_right, lane_points = lane_analysis(lane_mask, th)
    if lane_mask is None:
        uncertainty.append("lane_mask_missing")
    elif lane_status != "valid":
        uncertainty.append(f"lane_{lane_status}")
    corridor_fraction: float | None = None
    if drivable_mask is None:
        uncertainty.append("drivable_mask_missing")
    else:
        corridor = _corridor_mask(width, height, th)
        corridor_fraction = float(drivable_mask[corridor].mean()) if corridor.any() else None
        if offset is None:
            offset = drivable_offset(drivable_mask, th)
            if offset is not None:
                uncertainty.append("lane_offset_from_drivable_area")
    if offset is None:
        uncertainty.append("lane_offset_unavailable")

    cue, obstacle_idx = obstacle_cue(detections, width, height, th)
    person_cue, person_idx = family_cue(detections, "person", width, height, th)
    tw_cue, tw_idx = family_cue(detections, "two_wheeler", width, height, th)
    light_results, relevant_lights, light_state = analyze_lights(image, detections, th, light_th)
    lights_visible = any(r is not None for r in light_results)
    stop_signs = [
        i
        for i, d in enumerate(detections)
        if _is_family_evidence(d, "stop_sign", th)
        and (d.xyxy_pixels[3] - d.xyxy_pixels[1]) >= th.stop_sign_min_height * height
    ]
    if any(class_family(d.class_name) is None for d in detections):
        uncertainty.append("unverified_class_excluded")
    if lights_visible:
        uncertainty.append("light_state_hsv_heuristic")

    drivable_low = corridor_fraction is not None and corridor_fraction < th.drivable_corridor_low
    if drivable_low:
        uncertainty.append("drivable_area_low")
    if quality.dark:
        uncertainty.append("frame_dark")
    if quality.blurry:
        uncertainty.append("frame_blurry")

    corridor_code = (
        ReasonCode.NO_VEHICLE_IN_CORRIDOR
        if cue is None
        else (ReasonCode.VEHICLE_IN_CORRIDOR_NEAR if cue >= th.cue_near else ReasonCode.VEHICLE_IN_CORRIDOR)
    )
    lane_code = {
        "valid": ReasonCode.LANE_VALID,
        "uncertain": ReasonCode.LANE_UNCERTAIN,
        "unavailable": ReasonCode.LANE_UNAVAILABLE,
    }[lane_status]
    person_near = person_cue is not None and person_cue >= th.vru_cue_near
    tw_near = tw_cue is not None and tw_cue >= th.vru_cue_near

    flags = [
        code
        for code, on in (
            (ReasonCode.DRIVABLE_AREA_LOW, drivable_low),
            (ReasonCode.FRAME_DARK, quality.dark),
            (ReasonCode.FRAME_BLURRY, quality.blurry),
        )
        if on
    ]
    stop_codes = [
        code
        for code, on in (
            (ReasonCode.VEHICLE_IN_CORRIDOR_NEAR, corridor_code == ReasonCode.VEHICLE_IN_CORRIDOR_NEAR),
            (ReasonCode.PERSON_IN_CORRIDOR_NEAR, person_near),
            (ReasonCode.TWO_WHEELER_IN_CORRIDOR, tw_near),
            (ReasonCode.TRAFFIC_LIGHT_RED_HEURISTIC, light_state == "red"),
        )
        if on
    ]
    slow_codes = [
        code
        for code, on in (
            (ReasonCode.VEHICLE_IN_CORRIDOR, cue is not None and th.cue_slow <= cue < th.cue_near),
            (ReasonCode.PERSON_IN_CORRIDOR, person_cue is not None and not person_near),
            (ReasonCode.TWO_WHEELER_IN_CORRIDOR, tw_cue is not None and not tw_near),
            (ReasonCode.TRAFFIC_LIGHT_YELLOW_HEURISTIC, light_state == "yellow"),
            (ReasonCode.STOP_SIGN_VISIBLE, bool(stop_signs)),
            (lane_code, lane_status != "valid"),
        )
        if on
    ] + flags
    info_codes = [corridor_code, lane_code]
    if lights_visible:
        info_codes.append(ReasonCode.TRAFFIC_LIGHT_VISIBLE)
    if light_state == "green":
        info_codes.append(ReasonCode.TRAFFIC_LIGHT_GREEN_HEURISTIC)

    state: DecisionStateName
    if stop_codes:
        state, driving = "controlled_stop", stop_codes
    elif slow_codes:
        state, driving = "slow", slow_codes
    else:
        state, driving = "observe", []
    # State-driving codes first, then informational corridor/lane/light codes and the remaining flags.
    codes: list[ReasonCode] = []
    for code in [*driving, *info_codes, *slow_codes]:
        if code not in codes:
            codes.append(code)

    cues = [(c, i) for c, i in ((cue, obstacle_idx), (person_cue, person_idx), (tw_cue, tw_idx)) if c is not None]
    top_cue, top_idx = max(cues, key=lambda ci: ci[0]) if cues else (None, None)

    return VideoDecision(
        state=state,
        reason_codes=[str(c) for c in codes],
        lane_status=lane_status,
        image_lane_offset_normalized=None if offset is None else round(offset, 4),
        image_obstacle_cue=top_cue,
        quality=quality,
        uncertainty_reasons=uncertainty,
        lane_points=lane_points,
        obstacle_detection_index=top_idx,
        corridor_drivable_fraction=corridor_fraction,
        lane_rows_left=rows_left,
        lane_rows_right=rows_right,
        vehicle_cue=cue,
        person_cue=person_cue,
        person_detection_index=person_idx,
        two_wheeler_cue=tw_cue,
        two_wheeler_detection_index=tw_idx,
        light_states=light_results,
        relevant_light_indices=relevant_lights,
        light_state=light_state,
        stop_sign_indices=stop_signs,
    )


def inference_failed_decision(image: RgbImage, th: DecisionThresholds = DEFAULT_THRESHOLDS) -> VideoDecision:
    """Decision for a sampled frame whose inference failed: nothing is inferred, the frame is only observed."""
    quality = frame_quality(image, th)
    uncertainty = ["inference_failed"]
    if quality.dark:
        uncertainty.append("frame_dark")
    if quality.blurry:
        uncertainty.append("frame_blurry")
    return VideoDecision(
        state="observe",
        reason_codes=[str(ReasonCode.INFERENCE_FAILED)],
        lane_status="unavailable",
        image_lane_offset_normalized=None,
        image_obstacle_cue=None,
        quality=quality,
        uncertainty_reasons=uncertainty,
    )
