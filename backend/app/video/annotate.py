"""Overlay rendering for analyzed frames (spec section 5, step 4; class families from spec section 2.1).

Overlays: drivable-area mask green (alpha 0.35), lane mask cyan, ego-corridor outline, detector boxes and a HUD naming
the perception source (e.g. "YOLOPv2 + RF-DETR-N | learned_model | CPU"), the original PTS and the analytical
decision. All text drawn into frames is ASCII (OpenCV Hershey fonts cannot render other glyphs).

Boxes are drawn when score >= 0.5, or when they carry a confirmed track id and score >= the decision
thresholds' tracked_min_score (0.4): the same evidence rule the decisions use, so low-score ByteTrack continuation
boxes (kept for association only) never appear as findings. Colour per class family (RGB):
vehicle/bus/truck orange (yellow when untracked), person magenta, bicycle/motorcycle azure, traffic light in the
colour of its HSV-heuristic state (red / amber / green / light grey for unknown), stop sign crimson, classes outside
the enabled set grey. Labels read "class #track score" (e.g. "person #4 0.83"); traffic lights append the heuristic
state as "red(hsv)" so it is never mistaken for model output. Boxes that drive a slow/stop code are drawn thicker; the
largest corridor obstacle is outlined in red. The HUD panel is darkened only behind its own text (top-left); boxes
that intersect it are drawn again on top, so a traffic light near the top edge is never hidden.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from app.perception.video_decisions import (
    DEFAULT_THRESHOLDS,
    DecisionThresholds,
    VideoDecision,
    class_family,
    corridor_polygon,
)
from app.perception.world_state import TrackedDetection

RgbImage = NDArray[np.uint8]
BoolMask = NDArray[np.bool_]
Rgb = tuple[int, int, int]

DRIVABLE_RGB = (0, 200, 0)
DRIVABLE_ALPHA = 0.35
LANE_RGB = (0, 255, 255)
LANE_ALPHA = 0.75
BOX_TRACKED_RGB = (255, 170, 0)  # vehicle family, tracked
BOX_UNTRACKED_RGB = (255, 235, 59)  # vehicle family, untracked
BOX_UNVERIFIED_RGB = (170, 170, 170)
BOX_OBSTACLE_RGB = (255, 48, 48)
FAMILY_RGB: dict[str, Rgb] = {
    "person": (255, 0, 255),
    "two_wheeler": (0, 140, 255),
    "stop_sign": (200, 0, 90),
}
LIGHT_RGB: dict[str, Rgb] = {
    "red": (255, 40, 40),
    "yellow": (255, 200, 0),
    "green": (0, 220, 110),
    "unknown": (215, 215, 215),
}
STATE_RGB = {"observe": (76, 217, 100), "slow": (255, 193, 7), "controlled_stop": (255, 59, 48)}
BOX_DRAW_MIN_SCORE = 0.5


@dataclass(frozen=True)
class HudInfo:
    model_label: str  # e.g. "YOLOPv2" or "YOLOPv2 + RF-DETR-N"
    source: str  # perception source tag, e.g. "learned_model"
    device: str  # "CPU" / "CUDA"


def ascii_text(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def _blend_mask(image: RgbImage, mask: BoolMask, rgb: tuple[int, int, int], alpha: float) -> None:
    if not mask.any():
        return
    color = np.asarray(rgb, dtype=np.float32)
    pixels = image[mask].astype(np.float32)
    image[mask] = np.clip(pixels * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)


def _text_color(background: Rgb) -> Rgb:
    r, g, b = background
    return (0, 0, 0) if (0.299 * r + 0.587 * g + 0.114 * b) >= 128 else (255, 255, 255)


def _put_text(
    img: RgbImage,
    text: str,
    org: tuple[int, int],
    scale: float,
    rgb: tuple[int, int, int],
    thick: int,
    background: tuple[int, int, int] | None = None,
) -> None:
    """Single-pass text (a thicker outline pass would misalign: Hershey glyph advance depends on thickness)."""
    text = ascii_text(text)
    if background is not None:
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x, y = org
        cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + base), background, -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, rgb, thick, cv2.LINE_AA)


def detection_style(det: TrackedDetection, light_state: str | None) -> tuple[Rgb, str]:
    """(box colour, label) for one detection; label format "class #track score" plus "state(hsv)" for lights."""
    family = class_family(det.class_name)
    if family == "vehicle":
        color = BOX_TRACKED_RGB if det.track_id is not None else BOX_UNTRACKED_RGB
    elif family == "traffic_light":
        color = LIGHT_RGB.get(light_state or "unknown", LIGHT_RGB["unknown"])
    elif family is not None:
        color = FAMILY_RGB[family]
    else:
        color = BOX_UNVERIFIED_RGB
    label = det.class_name
    if det.track_id is not None:
        label += f" #{det.track_id}"
    label += f" {det.score:.2f}"
    if family == "traffic_light" and light_state is not None:
        label += f" {light_state}(hsv)"
    return color, label


def annotate_frame(
    image: RgbImage,
    *,
    drivable_mask: BoolMask | None,
    lane_mask: BoolMask | None,
    detections: Sequence[TrackedDetection],
    decision: VideoDecision,
    hud: HudInfo,
    pts_s: float,
    sample_index: int,
    thresholds: DecisionThresholds = DEFAULT_THRESHOLDS,
) -> RgbImage:
    out = np.array(image, dtype=np.uint8, copy=True)
    height, width = out.shape[:2]
    scale = max(0.4, min(1.2, width / 1600.0))
    thick = max(1, round(scale * 2))

    if drivable_mask is not None:
        _blend_mask(out, drivable_mask, DRIVABLE_RGB, DRIVABLE_ALPHA)
    if lane_mask is not None:
        _blend_mask(out, lane_mask, LANE_RGB, LANE_ALPHA)

    corridor = corridor_polygon(width, height, thresholds)
    cv2.polylines(out, [corridor], isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    emphasised = decision.highlight_indices()
    drawn: list[tuple[int, int, int]] = []  # (index, x1, y1)

    def draw_box(i: int, det: TrackedDetection) -> None:
        light_res = decision.light_states[i] if i < len(decision.light_states) else None
        color, label = detection_style(det, None if light_res is None else light_res.state)
        t = thick + 1 if i in emphasised else thick
        x1, y1, x2, y2 = (round(v) for v in det.xyxy_pixels)
        if i == decision.obstacle_detection_index:
            cv2.rectangle(out, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), BOX_OBSTACLE_RGB, thick + 1, cv2.LINE_AA)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, t, cv2.LINE_AA)
        label_y = max(int(22 * scale) + 2, y1 - 4)
        _put_text(out, label, (x1 + 2, label_y), scale * 0.7, _text_color(color), max(1, thick - 1), background=color)

    for i, det in enumerate(detections):
        tracked_ok = det.track_id is not None and det.score >= thresholds.tracked_min_score
        if det.score < BOX_DRAW_MIN_SCORE and not tracked_ok:
            continue
        draw_box(i, det)
        drawn.append((i, round(det.xyxy_pixels[0]), round(det.xyxy_pixels[1])))

    # HUD panel: darkened only behind its own text (top-left) so traffic lights near the top stay visible.
    line_h = int(30 * scale) + 6
    band_h = line_h * 4 + 8
    small = max(1, thick - 1)
    cue = "-" if decision.image_obstacle_cue is None else f"{decision.image_obstacle_cue:.2f}"
    offset = "-" if decision.image_lane_offset_normalized is None else f"{decision.image_lane_offset_normalized:+.2f}"
    light = "" if decision.light_state is None else f"  light {decision.light_state}(hsv)"
    state_color = STATE_RGB.get(decision.state, (255, 255, 255))
    lines = [
        (f"{hud.model_label} | {hud.source} | {hud.device}", scale, (255, 255, 255), thick),
        (
            f"t={pts_s:.3f}s  sample {sample_index}  lane {decision.lane_status}  offset {offset}  cue {cue}{light}",
            scale * 0.8,
            (230, 230, 230),
            small,
        ),
        (f"{decision.state.upper()}: {', '.join(decision.reason_codes[:3])}", scale * 0.8, state_color, thick),
        ("analytical only - no vehicle control", scale * 0.7, (200, 200, 200), small),
    ]
    text_w = max(cv2.getTextSize(ascii_text(text), cv2.FONT_HERSHEY_SIMPLEX, s, t)[0][0] for text, s, _, t in lines)
    panel_w = min(width, text_w + 20)
    out[:band_h, :panel_w] = (out[:band_h, :panel_w].astype(np.float32) * 0.40).astype(np.uint8)
    for k, (text, s, color, t) in enumerate(lines, start=1):
        _put_text(out, text, (10, line_h * k), s, color, t)
    # Boxes intersecting the panel are drawn again on top so the panel never hides a detection.
    label_h = int(30 * scale)  # the label sits above the box
    for i, x1, y1 in drawn:
        if y1 - label_h < band_h and x1 < panel_w:
            draw_box(i, detections[i])
    return out
