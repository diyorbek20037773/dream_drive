"""Classical HSV colour-state HEURISTIC for detected traffic-light crops (docs/IMPLEMENTATION_SPEC.md section 2.1).

This is NOT model output. RF-DETR-N (COCO) detects a `traffic_light` box but has no colour classes; this module looks
at the pixels inside that box and reports `red | yellow | green | unknown` with method `hsv_heuristic`. Every consumer
(timeline `light_state_method`, reason codes `TRAFFIC_LIGHT_*_HEURISTIC`, HUD "(hsv)", reports) keeps that label.
It has no accuracy measurement (no labelled light-state data); treat it as an indicator only.

Algorithm (OpenCV HSV: H 0..179, S 0..255, V 0..255), thresholds in `LightThresholds`:
1. Crop the detector box shrunk by `inset` (10 %) per side to drop pole/background pixels. A crop smaller than
   `min_crop_w` x `min_crop_h` (3 x 6 px) after the inset -> `unknown` (reason `crop_too_small`).
2. A pixel is "lit" when V >= 140 and S >= 80 (bright and saturated), or V >= 200 and S >= 40 ("bright halo" tier:
   real lamps are often exposed to a white core with a less saturated coloured halo). An unlit lamp, the housing, grey
   sky and the back of a signal head are rejected by one or both tests. Calibration (2026-10-10) on 5 hand-boxed crops
   of the CC BY 3.0 Memphis street fixture (docs/DATA_SOURCES.md), states read visually: two green heads have white
   cores (S < 12) and halo pixels at H 87-89, S 47-55, V >= 249 - green fraction 0.000 with the first tier alone,
   0.033 / 0.047 with the halo tier; two small red heads (S 88-130, V 205-255) give red 0.087 / 0.091; a head seen from
   the side and three overcast-sky patches give 0.000 for every colour. Five crops are a sanity check, not accuracy.
3. Lit pixels are binned by hue: red H <= 9 or H >= 160, yellow/amber 12..35, green 45..100 (camera-rendered signal
   greens range from green to cyan-green). Hues outside these bins (e.g. blue sky 100+) are ignored.
4. fraction(colour) = lit pixels of that colour / crop pixels. The state is the colour with the largest fraction when
   it is >= `min_fraction` (2 % of the crop) and >= `dominance` (2x) the runner-up; otherwise `unknown`
   (`no_lit_lamp` or `ambiguous_colour`).
5. Lamp-extent check: a lit lamp covers one lamp position, a painted housing (yellow signal heads are common in the
   US) or glare covers the whole head. When the winning colour's lit pixels span more than `max_lamp_span` (60 %) of
   the crop's long axis the state is `unknown` (`colour_spans_head`). Measured in the two-model end-to-end run on
   the Memphis fixture (artifacts/qa/spike/video_e2e2): 42 classified real lamps (22 green, 18 red, 1 yellow, plus
   1 dash-cam green) spanned 0.17-0.46; one yellow-painted housing that passed steps 2-4 spanned 0.79.
Known failure modes: lamps saturated to white by exposure, LED flicker captured between pulses, coloured
backgrounds (red brick, green foliage) inside the box, arrows/pedestrian heads, very small (distant) lights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

LightState = Literal["red", "yellow", "green", "unknown"]
LIGHT_STATE_METHOD = "hsv_heuristic"
LIGHT_STATES: tuple[LightState, ...] = ("red", "yellow", "green", "unknown")


@dataclass(frozen=True)
class LightThresholds:
    inset: float = 0.10
    min_crop_w: int = 3
    min_crop_h: int = 6
    lit_v_min: int = 140
    lit_s_min: int = 80
    halo_v_min: int = 200
    halo_s_min: int = 40
    red_h_max: int = 9
    red_h_min_wrap: int = 160
    yellow_h: tuple[int, int] = (12, 35)
    green_h: tuple[int, int] = (45, 100)
    min_fraction: float = 0.02
    dominance: float = 2.0
    max_lamp_span: float = 0.60

    def as_dict(self) -> dict[str, object]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


DEFAULT_LIGHT_THRESHOLDS = LightThresholds()


@dataclass(frozen=True)
class LightStateResult:
    state: LightState
    method: str
    fractions: dict[str, float]
    reason: str  # "classified" | "crop_too_small" | "no_lit_lamp" | "ambiguous_colour" | "colour_spans_head"

    def as_dict(self) -> dict[str, object]:
        return {
            "light_state": self.state,
            "light_state_method": self.method,
            "hsv_fractions": {k: round(v, 4) for k, v in self.fractions.items()},
            "light_state_reason": self.reason,
        }


def _crop(image: NDArray[np.uint8], xyxy: list[float] | tuple[float, ...], th: LightThresholds) -> NDArray[np.uint8]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    dx, dy = (x2 - x1) * th.inset, (y2 - y1) * th.inset
    ix1 = int(np.clip(np.floor(x1 + dx), 0, width))
    iy1 = int(np.clip(np.floor(y1 + dy), 0, height))
    ix2 = int(np.clip(np.ceil(x2 - dx), 0, width))
    iy2 = int(np.clip(np.ceil(y2 - dy), 0, height))
    return image[iy1:iy2, ix1:ix2]


def classify_light(
    image: NDArray[np.uint8], xyxy: list[float] | tuple[float, ...], th: LightThresholds = DEFAULT_LIGHT_THRESHOLDS
) -> LightStateResult:
    """Colour state of the traffic light inside `xyxy` (original-image pixels) of an RGB uint8 frame."""
    crop = _crop(image, xyxy, th)
    zero = {"red": 0.0, "yellow": 0.0, "green": 0.0}
    if crop.ndim != 3 or crop.shape[0] < th.min_crop_h or crop.shape[1] < th.min_crop_w:
        return LightStateResult("unknown", LIGHT_STATE_METHOD, zero, "crop_too_small")
    hsv = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    lit = ((val >= th.lit_v_min) & (sat >= th.lit_s_min)) | ((val >= th.halo_v_min) & (sat >= th.halo_s_min))
    total = float(hue.size)
    masks = {
        "red": lit & ((hue <= th.red_h_max) | (hue >= th.red_h_min_wrap)),
        "yellow": lit & (hue >= th.yellow_h[0]) & (hue <= th.yellow_h[1]),
        "green": lit & (hue >= th.green_h[0]) & (hue <= th.green_h[1]),
    }
    fractions = {name: float(mask.sum()) / total for name, mask in masks.items()}
    ranked = sorted(fractions.items(), key=lambda kv: kv[1], reverse=True)
    (best, best_f), (_, second_f) = ranked[0], ranked[1]
    if best_f < th.min_fraction:
        return LightStateResult("unknown", LIGHT_STATE_METHOD, fractions, "no_lit_lamp")
    if best_f < th.dominance * second_f:
        return LightStateResult("unknown", LIGHT_STATE_METHOD, fractions, "ambiguous_colour")
    winner = masks[best]
    axis_hits = winner.any(axis=1) if winner.shape[0] >= winner.shape[1] else winner.any(axis=0)
    idx = np.nonzero(axis_hits)[0]
    if (idx[-1] - idx[0] + 1) / axis_hits.size > th.max_lamp_span:
        return LightStateResult("unknown", LIGHT_STATE_METHOD, fractions, "colour_spans_head")
    return LightStateResult(best, LIGHT_STATE_METHOD, fractions, "classified")  # type: ignore[arg-type]
