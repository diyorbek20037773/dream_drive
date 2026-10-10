"""HSV traffic-light colour heuristic (app.perception.traffic_light) on synthetic signal-head crops.

These are synthetic renders (dark housing, three lamps, one lit) plus JPEG round trips; they prove the documented
thresholds behave as described, not that the heuristic is accurate on real footage (no labelled light-state data).
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from app.perception.traffic_light import (
    DEFAULT_LIGHT_THRESHOLDS,
    LIGHT_STATE_METHOD,
    LIGHT_STATES,
    classify_light,
)

LIT = {"red": (255, 40, 40), "yellow": (255, 190, 0), "green": (40, 230, 130)}
UNLIT = (55, 55, 55)
HOUSING = (25, 25, 25)


def signal_head(
    lit: str | None,
    *,
    w: int = 24,
    h: int = 66,
    horizontal: bool = False,
    lit_rgb: tuple[int, int, int] | None = None,
    background: tuple[int, int, int] = (150, 170, 200),
    pad: int = 10,
) -> tuple[np.ndarray, list[float]]:
    """RGB image with one signal head (top red, middle yellow, bottom green) and its tight box."""
    if horizontal:
        w, h = h, w
    img = np.full((h + 2 * pad, w + 2 * pad, 3), background, dtype=np.uint8)
    x1, y1 = pad, pad
    cv2.rectangle(img, (x1, y1), (x1 + w - 1, y1 + h - 1), HOUSING, -1)
    r = int(min(w, h) * 0.3)
    for k, name in enumerate(("red", "yellow", "green")):
        if horizontal:
            centre = (x1 + int(w * (k + 0.5) / 3), y1 + h // 2)
        else:
            centre = (x1 + w // 2, y1 + int(h * (k + 0.5) / 3))
        color = (lit_rgb or LIT[name]) if name == lit else UNLIT
        cv2.circle(img, centre, r, color, -1, cv2.LINE_AA)
    return img, [float(x1), float(y1), float(x1 + w), float(y1 + h)]


def jpeg_roundtrip(img: np.ndarray, quality: int = 90) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


@pytest.mark.parametrize("state", ["red", "yellow", "green"])
def test_lit_lamp_colour_is_classified(state: str) -> None:
    img, box = signal_head(state)
    res = classify_light(img, box)
    assert res.state == state and res.reason == "classified"
    assert res.method == LIGHT_STATE_METHOD == "hsv_heuristic"
    assert res.fractions[state] >= DEFAULT_LIGHT_THRESHOLDS.min_fraction


@pytest.mark.parametrize("state", ["red", "yellow", "green"])
def test_survives_jpeg_and_horizontal_heads(state: str) -> None:
    img, box = signal_head(state, horizontal=True)
    assert classify_light(jpeg_roundtrip(img, 85), box).state == state


def test_led_variants_amber_and_cyan_green() -> None:
    img, box = signal_head("yellow", lit_rgb=(255, 150, 20))  # amber LED (hue ~14 in OpenCV units)
    assert classify_light(img, box).state == "yellow"
    img, box = signal_head("green", lit_rgb=(20, 220, 190))  # cyan-green LED (hue ~86)
    assert classify_light(img, box).state == "green"


def test_unlit_or_back_of_head_is_unknown() -> None:
    img, box = signal_head(None)
    res = classify_light(img, box)
    assert res.state == "unknown" and res.reason == "no_lit_lamp"
    grey = np.full((80, 40, 3), 128, dtype=np.uint8)  # back plate / grey sky: unsaturated
    assert classify_light(grey, [5.0, 5.0, 35.0, 75.0]).state == "unknown"


def test_two_lit_colours_of_similar_size_are_ambiguous() -> None:
    img, box = signal_head("red")
    # Light the green lamp as well (red + green at once is not a valid single state).
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.circle(img, ((x1 + x2) // 2, y1 + int((y2 - y1) * 2.5 / 3)), int((x2 - x1) * 0.3), LIT["green"], -1)
    res = classify_light(img, box)
    assert res.state == "unknown" and res.reason == "ambiguous_colour"


def test_painted_yellow_housing_spanning_the_head_is_unknown() -> None:
    # A yellow-painted signal head in sunlight is saturated yellow over its whole height (real case: Memphis 11.8 s).
    img = np.full((60, 40, 3), (60, 90, 60), dtype=np.uint8)
    cv2.rectangle(img, (14, 10), (25, 49), (235, 190, 60), -1)
    res = classify_light(img, [14.0, 10.0, 26.0, 50.0])
    assert res.state == "unknown" and res.reason == "colour_spans_head"
    assert res.fractions["yellow"] > DEFAULT_LIGHT_THRESHOLDS.min_fraction  # the colour alone would say yellow
    lenient = replace(DEFAULT_LIGHT_THRESHOLDS, max_lamp_span=1.0)
    assert classify_light(img, [14.0, 10.0, 26.0, 50.0], lenient).state == "yellow"


def test_tiny_or_degenerate_crops_are_unknown() -> None:
    img, _ = signal_head("red")
    assert classify_light(img, [10.0, 10.0, 12.0, 14.0]).reason == "crop_too_small"
    assert classify_light(img, [50.0, 50.0, 50.0, 50.0]).reason == "crop_too_small"
    assert classify_light(img, [-100.0, -100.0, -50.0, -50.0]).reason == "crop_too_small"  # outside the image


def test_dim_red_below_value_threshold_is_unknown() -> None:
    img, box = signal_head("red", lit_rgb=(110, 20, 20))  # V = 110 < 140
    assert classify_light(img, box).state == "unknown"


def test_washed_out_lamp_with_coloured_halo_uses_the_halo_tier() -> None:
    # Measured on the real Memphis crops: white core (S < 12) inside a cyan-green halo (H 87-89, S 47-55, V >= 249).
    halo, box = signal_head("green", lit_rgb=(55, 55, 55))  # start from an unlit head
    x1, y1, x2, y2 = (int(v) for v in box)
    centre = ((x1 + x2) // 2, y1 + int((y2 - y1) * 2.5 / 3))
    cv2.circle(halo, centre, int((x2 - x1) * 0.3), (205, 255, 245), -1)  # V 255, S ~ 50: halo tier only
    cv2.circle(halo, centre, int((x2 - x1) * 0.15), (250, 255, 254), -1)  # white core, S ~ 5: rejected
    assert classify_light(halo, box).state == "green"
    no_halo = replace(DEFAULT_LIGHT_THRESHOLDS, halo_s_min=255)
    assert classify_light(halo, box, no_halo).state == "unknown"


def test_thresholds_are_configurable_and_recorded() -> None:
    img, box = signal_head("red", lit_rgb=(110, 20, 20))
    lenient = replace(DEFAULT_LIGHT_THRESHOLDS, lit_v_min=100)
    assert classify_light(img, box, lenient).state == "red"
    d = DEFAULT_LIGHT_THRESHOLDS.as_dict()
    assert d["lit_v_min"] == 140 and d["lit_s_min"] == 80 and d["dominance"] == 2.0
    assert set(LIGHT_STATES) == {"red", "yellow", "green", "unknown"}


def test_result_dict_labels_the_heuristic() -> None:
    img, box = signal_head("green")
    d = classify_light(img, box).as_dict()
    assert d["light_state"] == "green" and d["light_state_method"] == "hsv_heuristic"
    assert set(d["hsv_fractions"]) == {"red", "yellow", "green"}  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------------------------
# Real crops: hand-placed boxes on the CC BY 3.0 Memphis street fixture (docs/DATA_SOURCES.md). The expected states
# were read visually from the frames (green heads at 3.0 s, small red heads and a side-on head at 10.6 s, a painted
# yellow housing at 11.8 s whose colour alone passes the thresholds - rejected by the lamp-extent rule). Six crops
# are a sanity check of the documented thresholds on real footage, not an accuracy measurement.
# ---------------------------------------------------------------------------------------------------------------
REAL_CROPS = [
    (3.0, (262, 140, 283, 180), "green"),
    (3.0, (378, 140, 400, 180), "green"),
    (10.6, (124, 224, 136, 244), "red"),
    (10.6, (195, 227, 208, 249), "red"),
    (10.6, (38, 155, 58, 190), "unknown"),  # signal head seen from the side (no lamp facing the camera)
    (11.8, (405.59, 225.27, 415.03, 241.54), "unknown"),  # yellow-painted housing (RF-DETR box from the e2e run)
]


def _street_frames(times: list[float]) -> dict[float, np.ndarray]:
    import json
    import os
    from pathlib import Path

    import av

    root = Path(__file__).resolve().parents[3]
    manifest = json.loads((root / "tests" / "fixtures" / "manifest.json").read_text(encoding="utf-8"))
    entry = next(e for e in manifest["fixtures"] if e["id"] == "commons_memphis_front_st_ccby3")
    path = root / "tests" / "fixtures" / "cache" / entry["derived"]["file_name"]
    if not path.exists():
        if os.environ.get("DD_REQUIRE_LIVE") == "1":
            pytest.fail(f"fixture {path.name} missing: run scripts/fetch_fixtures.py (required when DD_REQUIRE_LIVE=1)")
        pytest.skip("street fixture not downloaded (python scripts/fetch_fixtures.py)")
    wanted, frames = sorted(set(times)), {}
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            while wanted and frame.time is not None and frame.time >= wanted[0] - 1e-6:
                frames[wanted.pop(0)] = frame.to_ndarray(format="rgb24")
            if not wanted:
                break
    return frames


def test_real_street_crops_match_visual_reading() -> None:
    frames = _street_frames([t for t, _, _ in REAL_CROPS])
    results = [(classify_light(frames[t], [float(v) for v in box]).state, expected) for t, box, expected in REAL_CROPS]
    assert [got for got, _ in results] == [expected for _, expected in results]
    # The green heads need the halo tier (their cores are exposed to white).
    t, box, _ = REAL_CROPS[0]
    no_halo = replace(DEFAULT_LIGHT_THRESHOLDS, halo_s_min=255)
    assert classify_light(frames[t], [float(v) for v in box], no_halo).state == "unknown"
