"""Analytical decisions for the RF-DETR-N classes (spec section 2.1): persons, two-wheelers, traffic lights (HSV
heuristic colour), stop signs, plus precedence against the existing vehicle/lane logic. Image-space only."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.perception.video_decisions import (
    CLASS_FAMILIES,
    DEFAULT_THRESHOLDS,
    class_family,
    decide,
    light_is_relevant,
)
from app.perception.world_state import TrackedDetection
from app.schemas.reason_codes import ReasonCode

W, H = 640, 360
IDS = {"vehicle": 3, "person": 101, "bicycle": 102, "motorcycle": 104, "bus": 106, "truck": 108}
IDS |= {"traffic_light": 110, "stop_sign": 113}
LIT = {"red": (255, 40, 40), "yellow": (255, 190, 0), "green": (40, 230, 130)}


def _image(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(70, 190, size=(H, W, 3), dtype=np.uint8)


def _lanes() -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    m[200:, 200:204] = True
    m[200:, 440:444] = True
    return m


def _drivable() -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    m[200:, 100:540] = True
    return m


def _det(cls: str, box: list[float], score: float = 0.9, track: int | None = 1) -> TrackedDetection:
    return TrackedDetection(IDS.get(cls, 999), cls, score, box, track, model_id="rfdetr_nano")


def _draw_light(img: np.ndarray, box: list[float], state: str | None) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), (25, 25, 25), -1)
    r = int((x2 - x1) * 0.3)
    for k, name in enumerate(("red", "yellow", "green")):
        centre = ((x1 + x2) // 2, y1 + int((y2 - y1) * (k + 0.5) / 3))
        cv2.circle(img, centre, r, LIT[name] if name == state else (55, 55, 55), -1)


def _with_light(state: str | None, box: list[float] | None = None) -> tuple[np.ndarray, TrackedDetection]:
    box = box or [310.0, 40.0, 334.0, 106.0]  # upper centre, above the corridor top (0.55 H = 198 px)
    img = _image()
    _draw_light(img, box, state)
    return img, _det("traffic_light", box, 0.8)


def _decide(img: np.ndarray, dets: list[TrackedDetection]):  # noqa: ANN202 - test helper
    return decide(img, _drivable(), _lanes(), dets)


def test_class_families_use_only_enabled_names() -> None:
    assert set(CLASS_FAMILIES) == set(IDS)
    assert class_family("car") is None  # COCO car is not enabled (YOLOPv2 vehicle covers cars)
    assert class_family("unverified_class_5") is None
    assert {class_family(c) for c in ("vehicle", "bus", "truck")} == {"vehicle"}


def test_near_person_in_corridor_requests_controlled_stop() -> None:
    near = _det("person", [300, 160, 350, 345])  # feet near the image bottom, tall box
    d = _decide(_image(), [near])
    assert d.state == "controlled_stop"
    assert d.reason_codes[0] == ReasonCode.PERSON_IN_CORRIDOR_NEAR
    assert d.person_cue is not None and d.person_cue >= DEFAULT_THRESHOLDS.vru_cue_near
    assert d.image_obstacle_cue == d.person_cue and d.obstacle_detection_index == 0
    assert ReasonCode.NO_VEHICLE_IN_CORRIDOR in d.reason_codes  # vehicle logic unchanged and informational


def test_far_person_in_corridor_slows() -> None:
    far = _det("person", [315, 185, 327, 215])
    d = _decide(_image(), [far])
    assert d.state == "slow" and d.reason_codes[0] == ReasonCode.PERSON_IN_CORRIDOR
    assert d.person_cue is not None and d.person_cue < DEFAULT_THRESHOLDS.vru_cue_near


def test_person_outside_corridor_and_low_score_untracked_person_are_ignored() -> None:
    side = _det("person", [20, 150, 60, 300])
    weak = _det("person", [300, 160, 350, 345], score=0.3, track=None)
    for det in (side, weak):
        d = _decide(_image(), [det])
        assert d.state == "observe" and d.person_cue is None
        assert not any(c.startswith("PERSON") for c in d.reason_codes)


def test_two_wheeler_near_stops_far_slows() -> None:
    near = _det("motorcycle", [270, 200, 370, 350])
    far = _det("bicycle", [312, 190, 330, 222])
    d_near, d_far = _decide(_image(), [near]), _decide(_image(), [far])
    assert d_near.state == "controlled_stop" and d_near.reason_codes[0] == ReasonCode.TWO_WHEELER_IN_CORRIDOR
    assert d_far.state == "slow" and d_far.reason_codes[0] == ReasonCode.TWO_WHEELER_IN_CORRIDOR


def test_bus_and_truck_count_as_vehicles() -> None:
    for cls in ("bus", "truck"):
        d = _decide(_image(), [_det(cls, [220, 200, 420, 350])])
        assert d.state == "controlled_stop" and d.reason_codes[0] == ReasonCode.VEHICLE_IN_CORRIDOR_NEAR


@pytest.mark.parametrize(
    ("state", "decision", "first_code"),
    [
        ("red", "controlled_stop", ReasonCode.TRAFFIC_LIGHT_RED_HEURISTIC),
        ("yellow", "slow", ReasonCode.TRAFFIC_LIGHT_YELLOW_HEURISTIC),
        ("green", "observe", ReasonCode.NO_VEHICLE_IN_CORRIDOR),
        (None, "observe", ReasonCode.NO_VEHICLE_IN_CORRIDOR),
    ],
)
def test_relevant_traffic_light_colour_heuristic(state: str | None, decision: str, first_code: str) -> None:
    img, light = _with_light(state)
    d = _decide(img, [light])
    assert d.state == decision and d.reason_codes[0] == first_code
    assert ReasonCode.TRAFFIC_LIGHT_VISIBLE in d.reason_codes
    assert d.relevant_light_indices == [0]
    expected = state or "unknown"
    assert d.light_state == expected
    assert d.light_states[0] is not None and d.light_states[0].method == "hsv_heuristic"
    assert "light_state_hsv_heuristic" in d.uncertainty_reasons
    assert (ReasonCode.TRAFFIC_LIGHT_GREEN_HEURISTIC in d.reason_codes) == (state == "green")


def test_red_light_outside_the_upper_central_region_does_not_stop() -> None:
    side_box = [600.0, 40.0, 624.0, 106.0]  # right edge of the frame (cross-street signal)
    assert not light_is_relevant(side_box, W, H)
    img, light = _with_light("red", side_box)
    d = _decide(img, [light])
    assert d.state == "observe"
    assert d.light_states[0] is not None and d.light_states[0].state == "red"  # classified, but not relevant
    assert d.light_state is None
    assert ReasonCode.TRAFFIC_LIGHT_RED_HEURISTIC not in d.reason_codes
    assert ReasonCode.TRAFFIC_LIGHT_VISIBLE in d.reason_codes


def test_low_light_box_below_region_is_not_relevant() -> None:
    assert not light_is_relevant([300, 250, 320, 300], W, H)
    assert light_is_relevant([300, 40, 320, 100], W, H)


def test_conflicting_relevant_lights_resolve_conservatively_to_red() -> None:
    img = _image()
    red_box, green_box = [250.0, 40.0, 274.0, 106.0], [370.0, 40.0, 394.0, 106.0]
    _draw_light(img, red_box, "red")
    _draw_light(img, green_box, "green")
    d = _decide(img, [_det("traffic_light", green_box, 0.8, 1), _det("traffic_light", red_box, 0.8, 2)])
    assert d.light_state == "red" and d.state == "controlled_stop"
    assert d.highlight_indices() == {1}


def test_tiny_or_weak_lights_are_not_evaluated() -> None:
    img, _ = _with_light("red")
    tiny = _det("traffic_light", [318.0, 50.0, 321.0, 53.0], 0.9)  # 3 px tall < 1.2 % of H
    weak = _det("traffic_light", [310.0, 40.0, 334.0, 106.0], 0.3, None)
    d = _decide(img, [tiny, weak])
    assert d.light_states == [None, None] and d.light_state is None
    assert ReasonCode.TRAFFIC_LIGHT_VISIBLE not in d.reason_codes


def test_stop_sign_slows() -> None:
    d = _decide(_image(), [_det("stop_sign", [500, 120, 530, 150], 0.7)])
    assert d.state == "slow" and d.reason_codes[0] == ReasonCode.STOP_SIGN_VISIBLE
    assert d.stop_sign_indices == [0]
    small = _decide(_image(), [_det("stop_sign", [500, 120, 503, 123], 0.7)])
    assert small.state == "observe"


def test_unknown_classes_are_ignored_for_decisions() -> None:
    other = TrackedDetection(102, "car", 0.95, [220, 200, 420, 350], 3)  # COCO car is not an enabled class
    d = _decide(_image(), [other])
    assert d.state == "observe" and d.image_obstacle_cue is None
    assert "unverified_class_excluded" in d.uncertainty_reasons


def test_precedence_stop_codes_first_then_slow_then_informational() -> None:
    img, light = _with_light("red")
    mid_vehicle = _det("vehicle", [270, 210, 370, 290], track=5)  # cue ~0.51 -> slow-level
    far_person = _det("person", [315, 185, 327, 215], track=6)
    stop = _det("stop_sign", [560, 120, 590, 150], 0.8, 7)
    d = _decide(img, [light, mid_vehicle, far_person, stop])
    assert d.state == "controlled_stop"
    assert d.reason_codes[0] == ReasonCode.TRAFFIC_LIGHT_RED_HEURISTIC
    rest = d.reason_codes[1:]
    for code in (
        ReasonCode.VEHICLE_IN_CORRIDOR,
        ReasonCode.PERSON_IN_CORRIDOR,
        ReasonCode.STOP_SIGN_VISIBLE,
        ReasonCode.TRAFFIC_LIGHT_VISIBLE,
        ReasonCode.LANE_VALID,
    ):
        assert code in rest
    assert len(d.reason_codes) == len(set(d.reason_codes))
    assert set(d.reason_codes) <= {c.value for c in ReasonCode}
    # Without the red light the same scene is "slow" with the vehicle code first.
    d2 = _decide(img, [mid_vehicle, far_person, stop])
    assert d2.state == "slow" and d2.reason_codes[:3] == [
        ReasonCode.VEHICLE_IN_CORRIDOR,
        ReasonCode.PERSON_IN_CORRIDOR,
        ReasonCode.STOP_SIGN_VISIBLE,
    ]


def test_vehicle_near_and_person_near_both_listed_vehicle_first() -> None:
    d = _decide(_image(), [_det("vehicle", [220, 200, 420, 350]), _det("person", [430, 170, 470, 345])])
    assert d.state == "controlled_stop"
    assert d.reason_codes[:2] == [ReasonCode.VEHICLE_IN_CORRIDOR_NEAR, ReasonCode.PERSON_IN_CORRIDOR_NEAR]


def test_measurements_include_family_cues() -> None:
    d = _decide(_image(), [_det("person", [315, 185, 327, 215])])
    m = d.measurements()
    assert m["person_cue"] == d.person_cue and m["vehicle_cue"] is None and m["two_wheeler_cue"] is None
