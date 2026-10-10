"""SyntheticRgbPerception: validity, lane geometry signs, obstacle distance/corridor, determinism, isolation."""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

import app.perception.synthetic_rgb_cv as cv_mod
from app.perception.synthetic_rgb_cv import SyntheticRgbPerception
from app.schemas.world_state import Perception
from app.simulators.lite.renderer import LiteRenderer, default_calibration
from app.simulators.lite.world import Box3D, get_world


@pytest.fixture(scope="module")
def env() -> tuple[LiteRenderer, SyntheticRgbPerception]:
    cal = default_calibration()
    return LiteRenderer(get_world(), cal), SyntheticRgbPerception(cal)


def render(env, x: float, y: float, yaw: float, extra: list[Box3D] | None = None) -> np.ndarray:  # type: ignore[no-untyped-def]
    return env[0].render(x, y, yaw, [*get_world().static_boxes, *(extra or [])])


def test_module_never_touches_simulator_objects() -> None:
    tree = ast.parse(Path(cv_mod.__file__).read_text(encoding="utf-8"))
    imported = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imported += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    allowed_sim = {"app.simulators.base"}  # CameraCalibration dataclass only
    assert not [m for m in imported if m and m.startswith("app.simulators") and m not in allowed_sim]
    assert list(inspect.signature(SyntheticRgbPerception.process).parameters) == ["self", "image"]


@pytest.mark.parametrize(
    ("image", "quality", "code"),
    [
        (np.zeros((360, 640, 3), np.uint8), "missing", "CAMERA_BLACK"),
        (np.full((360, 640, 3), 128, np.uint8), "invalid", "CAMERA_INVALID"),
        (np.zeros((100, 100, 3), np.uint8), "invalid", "CAMERA_INVALID"),
        (np.zeros((360, 640, 3), np.float32), "invalid", "CAMERA_INVALID"),
        (np.zeros((360, 640), np.uint8), "invalid", "CAMERA_INVALID"),
    ],
)
def test_invalid_frames(env, image: np.ndarray, quality: str, code: str) -> None:  # type: ignore[no-untyped-def]
    res = env[1].process(image)
    assert res.quality_status == quality and code in res.quality_codes
    assert res.lane_status == "unavailable" and res.image_obstacle_cue is None
    assert not res.usable


def test_lane_valid_on_straight_and_offset_signs(env) -> None:  # type: ignore[no-untyped-def]
    centred = env[1].process(render(env, 10.0, -1.75, 0.0))
    assert centred.quality_status == "valid" and centred.lane_status == "valid"
    assert abs(centred.lane_centre_left_m) < 0.15
    assert abs(centred.image_lane_offset_normalized) < 0.03
    # ego 0.5 m left of the lane centre (y = -1.25): the lane centre is 0.5 m to its right (negative)
    left = env[1].process(render(env, 10.0, -1.25, 0.0))
    assert left.lane_centre_left_m == pytest.approx(-0.5, abs=0.15)
    assert left.image_lane_offset_normalized > 0.03, "lane centre appears right of the image centre"
    right = env[1].process(render(env, 10.0, -2.25, 0.0))
    assert right.lane_centre_left_m == pytest.approx(0.5, abs=0.15)
    assert right.image_lane_offset_normalized < -0.03
    # heading: ego yawed +0.1 rad (left) -> lane runs to the right relative to the ego (negative angle)
    yawed = env[1].process(render(env, 10.0, -1.75, 0.1))
    assert yawed.lane_heading_rad == pytest.approx(-0.1, abs=0.03)
    assert len(centred.lane_points_px) >= 4


def test_curve_is_tracked(env) -> None:  # type: ignore[no-untyped-def]
    # on the C->G left arc (radius 40 m centreline, lane radius 41.75 m), tangent heading 0.5 rad
    x = 145 + 41.75 * math.sin(0.5)
    y = 40 - 41.75 * math.cos(0.5)
    res = env[1].process(render(env, x, y, 0.5))
    assert res.lane_status == "valid"
    assert abs(res.lane_centre_left_m) < 0.25
    assert res.lane_centre.coeffs[2] > 0.005, "left curve -> positive curvature term"


def test_junction_box_without_markings_is_unavailable(env) -> None:  # type: ignore[no-untyped-def]
    res = env[1].process(render(env, 48.0, -1.75, 0.0))
    assert res.lane_status == "unavailable" and "no_lane_markings" in res.uncertainty_reasons


@pytest.mark.parametrize("ahead", [10.0, 15.0, 20.0])
def test_obstacle_distance_estimate_and_cue(env, ahead: float) -> None:  # type: ignore[no-untyped-def]
    box = Box3D(20.0 + ahead + 0.5, -1.75, 0.0, 1.0, 1.0, 1.2, "obstacle")
    res = env[1].process(render(env, 20.0, -1.75, 0.0, [box]))
    assert len(res.obstacles) == 1
    ob = res.obstacles[0]
    assert ob.in_corridor
    assert ob.distance_m == pytest.approx(ahead, abs=0.8), "flat-ground IPM distance of the contact line"
    expected_cue = min(1.0, max(0.0, (24.0 - ob.distance_m) / 16.0))
    assert res.image_obstacle_cue == pytest.approx(expected_cue, abs=1e-3)
    assert "obstacle_distance_flat_ground_synthetic_estimate" in res.uncertainty_reasons
    assert res.lane_status == "valid", "obstacle stripes must not be mistaken for lane paint"


def test_obstacle_in_other_lane_is_outside_corridor(env) -> None:  # type: ignore[no-untyped-def]
    box = Box3D(30.5, 1.75, 0.0, 1.0, 1.0, 1.2, "obstacle")
    res = env[1].process(render(env, 20.0, -1.75, 0.0, [box]))
    assert len(res.obstacles) == 1 and not res.obstacles[0].in_corridor
    assert res.image_obstacle_cue == 0.0


def test_partial_occlusion_degrades_and_removes_lane(env) -> None:  # type: ignore[no-untyped-def]
    img = render(env, 10.0, -1.75, 0.0).copy()
    img[144:] = (52, 50, 48)
    res = env[1].process(img)
    assert res.quality_status == "degraded" and res.lane_status == "unavailable"
    assert "camera_view_partially_occluded" in res.uncertainty_reasons


def test_deterministic_and_world_state_shape(env) -> None:  # type: ignore[no-untyped-def]
    img = render(env, 20.0, -1.75, 0.0, [Box3D(35.5, -1.75, 0.0, 1.0, 1.0, 1.2, "obstacle")])
    a, b = env[1].process(img), env[1].process(img.copy())
    assert a.lane_centre == b.lane_centre and a.obstacles == b.obstacles
    assert a.image_obstacle_cue == b.image_obstacle_cue
    p = a.to_perception()
    assert isinstance(p, Perception)
    assert p.source == "synthetic_rgb_cv" and p.model_id is None and p.model_sha256 is None
    assert p.detections[0].class_name == "synthetic_obstacle"
    assert p.ttc_seconds is None
    overlay = a.overlay()
    assert overlay["frame_width"] == 640 and overlay["frame_height"] == 360
    assert overlay["obstacle_boxes"] and overlay["lane_points"]
