"""LiteRoad graph, world raster and renderer: topology, projection conventions, occlusion, determinism, speed."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from app.simulators.lite.graph import build_town_graph, junction_nodes, node_degrees, reachable_from
from app.simulators.lite.renderer import LiteRenderer, default_calibration
from app.simulators.lite.world import Box3D, get_world


@pytest.fixture(scope="module")
def renderer() -> LiteRenderer:
    return LiteRenderer(get_world(), default_calibration())


def test_graph_topology() -> None:
    g = build_town_graph()
    ids = [n.id for n in g.nodes]
    assert 9 <= len(ids) <= 12 and len(set(ids)) == len(ids)
    deg = node_degrees(g)
    assert deg["Q"] == 0, "closed depot must be isolated"
    assert "Q" not in reachable_from(g, "A")
    assert reachable_from(g, "A") == frozenset(i for i in ids if i != "Q")
    assert len(junction_nodes(g)) >= 3
    curved = [e for e in g.edges if len(e.polyline) > 4]
    assert len(curved) >= 3, "several curved edges are required"
    for n in g.nodes:
        assert n.label_uz and n.label_en
    for e in g.edges:
        assert e.polyline[0] == (g.node(e.a).x, g.node(e.a).y)
        assert e.polyline[-1] == (g.node(e.b).x, g.node(e.b).y)
        assert e.length_m > 10


def test_calibration_matches_spec() -> None:
    c = default_calibration()
    assert (c.width, c.height) == (640, 360)
    assert 2 * math.degrees(math.atan(c.cx / c.fx)) == pytest.approx(90.0)
    assert c.height_m == 1.4 and c.forward_offset_m == 1.0
    assert math.degrees(c.pitch_down_rad) == pytest.approx(8.0)


def test_render_shape_sky_and_determinism(renderer: LiteRenderer) -> None:
    img = renderer.render(10.0, -1.75, 0.0, list(get_world().static_boxes))
    assert img.shape == (360, 640, 3) and img.dtype == np.uint8
    assert renderer.horizon_row == 136
    sky = img[: renderer.horizon_row - 2].reshape(-1, 3).mean(axis=0)
    assert sky[2] > sky[0], "sky is blue above the horizon"
    img2 = renderer.render(10.0, -1.75, 0.0, list(get_world().static_boxes))
    assert np.array_equal(img, img2)
    moved = renderer.render(12.0, -1.75, 0.0, list(get_world().static_boxes))
    assert not np.array_equal(img, moved), "a new pose must produce a new observation"


def _orange_columns(img: np.ndarray) -> np.ndarray:
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    mask = (r > 180) & (g > 60) & (g < 140) & (b < 60)
    return np.nonzero(mask.any(axis=0))[0]


def test_obstacle_projects_straight_ahead_and_yaw_sign(renderer: LiteRenderer) -> None:
    box = Box3D(30.0, 0.0, 0.0, 1.0, 1.0, 1.2, "obstacle")
    img = renderer.render(10.0, 0.0, 0.0, [box])
    cols = _orange_columns(img)
    assert len(cols) > 5
    assert abs(cols.mean() - 320) < 3, "an obstacle on the optical axis appears at the image centre column"
    # yawing the ego to the left (CCW, +yaw) moves world objects to the right in the image
    img_left = renderer.render(10.0, 0.0, 0.1, [box])
    assert _orange_columns(img_left).mean() > 320 + 20
    # an obstacle north of the ego (its left, +y) appears left of centre
    img_side = renderer.render(10.0, 0.0, 0.0, [Box3D(30.0, 4.0, 0.0, 1.0, 1.0, 1.2, "obstacle")])
    assert _orange_columns(img_side).mean() < 320 - 20


def test_occlusion_order_near_box_hides_far_box(renderer: LiteRenderer) -> None:
    near = Box3D(20.0, 0.0, 0.0, 1.0, 1.0, 1.2, "obstacle")
    far_grey = Box3D(30.0, 0.0, 0.0, 1.0, 1.0, 1.2, "building", color=(20, 20, 200))
    a = renderer.render(10.0, 0.0, 0.0, [far_grey, near])
    b = renderer.render(10.0, 0.0, 0.0, [near, far_grey])
    assert np.array_equal(a, b), "draw order is decided by distance, not list order"
    blue = (a[..., 2] > 150) & (a[..., 0] < 60)
    assert not blue[:, 312:328].any(), "the far box is hidden behind the nearer one"


def test_render_time_budget(renderer: LiteRenderer) -> None:
    boxes = [*get_world().static_boxes, Box3D(25.0, -1.75, 0.0, 1.0, 1.0, 1.2, "obstacle")]
    for _ in range(5):
        renderer.render(0.0, -1.75, 0.0, boxes)
    times = []
    for i in range(40):
        t0 = time.perf_counter()
        renderer.render(i * 0.5, -1.75, 0.0, boxes)
        times.append(time.perf_counter() - t0)
    assert float(np.median(times)) * 1000 <= 15.0
