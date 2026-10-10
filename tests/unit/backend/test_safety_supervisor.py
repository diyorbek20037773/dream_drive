"""Safety supervisor checks: frames, perception outputs, ego bounds."""

from __future__ import annotations

import time

import numpy as np
import pytest

from app.agents.safety import SafetyLimits, SafetySupervisor
from app.navigation.route import RoadSurface
from app.simulators.base import EgoState, RgbFrame
from app.simulators.lite.graph import JUNCTION_HALF_SIZE_M, build_town_graph, junction_nodes

DT = 0.05


@pytest.fixture(scope="module")
def sup() -> SafetySupervisor:
    g = build_town_graph()
    return SafetySupervisor(SafetyLimits(), RoadSurface(g, JUNCTION_HALF_SIZE_M, junction_nodes(g)))


def frame(fid: int, t: float, acquired: float | None = None) -> RgbFrame:
    return RgbFrame(fid, t, np.zeros((2, 2, 3), np.uint8), time.monotonic() if acquired is None else acquired)


def test_frame_checks(sup: SafetySupervisor) -> None:
    now = time.monotonic()
    assert sup.check_frame(frame(5, 0.25), 4, 0.25, DT, now).ok
    missing = sup.check_frame(None, 4, 0.25, DT, now)
    assert not missing.ok and missing.quality_status == "missing" and missing.codes == ("CAMERA_INVALID",)
    same_id = sup.check_frame(frame(4, 0.25), 4, 0.25, DT, now)
    assert not same_id.ok and same_id.codes == ("FRAME_STALE",)
    old_sim = sup.check_frame(frame(5, 0.10), 4, 0.25, DT, now)
    assert not old_sim.ok and old_sim.codes == ("FRAME_STALE",)
    old_wall = sup.check_frame(frame(5, 0.25, acquired=now - 2.0), 4, 0.25, DT, now)
    assert not old_wall.ok and old_wall.codes == ("FRAME_STALE",)


def test_perception_checks(sup: SafetySupervisor) -> None:
    assert sup.check_perception("valid", (), "valid", 0.1, 0.2, 0.01) == ()
    assert sup.check_perception("degraded", (), "unavailable", None, 0.0, 0.01) == ()
    assert sup.check_perception("missing", ("CAMERA_BLACK",), "unavailable", None, None, 0.01) == ("CAMERA_BLACK",)
    assert sup.check_perception("invalid", (), "unavailable", None, None, 0.01) == ("CAMERA_INVALID",)
    assert sup.check_perception("valid", (), "valid", 0.0, 0.0, 6.0) == ("PERCEPTION_TIMEOUT",)
    for cue in (float("nan"), 1.5, -0.1):
        assert "CAMERA_INVALID" in sup.check_perception("valid", (), "valid", 0.0, cue, 0.01)
    assert "CAMERA_INVALID" in sup.check_perception("valid", (), "valid", float("inf"), 0.0, 0.01)
    assert "CAMERA_INVALID" in sup.check_perception("valid", (), "bogus", 0.0, 0.0, 0.01)


def test_ego_bounds(sup: SafetySupervisor) -> None:
    assert sup.check_ego(EgoState(30.0, -1.75, 0.0, 3.0)) is None
    assert sup.check_ego(EgoState(30.0, -9.0, 0.0, 3.0)) == "EGO_OUT_OF_BOUNDS"  # 9 m from the road centreline
    assert sup.check_ego(EgoState(float("nan"), 0.0, 0.0, 1.0)) == "EGO_OUT_OF_BOUNDS"
    assert sup.check_ego(EgoState(30.0, -1.75, 0.0, 9.0)) == "EGO_OUT_OF_BOUNDS"
    assert sup.check_ego(EgoState(1000.0, 0.0, 0.0, 1.0)) == "EGO_OUT_OF_BOUNDS"
