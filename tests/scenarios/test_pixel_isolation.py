"""Pixel ablation, hidden-truth isolation and deterministic repeat (docs/QA.md "Pixel separation" gate).

- Pixel ablation: identical ego/route, only the RGB differs -> perception, decision and command react.
- Hidden-truth isolation: two worlds that differ only in an evaluator-only obstacle (never rendered, so the RGB is
  identical) produce identical runtime commands; the evaluator still sees the difference.
- Determinism: the same seed reproduces the exact command sequence.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.agents.runner import SessionConfig, SimulationSession
from app.agents.runtime_view import RuntimeView
from app.control.controller import SLOW_SPEED_MPS
from app.simulators.lite.adapter import LiteRoadAdapter
from app.simulators.lite.world import Box3D

pytestmark = pytest.mark.timeout(600)


def session(adapter: LiteRoadAdapter | None = None, scenario: str = "normal", seed: int = 7) -> SimulationSession:
    return SimulationSession(
        SessionConfig(
            run_id="iso", start_node="A", destination_node="C", scenario=scenario, seed=seed, encode_jpeg=False
        ),
        adapter=adapter,
    )


def cmd_tuple(r) -> tuple:  # type: ignore[no-untyped-def]
    c = r.live["command"]
    return (c["steer"], c["throttle"], c["brake"], c["input_frame_id"], tuple(c["reason_codes"]), c["safety_override"])


class LaterallyShiftedCamera(LiteRoadAdapter):
    """Camera rendered 0.6 m to the left of the true pose; ego telemetry (localization) is unchanged."""

    shift_m = 0.6

    def _render(self) -> np.ndarray:
        st = self._state
        x = st.x - self.shift_m * math.sin(st.yaw)
        y = st.y + self.shift_m * math.cos(st.yaw)
        boxes = list(self._world.static_boxes)
        boxes += [
            Box3D(o.x, o.y, o.yaw, o.size_m, o.size_m, o.height_m, "obstacle") for o in self._obstacles if o.visible
        ]
        return self._renderer.render(x, y, st.yaw, boxes)


def test_runtime_view_exposes_no_truth() -> None:
    view = RuntimeView(LiteRoadAdapter())
    for name in ("evaluator", "step", "apply_control", "_obstacles", "_state", "add_obstacle", "reset"):
        assert not hasattr(view, name)
    assert callable(view.get_rgb_frame) and callable(view.get_ego_telemetry)


@pytest.mark.parametrize("fault", ["occlude", "black"])
def test_pixel_ablation_changes_perception_and_control(fault: str) -> None:
    a, b = session(), session()
    a.start()
    b.start()
    a.tick()
    b.tick()  # first tick consumes the identical initial frame in both runs
    b._adapter.set_camera_fault(fault, 0.0)  # noqa: SLF001 - camera-model fault, ego/route untouched
    ra, rb = a.tick(), b.tick()
    assert ra.live["ego"] == rb.live["ego"], "same ego state; only the pixels differ"
    assert ra.frame_id == rb.frame_id
    assert ra.live["perception"]["lane_status"] == "valid"
    assert rb.live["perception"]["lane_status"] == "unavailable"
    assert cmd_tuple(ra) != cmd_tuple(rb)
    if fault == "occlude":
        assert rb.live["perception"]["quality_status"] == "degraded"
        assert rb.live["decision"]["state"] == "slow" and "LANE_UNCERTAIN_SLOW" in rb.live["decision"]["reason_codes"]
        assert ra.live["decision"]["state"] == "follow_route"
        # downstream: the occluded run slows to the lane-uncertain speed while the clear run cruises
        for _ in range(60):
            ra, rb = a.tick(), b.tick()
        assert ra.live["ego"]["speed_mps"] > rb.live["ego"]["speed_mps"] + 1.0
        assert rb.live["ego"]["speed_mps"] <= SLOW_SPEED_MPS + 0.3  # measured PID settling overshoot ~0.17 m/s
    else:
        assert rb.live["perception"]["quality_status"] == "missing"
        assert rb.live["decision"]["state"] == "safety_stop"
        assert rb.live["command"]["brake"] == 1.0 and rb.live["command"]["throttle"] == 0.0


def test_pixel_lane_offset_changes_steering() -> None:
    a, b = session(), session(adapter=LaterallyShiftedCamera())
    ra, rb = a.start(), b.start()
    assert ra.live["ego"] == rb.live["ego"]
    pa = ra.live["perception"]["image_lane_offset_normalized"]
    pb = rb.live["perception"]["image_lane_offset_normalized"]
    assert pb > pa + 0.03, "camera left of the lane centre -> lane centre appears further right"
    ra, rb = a.tick(), b.tick()
    assert ra.live["ego"] == rb.live["ego"] and ra.frame_id == rb.frame_id == 0
    assert rb.live["command"]["steer"] < ra.live["command"]["steer"] - 0.05, "pixel lane offset steers right"


def test_hidden_evaluator_obstacle_does_not_change_commands() -> None:
    plain, hidden_adapter = LiteRoadAdapter(), LiteRoadAdapter()
    a, b = session(plain), session(hidden_adapter)
    a.start()
    b.start()
    # evaluator-only obstacle in the ego lane 25 m ahead; never rendered, so the RGB stays identical
    hidden_adapter.add_obstacle(25.0, -1.75, 0.0, visible=False, source="hidden_test")
    assert len(hidden_adapter.evaluator().obstacle_positions()) == 1
    assert plain.evaluator().obstacle_positions() == []
    n = 0
    while True:
        ra, rb = a.tick(), b.tick()
        n += 1
        assert np.array_equal(a.last_frame_image, b.last_frame_image), f"RGB differs at tick {n}"
        if rb.terminal or ra.terminal:
            break
        assert cmd_tuple(ra) == cmd_tuple(rb), f"runtime command differs at tick {n}"
        assert ra.live["decision"] == rb.live["decision"]
        assert ra.live["perception"] == rb.live["perception"]
    # the evaluator (and only the evaluator) observed the hidden obstacle: the runtime drove into it
    assert rb.outcome == "failed_collision" and not ra.terminal
    assert rb.live["decision"]["reason_codes"] == ["COLLISION_DETECTED_BY_EVALUATOR"]
    assert cmd_tuple(ra) == cmd_tuple(rb), "the colliding tick's command was computed identically"
    assert n > 50


def test_deterministic_repeat_with_fixed_seed() -> None:
    runs = []
    for _ in range(2):
        s = session(scenario="obstacle_stop", seed=3)
        r = s.start()
        seq = []
        while not r.terminal:
            r = s.tick()
            seq.append(cmd_tuple(r))
        runs.append((seq, r.outcome, s.summary()["evaluator"]["final_obstacle_gap_m"]))
    assert runs[0] == runs[1]
    other = session(scenario="obstacle_stop", seed=4)
    r = other.start()
    seq4 = []
    while not r.terminal:
        r = other.tick()
        seq4.append(cmd_tuple(r))
    assert seq4 != runs[0][0], "a different seed changes the scene and therefore the commands"
