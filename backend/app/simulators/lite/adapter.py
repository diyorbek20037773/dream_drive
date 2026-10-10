"""LiteRoadAdapter: deterministic fixed-step CPU simulator implementing app.simulators.base.SimulatorAdapter.

Runtime view (``get_rgb_frame``, ``get_ego_telemetry``, ``get_route_graph``, ``get_camera_calibration``) returns
rendered pixels, ego localization telemetry and static topology only. Scene objects stay private; evaluation truth is
reachable only through ``evaluator()``. Camera faults are part of the camera model: the scenario schedule decides
whether a frame is rendered normally, returned black, or frozen (same frame id, timestamp and pixels).
"""

from __future__ import annotations

import time

import numpy as np

from app.simulators.base import CameraCalibration, ControlCommand, EgoState, RgbFrame, RoadGraph
from app.simulators.lite.dynamics import VehicleState, step_bicycle
from app.simulators.lite.evaluator import LiteRoadEvaluator, Obstacle
from app.simulators.lite.renderer import LiteRenderer, default_calibration
from app.simulators.lite.scenarios import SCENARIOS
from app.simulators.lite.world import Box3D, get_world

CAMERA_FAULTS = ("black", "freeze", "occlude", "uniform")


class LiteRoadAdapter:
    """One simulated world. Not thread-safe; one instance per simulation session."""

    def __init__(self) -> None:
        self._world = get_world()
        self._calib = default_calibration()
        self._renderer = LiteRenderer(self._world, self._calib)
        self._state = VehicleState(0.0, 0.0, 0.0, 0.0)
        self._obstacles: list[Obstacle] = []
        self._next_obstacle_id = 1
        self._tick = 0
        self._sim_time = 0.0
        self._command = ControlCommand(0.0, 0.0, 1.0, 0)
        self._frame: RgbFrame | None = None
        self._frozen: RgbFrame | None = None
        self._fault: str | None = None
        self._fault_start_s: float | None = None
        self._scenario = "normal"
        self._seed = 0
        self._evaluator = LiteRoadEvaluator(self)
        self._closed = False

    # ------------------------------------------------------------------------------------------- scene setup
    def reset(self, seed: int, scenario: str) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}")
        spec = SCENARIOS[scenario]
        self._scenario, self._seed = scenario, seed
        self._state = VehicleState(0.0, 0.0, 0.0, 0.0)
        self._obstacles = []
        self._next_obstacle_id = 1
        self._tick = 0
        self._sim_time = 0.0
        self._command = ControlCommand(0.0, 0.0, 1.0, 0)
        self._frame = None
        self._frozen = None
        self._fault = spec.camera_fault
        self._fault_start_s = spec.fault_start_s
        self._evaluator.reset()

    def spawn_ego(self, x: float, y: float, yaw: float, speed: float = 0.0) -> None:
        self._state = VehicleState(x, y, yaw, speed)
        self._frame = None

    def add_obstacle(self, x: float, y: float, yaw: float, visible: bool = True, source: str = "scenario") -> int:
        ob = Obstacle(self._next_obstacle_id, x, y, yaw, visible=visible, source=source)
        self._next_obstacle_id += 1
        self._obstacles.append(ob)
        self._frame = None
        return ob.id

    def set_camera_fault(self, kind: str | None, start_s: float = 0.0) -> None:
        """Camera-model fault from sim time ``start_s``: black | freeze | occlude (lower 60 % covered) | uniform."""
        if kind is not None and kind not in CAMERA_FAULTS:
            raise ValueError(f"unknown camera fault {kind!r}")
        self._fault, self._fault_start_s = kind, start_s
        self._frame = None
        self._frozen = None

    # ------------------------------------------------------------------------------------------- runtime view
    def _fault_active(self) -> bool:
        return self._fault is not None and self._fault_start_s is not None and self._sim_time >= self._fault_start_s

    def _render(self) -> np.ndarray:
        boxes = list(self._world.static_boxes)
        boxes += [
            Box3D(o.x, o.y, o.yaw, o.size_m, o.size_m, o.height_m, "obstacle") for o in self._obstacles if o.visible
        ]
        return self._renderer.render(self._state.x, self._state.y, self._state.yaw, boxes)

    def get_rgb_frame(self) -> RgbFrame:
        if self._closed:
            raise RuntimeError("adapter closed")
        if self._frame is not None and self._frame.frame_id == self._tick:
            return self._frame
        active = self._fault_active()
        if active and self._fault == "freeze" and self._frozen is not None:
            return self._frozen  # same id, timestamp, pixels: the camera stopped delivering new frames
        if active and self._fault == "black":
            image = np.zeros((self._calib.height, self._calib.width, 3), dtype=np.uint8)
        elif active and self._fault == "uniform":
            image = np.full((self._calib.height, self._calib.width, 3), 128, dtype=np.uint8)
        else:
            image = self._render()
            if active and self._fault == "occlude":
                cut = int(self._calib.height * 0.4)
                image[cut:] = (52, 50, 48)
        image.setflags(write=False)
        frame = RgbFrame(
            frame_id=self._tick,
            sim_time_s=round(self._sim_time, 6),
            image=image,
            acquired_wall_monotonic=time.monotonic(),
        )
        self._frame = frame
        if self._fault == "freeze" and (not active or self._frozen is None):
            self._frozen = frame  # last frame delivered before the freeze starts
        return frame

    def get_ego_telemetry(self) -> EgoState:
        s = self._state
        return EgoState(x_m=s.x, y_m=s.y, yaw_rad=s.yaw, speed_mps=s.speed)

    def get_route_graph(self) -> RoadGraph:
        return self._world.graph

    def get_camera_calibration(self) -> CameraCalibration:
        return self._calib

    # ------------------------------------------------------------------------------------------- actuation
    def apply_control(self, command: ControlCommand) -> None:
        self._command = command

    def step(self, dt: float) -> None:
        if self._closed:
            raise RuntimeError("adapter closed")
        c = self._command
        self._state = step_bicycle(self._state, c.steer, c.throttle, c.brake, dt)
        self._tick += 1
        self._sim_time = self._tick * dt
        self._evaluator.after_step(self._state, self._obstacles)

    @property
    def sim_time_s(self) -> float:
        return self._sim_time

    @property
    def tick(self) -> int:
        return self._tick

    def close(self) -> None:
        self._closed = True

    def evaluator(self) -> LiteRoadEvaluator:
        return self._evaluator
