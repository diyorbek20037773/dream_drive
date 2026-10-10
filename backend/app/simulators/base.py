"""Simulator interfaces with an explicit runtime / evaluator separation (docs/SIMULATION.md).

Runtime decision code (perception, FSM, controller, safety) only ever receives a `RuntimeSimulatorView`.
Evaluation truth (scene objects, collisions, lateral deviation) is reachable only through `EvaluatorTruth`,
which the SimulationSession keeps private and uses for measurement after each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class RgbFrame:
    frame_id: int
    sim_time_s: float
    image: np.ndarray  # HxWx3 uint8, RGB order
    acquired_wall_monotonic: float


@dataclass(frozen=True)
class EgoState:
    """Ego localization telemetry (allowed runtime input). ENU frame, yaw CCW from +x."""

    x_m: float
    y_m: float
    yaw_rad: float
    speed_mps: float


@dataclass(frozen=True)
class CameraCalibration:
    """Static pinhole calibration; allowed as a perception input."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    height_m: float
    pitch_down_rad: float
    forward_offset_m: float


@dataclass(frozen=True)
class GraphNode:
    id: str
    x: float
    y: float
    label_uz: str
    label_en: str


@dataclass(frozen=True)
class GraphEdge:
    id: str
    a: str
    b: str
    polyline: tuple[tuple[float, float], ...]  # from node a to node b, metres

    @property
    def length_m(self) -> float:
        pts = np.asarray(self.polyline, dtype=float)
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


@dataclass(frozen=True)
class RoadGraph:
    """Static road topology (two-way edges). Allowed runtime navigation input."""

    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    lane_width_m: float = 3.5

    def node(self, node_id: str) -> GraphNode:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)


@dataclass(frozen=True)
class ControlCommand:
    steer: float  # [-1, 1], +1 = full left
    throttle: float  # [0, 1]
    brake: float  # [0, 1]
    input_frame_id: int
    safety_override: bool = False
    reason_codes: tuple[str, ...] = field(default_factory=tuple)


@runtime_checkable
class RuntimeSimulatorView(Protocol):
    """Everything runtime decision code may touch."""

    def get_rgb_frame(self) -> RgbFrame: ...

    def get_ego_telemetry(self) -> EgoState: ...

    def get_route_graph(self) -> RoadGraph: ...

    def get_camera_calibration(self) -> CameraCalibration: ...


class EvaluatorTruth(Protocol):
    """Evaluator-only access to simulator truth. Never passed to perception/FSM/controller."""

    def obstacle_positions(self) -> list[tuple[float, float]]: ...

    def collision_count(self) -> int: ...

    def lateral_deviation_m(self) -> float: ...

    def route_progress(self) -> float: ...


class SimulatorAdapter(RuntimeSimulatorView, Protocol):
    def reset(self, seed: int, scenario: str) -> None: ...

    def apply_control(self, command: ControlCommand) -> None: ...

    def step(self, dt: float) -> None: ...

    def close(self) -> None: ...

    def evaluator(self) -> EvaluatorTruth: ...
