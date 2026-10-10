"""Evaluator-only truth for LiteRoad (collisions, lateral deviation, route progress, obstacle gaps).

This object reads the simulator's private scene. The SimulationSession uses it strictly after each physics step for
measurement and reporting; it is never passed to perception, the FSM or the controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from app.navigation.route import Route
from app.simulators.lite.dynamics import VehicleState, footprint

if TYPE_CHECKING:
    from app.simulators.lite.adapter import LiteRoadAdapter


@dataclass(frozen=True)
class Obstacle:
    id: int
    x: float
    y: float
    yaw: float
    size_m: float = 1.0
    height_m: float = 1.2
    visible: bool = True  # False = hidden evaluator-only obstacle (test of hidden-truth isolation)
    source: str = "scenario"  # "scenario" | "injected" | "hidden_test"

    def corners(self) -> list[tuple[float, float]]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        h = self.size_m / 2
        return [(self.x + lx * c - ly * s, self.y + lx * s + ly * c) for lx, ly in ((h, h), (-h, h), (-h, -h), (h, -h))]


def _axes(poly: np.ndarray) -> np.ndarray:
    edges = np.roll(poly, -1, axis=0) - poly
    normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def polygons_overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    """Separating axis test for two convex polygons."""
    pa, pb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    for axis in np.concatenate([_axes(pa), _axes(pb)]):
        ra, rb = pa @ axis, pb @ axis
        if ra.max() < rb.min() or rb.max() < ra.min():
            return False
    return True


def _point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(a + t * ab - p))


def polygon_distance(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Minimum distance between two convex polygons (0 when they overlap)."""
    if polygons_overlap(a, b):
        return 0.0
    pa, pb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    best = math.inf
    for p_set, q_set in ((pa, pb), (pb, pa)):
        for p in p_set:
            for i in range(len(q_set)):
                best = min(best, _point_segment(p, q_set[i], q_set[(i + 1) % len(q_set)]))
    return best


class LiteRoadEvaluator:
    """Implements app.simulators.base.EvaluatorTruth for one adapter instance."""

    def __init__(self, adapter: LiteRoadAdapter) -> None:
        self._adapter = adapter
        self._route: Route | None = None
        self._hint: int | None = None
        self._collided: set[int] = set()
        self._collision_events = 0

    # -- setup -------------------------------------------------------------------------------------------------
    def set_route(self, route: Route) -> None:
        self._route = route
        self._hint = None

    def reset(self) -> None:
        self._collided.clear()
        self._collision_events = 0
        self._hint = None

    # -- bookkeeping, called by the adapter after each physics step ---------------------------------------------
    def after_step(self, state: VehicleState, obstacles: list[Obstacle]) -> None:
        fp = footprint(state)
        for ob in obstacles:
            if ob.id not in self._collided and polygons_overlap(fp, ob.corners()):
                self._collided.add(ob.id)
                self._collision_events += 1

    # -- EvaluatorTruth ------------------------------------------------------------------------------------------
    def obstacle_positions(self) -> list[tuple[float, float]]:
        return [(ob.x, ob.y) for ob in self._adapter._obstacles]  # noqa: SLF001 - evaluator owns truth access

    def collision_count(self) -> int:
        return self._collision_events

    def _projection(self) -> tuple[float, float]:
        if self._route is None:
            raise RuntimeError("evaluator route not set")
        st = self._adapter._state  # noqa: SLF001
        proj = self._route.project(st.x, st.y, self._hint)
        self._hint = proj.index
        return proj.lateral_m, proj.s_m

    def lateral_deviation_m(self) -> float:
        """Absolute distance of the ego origin (rear-axle centre) from the truth lane-centre path of the route."""
        return abs(self._projection()[0])

    def route_progress(self) -> float:
        if self._route is None:
            raise RuntimeError("evaluator route not set")
        return self._route.progress(self._projection()[1])

    def min_obstacle_gap_m(self) -> float | None:
        """Distance between the ego footprint and the nearest obstacle footprint (None without obstacles)."""
        obs = self._adapter._obstacles  # noqa: SLF001
        if not obs:
            return None
        fp = footprint(self._adapter._state)  # noqa: SLF001
        return min(polygon_distance(fp, ob.corners()) for ob in obs)
