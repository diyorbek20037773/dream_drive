"""Lane-centre route path from the static road graph.

The planner concatenates edge centrelines in travel direction, rounds the corners that occur inside junction
boxes with a circular fillet, then offsets the result 1.75 m to the right of the travel direction (right-hand
traffic, docs/IMPLEMENTATION_SPEC.md section 4) and resamples it to evenly spaced waypoints.
All quantities are metres / radians in the ENU world frame; curvature is signed (+ = turning left / CCW).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from app.navigation.astar import RouteUnreachable, astar
from app.simulators.base import RoadGraph

LANE_OFFSET_M = 1.75
WAYPOINT_SPACING_M = 0.5
JUNCTION_FILLET_RADIUS_M = 8.0
_MIN_FILLET_ANGLE_RAD = math.radians(20.0)


@dataclass(frozen=True)
class Projection:
    index: int  # index of the segment start waypoint
    s_m: float  # arc length along the path of the projected point
    lateral_m: float  # signed distance of the query point from the path, + = left of travel direction
    heading_rad: float  # path tangent heading at the projection


def _resample(points: np.ndarray, spacing: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-9])
    points = points[keep]
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(math.ceil(s[-1] / spacing)) + 1)
    s_new = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(s_new, s, points[:, 0]), np.interp(s_new, s, points[:, 1])], axis=1)


def _turn_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    d1, d2 = b - a, c - b
    n1, n2 = float(np.linalg.norm(d1)), float(np.linalg.norm(d2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    d1, d2 = d1 / n1, d2 / n2
    return math.atan2(d1[0] * d2[1] - d1[1] * d2[0], float(d1 @ d2))


def _fillet(points: np.ndarray, radius: float) -> np.ndarray:
    """Replace sharp polyline corners (turn > 20 deg) by circular arcs of up to ``radius`` metres.

    A fillet may use up to 95 % of an adjacent segment unless the vertex at the segment's other end is filleted too
    (then 45 %, so the two arcs never overlap)."""
    if len(points) < 3:
        return points
    n_pts = len(points)
    sharp = (
        [False]
        + [
            abs(_turn_angle(points[i - 1], points[i], points[i + 1])) >= _MIN_FILLET_ANGLE_RAD
            for i in range(1, n_pts - 1)
        ]
        + [False]
    )
    out: list[np.ndarray] = [points[0]]
    for i in range(1, n_pts - 1):
        a, b, c = points[i - 1], points[i], points[i + 1]
        l1 = float(np.linalg.norm(b - a))
        l2 = float(np.linalg.norm(c - b))
        if l1 < 1e-9 or l2 < 1e-9:
            continue
        if not sharp[i]:
            out.append(b)
            continue
        d1 = (b - a) / l1
        theta = _turn_angle(a, b, c)
        half = abs(theta) / 2.0
        t = min(radius * math.tan(half), (0.45 if sharp[i - 1] else 0.95) * l1, (0.45 if sharp[i + 1] else 0.95) * l2)
        r = t / math.tan(half)
        p1 = b - d1 * t
        left = np.array([-d1[1], d1[0]])
        centre = p1 + left * r * math.copysign(1.0, theta)
        a0 = math.atan2(p1[1] - centre[1], p1[0] - centre[0])
        n = max(3, int(math.ceil(abs(theta) * r / 0.5)) + 1)
        for k in range(n):
            ang = a0 + theta * k / (n - 1)
            out.append(centre + r * np.array([math.cos(ang), math.sin(ang)]))
    out.append(points[-1])
    return np.asarray(out, dtype=float)


def _headings(points: np.ndarray) -> np.ndarray:
    d = np.gradient(points, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


@dataclass(frozen=True)
class Route:
    node_ids: tuple[str, ...]
    path: np.ndarray  # (N, 2) lane-centre waypoints in travel order
    s: np.ndarray  # (N,) cumulative arc length
    heading: np.ndarray  # (N,) tangent heading, rad CCW from +x
    curvature: np.ndarray  # (N,) signed curvature, 1/m

    @property
    def length_m(self) -> float:
        return float(self.s[-1])

    @property
    def destination(self) -> tuple[float, float]:
        return float(self.path[-1, 0]), float(self.path[-1, 1])

    def to_dict(self) -> dict[str, object]:
        """Wire shape ``{node_ids, polyline, length_m}`` (polyline decimated to ~2 m for the UI)."""
        step = max(1, int(round(2.0 / WAYPOINT_SPACING_M)))
        idx = list(range(0, len(self.path), step))
        if idx[-1] != len(self.path) - 1:
            idx.append(len(self.path) - 1)
        return {
            "node_ids": list(self.node_ids),
            "polyline": [[round(float(self.path[i, 0]), 3), round(float(self.path[i, 1]), 3)] for i in idx],
            "length_m": round(self.length_m, 3),
        }

    def point_at(self, s_m: float) -> tuple[float, float]:
        s_m = min(max(s_m, 0.0), self.length_m)
        return float(np.interp(s_m, self.s, self.path[:, 0])), float(np.interp(s_m, self.s, self.path[:, 1]))

    def heading_at(self, s_m: float) -> float:
        i = int(np.clip(np.searchsorted(self.s, s_m), 0, len(self.s) - 1))
        return float(self.heading[i])

    def max_abs_curvature(self, s0: float, s1: float) -> float:
        i0 = int(np.searchsorted(self.s, max(0.0, s0)))
        i1 = int(np.searchsorted(self.s, min(self.length_m, s1), side="right"))
        if i1 <= i0:
            i1 = min(i0 + 1, len(self.s))
        return float(np.abs(self.curvature[i0:i1]).max())

    def progress(self, s_m: float) -> float:
        return float(min(max(s_m / self.length_m, 0.0), 1.0)) if self.length_m > 0 else 1.0

    def project(self, x: float, y: float, hint_index: int | None = None, window: int = 120) -> Projection:
        """Closest point on the path. With ``hint_index`` only +-``window`` waypoints are searched, which keeps the
        projection on the correct branch where the route passes near itself."""
        n = len(self.path)
        if hint_index is None:
            lo, hi = 0, n - 1
        else:
            lo, hi = max(0, hint_index - window), min(n - 1, hint_index + window)
        a = self.path[lo:hi]
        b = self.path[lo + 1 : hi + 1]
        ab = b - a
        p = np.array([x, y])
        denom = np.maximum((ab * ab).sum(axis=1), 1e-12)
        t = np.clip(((p - a) * ab).sum(axis=1) / denom, 0.0, 1.0)
        closest = a + ab * t[:, None]
        d2 = ((closest - p) ** 2).sum(axis=1)
        k = int(np.argmin(d2))
        i = lo + k
        seg = ab[k]
        cross = seg[0] * (p[1] - a[k, 1]) - seg[1] * (p[0] - a[k, 0])
        lateral = math.copysign(math.sqrt(float(d2[k])), cross) if d2[k] > 0 else 0.0
        s_m = float(self.s[i] + t[k] * math.sqrt(float(denom[k])))
        return Projection(index=i, s_m=s_m, lateral_m=lateral, heading_rad=float(math.atan2(seg[1], seg[0])))


def _centreline(graph: RoadGraph, node_ids: list[str]) -> np.ndarray:
    by_pair = {}
    for e in graph.edges:
        by_pair[(e.a, e.b)] = list(e.polyline)
        by_pair[(e.b, e.a)] = list(reversed(e.polyline))
    pts: list[tuple[float, float]] = []
    for a, b in zip(node_ids[:-1], node_ids[1:], strict=True):
        poly = by_pair[(a, b)]
        pts.extend(poly if not pts else poly[1:])
    return np.asarray(pts, dtype=float)


def build_route(graph: RoadGraph, node_ids: list[str] | tuple[str, ...]) -> Route:
    """Lane-centre path for a node sequence (consecutive nodes must share an edge)."""
    node_ids = list(node_ids)
    if len(node_ids) < 2:
        raise RouteUnreachable("a route needs at least two nodes", code="same_start_destination")
    centre = _resample(_fillet(_centreline(graph, node_ids), JUNCTION_FILLET_RADIUS_M), WAYPOINT_SPACING_M)
    hdg = _headings(centre)
    right = np.stack([np.sin(hdg), -np.cos(hdg)], axis=1)  # unit vector to the right of travel
    lane = _resample(centre + LANE_OFFSET_M * right, WAYPOINT_SPACING_M)
    heading = np.unwrap(_headings(lane))
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(lane, axis=0), axis=1))])
    ds = np.maximum(np.gradient(s), 1e-9)
    curvature = np.gradient(heading) / ds
    # 5-tap moving average removes resampling noise at arc joints without hiding real turns.
    curvature = np.convolve(np.pad(curvature, 2, mode="edge"), np.ones(5) / 5.0, mode="valid")
    return Route(node_ids=tuple(node_ids), path=lane, s=s, heading=np.unwrap(heading), curvature=curvature)


def plan_route(graph: RoadGraph, start: str, destination: str) -> Route:
    return build_route(graph, astar(graph, start, destination))


class RoadSurface:
    """Static-topology distance queries (allowed runtime input) used by the out-of-bounds safety check."""

    def __init__(self, graph: RoadGraph, junction_half_size_m: float, junction_ids: list[str]) -> None:
        segs = []
        for e in graph.edges:
            pts = np.asarray(e.polyline, dtype=float)
            segs.append(np.concatenate([pts[:-1], pts[1:]], axis=1))
        self._segs = np.concatenate(segs, axis=0)
        self._boxes = [(graph.node(n).x, graph.node(n).y) for n in junction_ids]
        self._half = junction_half_size_m

    def distance_to_centreline(self, x: float, y: float) -> float:
        a = self._segs[:, 0:2]
        ab = self._segs[:, 2:4] - a
        p = np.array([x, y])
        t = np.clip(((p - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-12), 0.0, 1.0)
        d = float(np.sqrt((((a + ab * t[:, None]) - p) ** 2).sum(axis=1).min()))
        for bx, by in self._boxes:
            if abs(x - bx) <= self._half and abs(y - by) <= self._half:
                return 0.0
        return d


@lru_cache(maxsize=64)
def _cached_route(graph: RoadGraph, start: str, destination: str) -> Route:
    return plan_route(graph, start, destination)


def cached_route(graph: RoadGraph, start: str, destination: str) -> Route:
    """Memoised ``plan_route`` (RoadGraph is a frozen, hashable dataclass)."""
    return _cached_route(graph, start, destination)
