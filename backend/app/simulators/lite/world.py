"""Static LiteRoad world: top-down ground raster (built once per process) and static roadside boxes.

Raster layout: 0.1 m per pixel, row 0 = north edge (``Y_MAX``), column 0 = west edge (``X_MIN``), RGB uint8.
Layers: textured grass, roadside ground features (lake, gravel patches), textured asphalt roads 7 m wide,
white solid edge lines at +-3.5 m, yellow dashed centre line (3 m dash / 3 m gap). Junction boxes (nodes of
degree >= 3) are plain asphalt squares without markings. Two coarser mip levels reduce aliasing of far rows.

The colours below are part of the synthetic operating domain and are documented in docs/SIMULATION_EVIDENCE.md;
the pixel perception uses its own HSV thresholds and never imports this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from app.simulators.base import RoadGraph
from app.simulators.lite.graph import JUNCTION_HALF_SIZE_M, ROAD_HALF_WIDTH_M, build_town_graph, junction_nodes

RES_M = 0.1
X_MIN, X_MAX = -50.0, 250.0
Y_MIN, Y_MAX = -50.0, 195.0
LINE_WIDTH_M = 0.2
DASH_M = 3.0
GAP_M = 3.0

GRASS_RGB = (78, 128, 58)
ASPHALT_RGB = (80, 81, 84)
WHITE_RGB = (236, 236, 232)
YELLOW_RGB = (232, 188, 36)
LAKE_RGB = (58, 104, 150)
GRAVEL_RGB = (122, 116, 104)
DEPOT_PAD_RGB = (96, 96, 100)
FENCE_RGB = (60, 60, 64)


@dataclass(frozen=True)
class Box3D:
    """Axis box standing on the ground; (x, y) is its footprint centre in the ENU world frame."""

    x: float
    y: float
    yaw: float
    length: float  # along yaw
    width: float
    height: float
    kind: str  # "obstacle" (orange/white striped), "building", "tree"
    color: tuple[int, int, int] = (200, 180, 150)
    roof_color: tuple[int, int, int] = (90, 92, 98)


@dataclass
class WorldRaster:
    levels: list[np.ndarray]  # mip pyramid, level k has resolution RES_M * 2**k
    x_min: float
    y_max: float
    res_m: float
    static_boxes: tuple[Box3D, ...]
    graph: RoadGraph

    def to_pixel(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.x_min) / self.res_m, (self.y_max - y) / self.res_m


def _px(points: np.ndarray) -> np.ndarray:
    """World metres (N,2) -> raster pixel coordinates scaled by 16 for cv2 ``shift=4`` sub-pixel drawing."""
    cols = (points[:, 0] - X_MIN) / RES_M
    rows = (Y_MAX - points[:, 1]) / RES_M
    return np.round(np.stack([cols, rows], axis=1) * 16).astype(np.int32)


def _offset_polyline(points: np.ndarray, offset_m: float) -> np.ndarray:
    """Offset to the left (+) / right (-) of the polyline direction using vertex-averaged normals."""
    d = np.diff(points, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    seg_n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    vn = np.empty_like(points)
    vn[0] = seg_n[0]
    vn[-1] = seg_n[-1]
    for i in range(1, len(points) - 1):
        m = seg_n[i - 1] + seg_n[i]
        m /= np.linalg.norm(m)
        # miter length correction keeps the offset distance exact at gentle bends
        vn[i] = m / max(0.5, float(m @ seg_n[i]))
    return points + offset_m * vn


def _densify(points: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(math.ceil(s[-1] / step)) + 1)
    s_new = np.linspace(0.0, s[-1], n)
    pts = np.stack([np.interp(s_new, s, points[:, 0]), np.interp(s_new, s, points[:, 1])], axis=1)
    return pts, s_new


def _static_boxes() -> tuple[Box3D, ...]:
    beige = (200, 180, 150)
    grey = (140, 140, 146)
    tree = (44, 92, 44)
    return (
        Box3D(25.0, 22.0, 0.0, 14.0, 10.0, 7.0, "building", beige),
        Box3D(95.0, -22.0, 0.0, 18.0, 9.0, 9.0, "building", grey),
        Box3D(95.0, 25.0, 0.0, 16.0, 16.0, 6.0, "building", beige),
        Box3D(160.0, 30.0, 0.3, 10.0, 8.0, 5.0, "building", grey),
        Box3D(100.0, 100.0, 0.0, 20.0, 14.0, 8.0, "building", beige),
        Box3D(160.0, 110.0, 0.0, 14.0, 10.0, 12.0, "building", grey),
        Box3D(30.0, -15.0, 0.0, 2.2, 2.2, 4.5, "tree", tree),
        Box3D(45.0, -16.0, 0.0, 2.2, 2.2, 5.0, "tree", tree),
        Box3D(75.0, 45.0, 0.0, 2.2, 2.2, 4.0, "tree", tree),
        Box3D(115.0, 52.0, 0.0, 2.2, 2.2, 4.5, "tree", tree),
        Box3D(145.0, 85.0, 0.0, 2.2, 2.2, 5.0, "tree", tree),
        Box3D(40.0, 95.0, 0.0, 2.2, 2.2, 4.0, "tree", tree),
        Box3D(200.0, 20.0, 0.0, 2.2, 2.2, 4.5, "tree", tree),
        Box3D(118.0, -15.0, 0.0, 2.2, 2.2, 4.0, "tree", tree),
    )


def _build(graph: RoadGraph, seed: int = 7) -> WorldRaster:
    w = int(round((X_MAX - X_MIN) / RES_M))
    h = int(round((Y_MAX - Y_MIN) / RES_M))
    rng = np.random.default_rng(seed)
    # Low-frequency texture: random field at 1/8 resolution, upsampled (cheap and smooth) plus fine grain.
    coarse = rng.normal(0.0, 1.0, size=(h // 8 + 1, w // 8 + 1)).astype(np.float32)
    tex = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    fine = rng.integers(-6, 7, size=(h, w), dtype=np.int8).astype(np.float32)

    img = np.empty((h, w, 3), dtype=np.uint8)
    grass_noise = tex * 9.0 + fine * 0.7
    for c in range(3):
        img[:, :, c] = np.clip(GRASS_RGB[c] + grass_noise, 0, 255).astype(np.uint8)
    del grass_noise

    def paint(mask: np.ndarray, rgb: tuple[int, int, int], noise: np.ndarray | None) -> None:
        for c in range(3):
            vals = np.full(int(mask.sum()), float(rgb[c]), dtype=np.float32)
            if noise is not None:
                vals += noise[mask]
            img[:, :, c][mask] = np.clip(vals, 0, 255).astype(np.uint8)

    # Roadside ground features (drawn before roads so roads overwrite them).
    feat = np.zeros((h, w), dtype=np.uint8)
    lake = np.array(
        [
            [-55.0, 95.0],
            [-40.0, 125.0],
            [-45.0, 175.0],
            [-25.0, 185.0],
            [-5.0, 168.0],
            [-8.0, 140.0],
            [-30.0, 120.0],
            [-35.0, 98.0],
        ]
    )
    cv2.fillPoly(feat, [_px(lake)], 1, lineType=cv2.LINE_8, shift=4)
    gravel = [
        np.array([[70.0, -40.0], [110.0, -40.0], [110.0, -30.0], [70.0, -30.0]]),
        np.array([[150.0, 60.0], [170.0, 60.0], [170.0, 90.0], [150.0, 90.0]]),
    ]
    for poly in gravel:
        cv2.fillPoly(feat, [_px(poly)], 2, shift=4)
    depot = graph.node("Q")
    pad = np.array(
        [
            [depot.x - 12, depot.y - 10],
            [depot.x + 12, depot.y - 10],
            [depot.x + 12, depot.y + 10],
            [depot.x - 12, depot.y + 10],
        ]
    )
    cv2.fillPoly(feat, [_px(pad)], 3, shift=4)
    paint(feat == 1, LAKE_RGB, tex * 5.0)
    paint(feat == 2, GRAVEL_RGB, fine * 1.2)
    paint(feat == 3, DEPOT_PAD_RGB, fine * 0.5)
    cv2.polylines(feat, [_px(pad)], True, 4, thickness=4, shift=4)  # fence around the closed depot
    paint(feat == 4, FENCE_RGB, None)

    # Road surface mask: thick polylines (7 m) for edges plus junction squares.
    road = np.zeros((h, w), dtype=np.uint8)
    thick = int(round(2 * ROAD_HALF_WIDTH_M / RES_M))
    for e in graph.edges:
        cv2.polylines(road, [_px(np.asarray(e.polyline))], False, 1, thickness=thick, lineType=cv2.LINE_8, shift=4)
    junction = np.zeros((h, w), dtype=np.uint8)
    for nid in junction_nodes(graph):
        n = graph.node(nid)
        hs = JUNCTION_HALF_SIZE_M
        sq = np.array([[n.x - hs, n.y - hs], [n.x + hs, n.y - hs], [n.x + hs, n.y + hs], [n.x - hs, n.y + hs]])
        cv2.fillPoly(junction, [_px(sq)], 1, shift=4)
    road |= junction
    paint(road.astype(bool), ASPHALT_RGB, tex * 3.0 + fine * 0.5)

    # Markings: white edges at +-3.5 m, yellow dashed centre; erased inside junction boxes.
    white = np.zeros((h, w), dtype=np.uint8)
    yellow = np.zeros((h, w), dtype=np.uint8)
    lw = max(1, int(round(LINE_WIDTH_M / RES_M)))
    for e in graph.edges:
        pts = np.asarray(e.polyline, dtype=float)
        dense, s = _densify(pts, 0.25)
        for off in (ROAD_HALF_WIDTH_M - LINE_WIDTH_M / 2, -(ROAD_HALF_WIDTH_M - LINE_WIDTH_M / 2)):
            cv2.polylines(white, [_px(_offset_polyline(dense, off))], False, 1, thickness=lw, shift=4)
        on = (s % (DASH_M + GAP_M)) < DASH_M
        start = None
        for i, flag in enumerate(on):
            if flag and start is None:
                start = i
            if (not flag or i == len(on) - 1) and start is not None:
                end = i if flag else i - 1
                if end > start:
                    cv2.polylines(yellow, [_px(dense[start : end + 1])], False, 1, thickness=lw, shift=4)
                start = None
    keep = junction == 0
    img[(white == 1) & keep] = WHITE_RGB
    img[(yellow == 1) & keep] = YELLOW_RGB

    levels: list[np.ndarray] = [img]
    for _ in range(2):
        prev = levels[-1]
        levels.append(cv2.resize(prev, (prev.shape[1] // 2, prev.shape[0] // 2), interpolation=cv2.INTER_AREA))
    return WorldRaster(levels=levels, x_min=X_MIN, y_max=Y_MAX, res_m=RES_M, static_boxes=_static_boxes(), graph=graph)


@lru_cache(maxsize=1)
def get_world() -> WorldRaster:
    """The shared static world (about 30 MB, built in well under a second, read-only afterwards)."""
    world = _build(build_town_graph())
    for lvl in world.levels:
        lvl.setflags(write=False)
    return world
