"""SyntheticRgbPerception: classical pixel perception for LiteRoad renders (source ``synthetic_rgb_cv``).

Inputs: ONLY an RGB frame (HxWx3 uint8) and the static ``CameraCalibration``. This module never touches simulator
objects, scene lists or evaluator truth; it does not import the simulator package.

Algorithm (bounded synthetic operating domain: clear daylight, painted lanes, orange/white striped obstacles):
1. Frame validity: wrong shape/dtype -> ``invalid`` (CAMERA_INVALID); near-black -> ``missing`` (CAMERA_BLACK);
   near-uniform -> ``invalid`` (CAMERA_INVALID); uniform lower region -> ``degraded`` (partial occlusion).
2. HSV colour segmentation: white markings, yellow centre markings, orange obstacle paint.
3. Obstacles: orange connected components, merged across the white stripes, -> pixel boxes. The bottom edge of a box
   is its ground contact line; inverse perspective mapping (IPM) of that row on a flat ground plane with the static
   calibration gives a *synthetic-calibration distance estimate*. It is valid only for this flat synthetic world and
   is not metric depth of real scenes.
4. Lanes: IPM of marking pixels to the vehicle ground frame (X forward from the ego origin, Y left), robust
   quadratic RANSAC fits of the yellow centre line and white edge lines, right-lane centre estimate 1.75 m right of
   the centre line / left of the right edge, consistency check of the 3.5 m lane width.
5. Outputs: lane status valid/uncertain/unavailable with reasons, lane-centre lateral position and heading in the
   vehicle frame, ``image_lane_offset_normalized`` (horizontal image position of the lane centre 8 m ahead,
   (u - cx) / (W / 2): + = lane centre right of image centre), ``image_obstacle_cue`` in [0, 1] from the nearest box
   in the ego corridor (0 at >= 24 m, 1 at <= 8 m), plus overlay data (lane points, boxes) in frame pixels.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.schemas.reason_codes import ReasonCode
from app.schemas.world_state import Detection, Perception
from app.simulators.base import CameraCalibration

SOURCE = "synthetic_rgb_cv"
LANE_WIDTH_M = 3.5
HALF_LANE_M = LANE_WIDTH_M / 2
ROI_MIN_FWD_M = 3.0
ROI_MAX_FWD_M = 24.0
CORRIDOR_HALF_WIDTH_M = 1.6  # ego half width 0.9 m + 0.7 m margin
CUE_FAR_M = 24.0
CUE_NEAR_M = 8.0
IMAGE_OFFSET_REF_M = 8.0
OBSTACLE_CLASS_NAME = "synthetic_obstacle"

# HSV thresholds (OpenCV ranges H 0..180, S/V 0..255), tuned on LiteRoad renders.
_WHITE_S_MAX, _WHITE_V_MIN = 40, 185
_YELLOW_H, _YELLOW_S_MIN, _YELLOW_V_MIN = (19, 35), 120, 140
_ORANGE_H, _ORANGE_S_MIN, _ORANGE_V_MIN = (4, 18), 150, 120

_MAX_FIT_POINTS = 2500
_RANSAC_ITERS = 64
_RANSAC_TOL_M = 0.2
_MIN_INLIERS = 30
_MIN_SPAN_M = 3.0
_MAX_C2 = 0.04  # |curvature term|: lanes with radius >= ~12 m
_MAX_SLOPE = 1.0  # |dY/dX| at 6 m (45 deg)


@dataclass(frozen=True)
class CurveFit:
    """Y(X) = c0 + c1 X + c2 X^2 in the vehicle ground frame (metres, Y left positive)."""

    coeffs: tuple[float, float, float]
    inliers: int
    x_min: float
    x_max: float

    def y_at(self, x: float) -> float:
        c0, c1, c2 = self.coeffs
        return c0 + c1 * x + c2 * x * x

    def slope_at(self, x: float) -> float:
        return self.coeffs[1] + 2 * self.coeffs[2] * x

    def shifted(self, dy: float) -> CurveFit:
        c0, c1, c2 = self.coeffs
        return CurveFit((c0 + dy, c1, c2), self.inliers, self.x_min, self.x_max)


@dataclass(frozen=True)
class ObstacleBox:
    xyxy: tuple[int, int, int, int]  # pixels, x2/y2 exclusive
    fill_ratio: float  # orange pixels / box area (segmentation fill, not a calibrated probability)
    distance_m: float | None  # flat-ground IPM forward distance of the contact line from the ego origin
    lateral_m: tuple[float, float] | None  # (min, max) lateral extent of the contact line, + = left
    truncated: bool  # box touches the bottom image edge (closer than the visible ground)
    in_corridor: bool


@dataclass(frozen=True)
class SyntheticPerceptionResult:
    source: str
    quality_status: str  # valid | degraded | missing | invalid
    quality_codes: tuple[str, ...]  # CAMERA_BLACK / CAMERA_INVALID when the frame is unusable
    lane_status: str  # valid | uncertain | unavailable
    lane_centre: CurveFit | None
    lane_centre_left_m: float | None  # lane centre lateral position at the ego origin (+ = left of ego)
    lane_heading_rad: float | None  # lane direction relative to ego heading at 3 m ahead (+ = CCW / left)
    image_lane_offset_normalized: float | None
    image_obstacle_cue: float | None
    obstacles: tuple[ObstacleBox, ...]
    lane_points_px: tuple[tuple[float, float], ...]
    uncertainty_reasons: tuple[str, ...]
    frame_width: int
    frame_height: int
    timing_ms: float = 0.0
    model_id: str | None = None
    model_sha256: str | None = None
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.quality_status in ("valid", "degraded")

    def nearest_corridor_distance_m(self) -> float | None:
        ds = [o.distance_m for o in self.obstacles if o.in_corridor and o.distance_m is not None]
        return min(ds) if ds else None

    def to_perception(self) -> Perception:
        """world-state ``perception`` object (contracts/world-state.schema.json)."""
        dets = [
            Detection(
                track_id=None,
                class_id=0,
                class_name=OBSTACLE_CLASS_NAME,
                score=round(o.fill_ratio, 4),
                xyxy_pixels=[float(o.xyxy[0]), float(o.xyxy[1]), float(o.xyxy[2]), float(o.xyxy[3])],
            )
            for o in self.obstacles
        ]
        return Perception(
            source=self.source,  # type: ignore[arg-type]
            model_id=self.model_id,
            model_sha256=self.model_sha256,
            detections=dets,
            lane_mask_ref=None,
            drivable_mask_ref=None,
            lane_status=self.lane_status,  # type: ignore[arg-type]
            image_lane_offset_normalized=self.image_lane_offset_normalized,
            image_obstacle_cue=self.image_obstacle_cue,
            ttc_seconds=None,
            uncertainty_reasons=list(self.uncertainty_reasons),
        )

    def overlay(self) -> dict[str, object]:
        return {
            "obstacle_boxes": [[float(v) for v in o.xyxy] for o in self.obstacles],
            "lane_points": [[round(x, 1), round(y, 1)] for x, y in self.lane_points_px],
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
        }


def unavailable_result(
    calib: CameraCalibration, quality: str, codes: tuple[str, ...], reasons: tuple[str, ...], source: str = SOURCE
) -> SyntheticPerceptionResult:
    return SyntheticPerceptionResult(
        source=source,
        quality_status=quality,
        quality_codes=codes,
        lane_status="unavailable",
        lane_centre=None,
        lane_centre_left_m=None,
        lane_heading_rad=None,
        image_lane_offset_normalized=None,
        image_obstacle_cue=None,
        obstacles=(),
        lane_points_px=(),
        uncertainty_reasons=reasons,
        frame_width=calib.width,
        frame_height=calib.height,
    )


class GroundProjector:
    """Flat-ground inverse perspective mapping from the static calibration only."""

    def __init__(self, calib: CameraCalibration) -> None:
        self.calib = calib
        s, c = math.sin(calib.pitch_down_rad), math.cos(calib.pitch_down_rad)
        # camera axes in the vehicle frame (X forward, Y left, Z up); OpenCV camera x right, y down, z forward
        self._x_ax = np.array([0.0, -1.0, 0.0])
        self._y_ax = np.array([-s, 0.0, -c])
        self._z_ax = np.array([c, 0.0, -s])
        self._cam = np.array([calib.forward_offset_m, 0.0, calib.height_m])
        horizon = calib.cy - calib.fy * math.tan(calib.pitch_down_rad)
        self.first_ground_row = int(math.floor(horizon)) + 1
        rows = np.arange(calib.height, dtype=np.float64)
        cols = np.arange(calib.width, dtype=np.float64)
        xn = (cols + 0.5 - calib.cx) / calib.fx
        yn = (rows + 0.5 - calib.cy) / calib.fy
        xg, yg = np.meshgrid(xn, yn)
        rx = xg * self._x_ax[0] + yg * self._y_ax[0] + self._z_ax[0]
        ry = xg * self._x_ax[1] + yg * self._y_ax[1] + self._z_ax[1]
        rz = xg * self._x_ax[2] + yg * self._y_ax[2] + self._z_ax[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(rz < -1e-9, -calib.height_m / rz, np.nan)
        self.fwd = (self._cam[0] + t * rx).astype(np.float32)  # NaN above the horizon
        self.left = (t * ry).astype(np.float32)
        centre_fwd = self.fwd[:, calib.width // 2]
        roi_rows = np.nonzero(np.isfinite(centre_fwd) & (centre_fwd <= ROI_MAX_FWD_M + 4.0))[0]
        self.roi_top = int(roi_rows.min()) if len(roi_rows) else self.first_ground_row
        self.min_visible_fwd = float(np.nanmin(self.fwd[-1]))

    def ground_to_pixel(self, x: float, y: float) -> tuple[float, float] | None:
        q = np.array([x, y, 0.0]) - self._cam
        zc = float(q @ self._z_ax)
        if zc <= 0.1:
            return None
        u = self.calib.cx + self.calib.fx * float(q @ self._x_ax) / zc
        v = self.calib.cy + self.calib.fy * float(q @ self._y_ax) / zc
        return u, v


def _fit_poly(x: np.ndarray, y: np.ndarray, deg: int) -> tuple[float, float, float]:
    """Least-squares (exact for 3 points) polynomial Y = c0 + c1 X + c2 X^2 with X scaled by 1/10 for conditioning."""
    xs = x / 10.0
    cols = [np.ones_like(xs), xs] + ([xs * xs] if deg == 2 else [])
    a = np.stack(cols, axis=1)
    # normal equations on a 2x2 / 3x3 system: cheaper than an SVD-based lstsq and well conditioned after scaling
    c = np.linalg.solve(a.T @ a, a.T @ y)
    if not np.all(np.isfinite(c)):
        raise ValueError("degenerate fit")
    c0, c1 = float(c[0]), float(c[1]) / 10.0
    c2 = float(c[2]) / 100.0 if deg == 2 else 0.0
    return c0, c1, c2


def _plausible(coeffs: tuple[float, float, float]) -> bool:
    return abs(coeffs[2]) <= _MAX_C2 and abs(coeffs[1] + 2 * coeffs[2] * 6.0) <= _MAX_SLOPE


def ransac_curve(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[CurveFit, np.ndarray] | None:
    """Robust quadratic Y(X) fit; returns (fit, inlier mask) or None when no plausible marking line is present.

    All hypotheses are evaluated at once: each random triple defines an exact parabola (closed-form divided
    differences), implausible ones are discarded, and the hypothesis with the most inliers is refined by least
    squares (quadratic only when the inliers span at least 8 m)."""
    n = len(x)
    if n < _MIN_INLIERS:
        return None
    idx = np.sort(rng.integers(0, n, size=(_RANSAC_ITERS, 3)), axis=1)
    x0, x1, x2 = x[idx[:, 0]], x[idx[:, 1]], x[idx[:, 2]]
    y0, y1, y2 = y[idx[:, 0]], y[idx[:, 1]], y[idx[:, 2]]
    d01, d02, d12 = x1 - x0, x2 - x0, x2 - x1
    ok = (np.abs(d01) > 0.3) & (np.abs(d12) > 0.3) & (np.abs(d02) > 1.5)
    if not ok.any():
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        s01 = (y1 - y0) / d01
        s12 = (y2 - y1) / d12
        c2 = (s12 - s01) / d02
        c1 = s01 - c2 * (x0 + x1)
        c0 = y0 - c1 * x0 - c2 * x0 * x0
        ok &= np.isfinite(c2) & np.isfinite(c1) & (np.abs(c2) <= _MAX_C2) & (np.abs(c1 + 12.0 * c2) <= _MAX_SLOPE)
    if not ok.any():
        return None
    c0, c1, c2 = c0[ok], c1[ok], c2[ok]
    resid = np.abs(y[None, :] - (c0[:, None] + c1[:, None] * x[None, :] + c2[:, None] * (x * x)[None, :]))
    counts = (resid < _RANSAC_TOL_M).sum(axis=1)
    best = int(np.argmax(counts))
    if counts[best] < _MIN_INLIERS:
        return None
    mask = resid[best] < _RANSAC_TOL_M
    xi, yi = x[mask], y[mask]
    span = float(xi.max() - xi.min())
    if span < _MIN_SPAN_M:
        return None
    try:
        coeffs = _fit_poly(xi, yi, 2 if span >= 8.0 else 1)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not _plausible(coeffs):
        return None
    mask = np.abs(y - (coeffs[0] + coeffs[1] * x + coeffs[2] * x * x)) < _RANSAC_TOL_M
    if int(mask.sum()) < _MIN_INLIERS:
        return None
    return CurveFit(coeffs, int(mask.sum()), float(x[mask].min()), float(x[mask].max())), mask


def _average(a: CurveFit, b: CurveFit) -> CurveFit:
    coeffs = tuple((ca + cb) / 2 for ca, cb in zip(a.coeffs, b.coeffs, strict=True))
    return CurveFit(coeffs, a.inliers + b.inliers, min(a.x_min, b.x_min), max(a.x_max, b.x_max))  # type: ignore[arg-type]


class SyntheticRgbPerception:
    """Stateless per frame (deterministic for identical pixels). Construct once per session."""

    source = SOURCE

    def __init__(self, calibration: CameraCalibration) -> None:
        self.calib = calibration
        self.proj = GroundProjector(calibration)

    # ------------------------------------------------------------------------------------------ validity
    def _validate(self, image: object) -> SyntheticPerceptionResult | None:
        c = self.calib
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (c.height, c.width, 3):
            return unavailable_result(
                c,
                "invalid",
                (ReasonCode.CAMERA_INVALID.value,),
                (ReasonCode.CAMERA_INVALID.value, "frame_shape_or_dtype_invalid"),
            )
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean, std = cv2.meanStdDev(gray)
        if float(mean[0, 0]) < 10.0 and int(gray.max()) < 40:
            return unavailable_result(
                c, "missing", (ReasonCode.CAMERA_BLACK.value,), (ReasonCode.CAMERA_BLACK.value, "frame_black")
            )
        if float(std[0, 0]) < 4.0:
            return unavailable_result(
                c,
                "invalid",
                (ReasonCode.CAMERA_INVALID.value,),
                (ReasonCode.CAMERA_INVALID.value, "frame_near_uniform"),
            )
        return None

    # ------------------------------------------------------------------------------------------ obstacles
    def _obstacle_boxes(self, hsv: np.ndarray, row0: int) -> list[tuple[int, int, int, int, float]]:
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        mask = (
            (h_ch >= _ORANGE_H[0]) & (h_ch <= _ORANGE_H[1]) & (s_ch >= _ORANGE_S_MIN) & (v_ch >= _ORANGE_V_MIN)
        ).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        comps = [
            (int(st[0]), int(st[1]) + row0, int(st[0] + st[2]), int(st[1] + st[3]) + row0, int(st[4]))
            for st in stats[1:n]
            if st[4] >= 4
        ]
        # Merge stripe bands of the same box: strong horizontal overlap and a small vertical gap.
        merged = True
        while merged and len(comps) > 1:
            merged = False
            out: list[tuple[int, int, int, int, int]] = []
            used = [False] * len(comps)
            for i, a in enumerate(comps):
                if used[i]:
                    continue
                ax1, ay1, ax2, ay2, aa = a
                for j in range(i + 1, len(comps)):
                    if used[j]:
                        continue
                    bx1, by1, bx2, by2, ba = comps[j]
                    overlap = min(ax2, bx2) - max(ax1, bx1)
                    narrow = min(ax2 - ax1, bx2 - bx1)
                    gap = max(by1 - ay2, ay1 - by2)
                    tall = max(ay2 - ay1, by2 - by1)
                    if overlap >= 0.5 * narrow and gap <= max(2, 1.6 * tall):
                        ax1, ay1, ax2, ay2, aa = min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2), aa + ba
                        used[j] = True
                        merged = True
                out.append((ax1, ay1, ax2, ay2, aa))
                used[i] = True
            comps = out
        boxes = []
        for x1, y1, x2, y2, area in comps:
            w, h = x2 - x1, y2 - y1
            if w < 3 or h < 4:
                continue
            fill = area / float(w * h)
            if fill < 0.3:
                continue
            boxes.append((x1, y1, x2, y2, min(1.0, fill)))
        return boxes

    # ------------------------------------------------------------------------------------------ main entry
    def process(self, image: np.ndarray) -> SyntheticPerceptionResult:
        t0 = time.perf_counter()
        bad = self._validate(image)
        if bad is not None:
            return bad
        c = self.calib
        proj = self.proj
        reasons: list[str] = []
        quality = "valid"
        row0 = proj.first_ground_row
        ground = image[row0:]
        hsv = cv2.cvtColor(ground, cv2.COLOR_RGB2HSV)
        roi = image[proj.roi_top :]
        roi_std = float(cv2.meanStdDev(cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY))[1][0, 0])
        if roi_std < 4.0:
            quality = "degraded"
            reasons.append("camera_view_partially_occluded")

        raw_boxes = self._obstacle_boxes(hsv, row0)
        # Marking masks inside the lane ROI, excluding (dilated) obstacle boxes whose white stripes are not paint.
        r_off = proj.roi_top - row0
        hsv_roi = hsv[r_off:]
        h_ch, s_ch, v_ch = hsv_roi[:, :, 0], hsv_roi[:, :, 1], hsv_roi[:, :, 2]
        white = (s_ch <= _WHITE_S_MAX) & (v_ch >= _WHITE_V_MIN)
        yellow = (h_ch >= _YELLOW_H[0]) & (h_ch <= _YELLOW_H[1]) & (s_ch >= _YELLOW_S_MIN) & (v_ch >= _YELLOW_V_MIN)
        for x1, y1, x2, y2, _ in raw_boxes:
            pad_x, pad_y = max(3, (x2 - x1) // 3), max(3, (y2 - y1) // 3)
            ry1, ry2 = max(0, y1 - pad_y - proj.roi_top), max(0, y2 + pad_y - proj.roi_top)
            white[ry1:ry2, max(0, x1 - pad_x) : x2 + pad_x] = False
            yellow[ry1:ry2, max(0, x1 - pad_x) : x2 + pad_x] = False
        fwd = proj.fwd[proj.roi_top :]
        left = proj.left[proj.roi_top :]
        rng = np.random.default_rng(20261010)  # fixed seed: identical pixels -> identical output

        def points(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            ys, xs = np.nonzero(mask)
            f, lft = fwd[ys, xs], left[ys, xs]
            keep = np.isfinite(f) & (f >= ROI_MIN_FWD_M) & (f <= ROI_MAX_FWD_M)
            f, lft = f[keep].astype(np.float64), lft[keep].astype(np.float64)
            if len(f) > _MAX_FIT_POINTS:
                step = int(math.ceil(len(f) / _MAX_FIT_POINTS))
                f, lft = f[::step], lft[::step]
            return f, lft

        yf, yl = points(yellow)
        wf, wl = points(white)
        yellow_fit = ransac_curve(yf, yl, rng)
        whites: list[CurveFit] = []
        rest_f, rest_l = wf, wl
        for _ in range(3):
            res = ransac_curve(rest_f, rest_l, rng)
            if res is None:
                break
            fit, mask = res
            whites.append(fit)
            rest_f, rest_l = rest_f[~mask], rest_l[~mask]

        centre_fit, lane_status = self._lane_centre(yellow_fit[0] if yellow_fit else None, whites, reasons)

        lane_centre_left = lane_heading = img_offset = None
        lane_points: list[tuple[float, float]] = []
        if centre_fit is not None:
            lane_centre_left = centre_fit.y_at(0.0)
            lane_heading = math.atan(centre_fit.slope_at(ROI_MIN_FWD_M))
            uv = proj.ground_to_pixel(IMAGE_OFFSET_REF_M, centre_fit.y_at(IMAGE_OFFSET_REF_M))
            if uv is not None:
                img_offset = float(np.clip((uv[0] - c.cx) / (c.width / 2), -1.0, 1.0))
            x_hi = min(ROI_MAX_FWD_M, max(centre_fit.x_max, 8.0))
            for xf in np.arange(4.0, x_hi + 1e-6, 2.0):
                uv = proj.ground_to_pixel(float(xf), centre_fit.y_at(float(xf)))
                if uv is not None and 0 <= uv[0] < c.width and 0 <= uv[1] < c.height:
                    lane_points.append((uv[0], uv[1]))

        obstacles: list[ObstacleBox] = []
        for x1, y1, x2, y2, fill in raw_boxes:
            truncated = y2 >= c.height - 1
            vb = min(y2 - 1, c.height - 1)
            dist = float(proj.fwd[vb, (x1 + x2 - 1) // 2])
            l_a = float(proj.left[vb, x1])
            l_b = float(proj.left[vb, max(x1, x2 - 1)])
            if not math.isfinite(dist):
                continue
            lat = (min(l_a, l_b), max(l_a, l_b))
            path_y = centre_fit.y_at(dist) if centre_fit is not None else 0.0
            in_corr = lat[1] >= path_y - CORRIDOR_HALF_WIDTH_M and lat[0] <= path_y + CORRIDOR_HALF_WIDTH_M
            obstacles.append(ObstacleBox((x1, y1, x2, y2), fill, dist, lat, truncated, in_corr))
        cue = 0.0
        for o in obstacles:
            if not o.in_corridor or o.distance_m is None:
                continue
            o_cue = 1.0 if o.truncated else (CUE_FAR_M - o.distance_m) / (CUE_FAR_M - CUE_NEAR_M)
            cue = max(cue, float(np.clip(o_cue, 0.0, 1.0)))
        if obstacles:
            reasons.append("obstacle_distance_flat_ground_synthetic_estimate")
        return SyntheticPerceptionResult(
            source=SOURCE,
            quality_status=quality,
            quality_codes=(),
            lane_status=lane_status,
            lane_centre=centre_fit,
            lane_centre_left_m=lane_centre_left,
            lane_heading_rad=lane_heading,
            image_lane_offset_normalized=img_offset,
            image_obstacle_cue=round(cue, 4),
            obstacles=tuple(obstacles),
            lane_points_px=tuple(lane_points),
            uncertainty_reasons=tuple(reasons),
            frame_width=c.width,
            frame_height=c.height,
            timing_ms=(time.perf_counter() - t0) * 1000.0,
            extra={"yellow_points": float(len(yf)), "white_points": float(len(wf)), "white_lines": float(len(whites))},
        )

    @staticmethod
    def _lane_centre(
        yellow: CurveFit | None, whites: list[CurveFit], reasons: list[str]
    ) -> tuple[CurveFit | None, str]:
        ref = 6.0
        right = left_edge = None
        if yellow is not None:
            y6 = yellow.y_at(ref)
            cands_r = [w for w in whites if abs(w.y_at(ref) - (y6 - LANE_WIDTH_M)) <= 0.9]
            cands_l = [w for w in whites if abs(w.y_at(ref) - (y6 + LANE_WIDTH_M)) <= 0.9]
            right = min(cands_r, key=lambda w: abs(w.y_at(ref) - (y6 - LANE_WIDTH_M)), default=None)
            left_edge = min(cands_l, key=lambda w: abs(w.y_at(ref) - (y6 + LANE_WIDTH_M)), default=None)
        else:
            cands_r = [w for w in whites if -4.0 <= w.y_at(ref) <= -0.2]
            cands_l = [w for w in whites if 3.0 <= w.y_at(ref) <= 7.5]
            right = max(cands_r, key=lambda w: w.y_at(ref), default=None)
            left_edge = min(cands_l, key=lambda w: w.y_at(ref), default=None)
        if yellow is not None and right is not None:
            width = yellow.y_at(ref) - right.y_at(ref)
            if abs(width - LANE_WIDTH_M) <= 0.7:
                return _average(yellow.shifted(-HALF_LANE_M), right.shifted(HALF_LANE_M)), "valid"
            reasons.append("lane_width_inconsistent")
            return yellow.shifted(-HALF_LANE_M), "uncertain"
        if yellow is not None:
            reasons.append("right_edge_not_found")
            return yellow.shifted(-HALF_LANE_M), "uncertain"
        if right is not None:
            reasons.append("centre_line_not_found")
            return right.shifted(HALF_LANE_M), "uncertain"
        if left_edge is not None:
            reasons.append("only_far_edge_found")
            return left_edge.shifted(-(LANE_WIDTH_M + HALF_LANE_M)), "uncertain"
        reasons.append("no_lane_markings")
        return None, "unavailable"
