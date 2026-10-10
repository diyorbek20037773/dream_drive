"""Fast perspective renderer for LiteRoad (CPU, no GPU).

Ground: every pixel below the horizon has a fixed ray in the vehicle frame (the camera is rigidly mounted), so its
flat-ground intersection (forward X, left Y) is precomputed once. Per frame the ego pose rotates/translates those
points into raster coordinates and ``cv2.remap`` samples the static world raster (with mip levels for far rows).
Boxes (obstacles, buildings, trees) are clipped against the near plane, projected and painted far-to-near
(painter's algorithm; visible faces of a convex box never overlap each other).

Camera convention (docs/IMPLEMENTATION_SPEC.md section 4): pinhole 640x360, horizontal FOV 90 deg, mounted 1.4 m
above ground and 1.0 m ahead of the ego origin (rear-axle centre), pitched 8 deg down, no roll.
Vehicle frame: X forward, Y left, Z up. Camera frame (OpenCV): x right, y down, z optical axis.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.simulators.base import CameraCalibration
from app.simulators.lite.world import Box3D, WorldRaster

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
CAMERA_HFOV_DEG = 90.0
CAMERA_HEIGHT_M = 1.4
CAMERA_FORWARD_M = 1.0
CAMERA_PITCH_DOWN_DEG = 8.0

SKY_TOP_RGB = np.array((92, 150, 220), dtype=np.float32)
SKY_HORIZON_RGB = np.array((168, 198, 232), dtype=np.float32)
HAZE_RGB = np.array((165, 190, 215), dtype=np.float32)
ORANGE_RGB = np.array((255, 112, 16), dtype=np.float32)
STRIPE_WHITE_RGB = np.array((245, 245, 245), dtype=np.float32)

_NEAR_Z = 0.05
_MAX_BOX_DISTANCE_M = 140.0
_MIP_SPLITS_M = (22.0, 55.0)  # forward distance where mip level 1 / 2 start
_FOG_START_M, _FOG_SPAN_M, _FOG_MAX = 55.0, 140.0, 0.85
_MAX_GROUND_M = 400.0
_SUN = np.array([0.35, -0.55, 0.76])


def default_calibration() -> CameraCalibration:
    f = (CAMERA_WIDTH / 2) / math.tan(math.radians(CAMERA_HFOV_DEG) / 2)
    return CameraCalibration(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fx=f,
        fy=f,
        cx=CAMERA_WIDTH / 2,
        cy=CAMERA_HEIGHT / 2,
        height_m=CAMERA_HEIGHT_M,
        pitch_down_rad=math.radians(CAMERA_PITCH_DOWN_DEG),
        forward_offset_m=CAMERA_FORWARD_M,
    )


def camera_axes(pitch_down_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera x (right), y (down), z (optical) axes expressed in the vehicle frame."""
    s, c = math.sin(pitch_down_rad), math.cos(pitch_down_rad)
    x_ax = np.array([0.0, -1.0, 0.0])
    y_ax = np.array([-s, 0.0, -c])
    z_ax = np.array([c, 0.0, -s])
    return x_ax, y_ax, z_ax


class LiteRenderer:
    def __init__(self, world: WorldRaster, calib: CameraCalibration) -> None:
        self.world = world
        self.calib = calib
        w, h = calib.width, calib.height
        self._x_ax, self._y_ax, self._z_ax = camera_axes(calib.pitch_down_rad)
        self._cam = np.array([calib.forward_offset_m, 0.0, calib.height_m])
        horizon = calib.cy - calib.fy * math.tan(calib.pitch_down_rad)
        self.horizon_row = int(math.floor(horizon)) + 1  # first row whose centre ray points below the horizon
        u = (np.arange(w, dtype=np.float64) + 0.5 - calib.cx) / calib.fx
        rows = np.arange(self.horizon_row, h, dtype=np.float64)
        v = (rows + 0.5 - calib.cy) / calib.fy
        xn, yn = np.meshgrid(u, v)
        rx = xn * self._x_ax[0] + yn * self._y_ax[0] + self._z_ax[0]
        ry = xn * self._x_ax[1] + yn * self._y_ax[1] + self._z_ax[1]
        rz = xn * self._x_ax[2] + yn * self._y_ax[2] + self._z_ax[2]
        t = np.where(rz < -1e-9, -calib.height_m / np.minimum(rz, -1e-9), np.inf)
        gx = self._cam[0] + t * rx
        gy = t * ry
        dist = np.hypot(gx, gy)
        far = ~np.isfinite(dist) | (dist > _MAX_GROUND_M)
        gx[far] = _MAX_GROUND_M
        gy[far] = 0.0
        self._gx = gx.astype(np.float32)
        self._gy = gy.astype(np.float32)
        self._far = far
        fog = np.clip((dist - _FOG_START_M) / _FOG_SPAN_M, 0.0, _FOG_MAX)
        fog[far] = 1.0
        fog_rows = np.nonzero(fog.max(axis=1) > 0)[0]
        self._fog_end = int(fog_rows.max()) + 1 if len(fog_rows) else 0
        self._fog = fog[: self._fog_end, :, None].astype(np.float32)
        centre_fwd = gx[:, w // 2]
        self._bands: list[tuple[int, int, int]] = []  # (row0, row1, mip level) in ground-row indices
        level_of_row = np.where(centre_fwd >= _MIP_SPLITS_M[1], 2, np.where(centre_fwd >= _MIP_SPLITS_M[0], 1, 0))
        start = 0
        for i in range(1, len(rows) + 1):
            if i == len(rows) or level_of_row[i] != level_of_row[start]:
                self._bands.append((start, i, int(level_of_row[start])))
                start = i
        self._hg = self._pixel_to_ground_h()
        self._border = tuple(int(v) for v in self.world.levels[0][0, 0])
        frac = (np.arange(self.horizon_row, dtype=np.float32) / max(1, self.horizon_row - 1))[:, None]
        sky = SKY_TOP_RGB * (1 - frac) + SKY_HORIZON_RGB * frac
        self._sky = np.repeat(sky[:, None, :], w, axis=1).astype(np.uint8)

    # ------------------------------------------------------------------------------------------------ ground
    def _pixel_to_ground_h(self) -> np.ndarray:
        """3x3 homography: ground-image pixel (u, v_band=0 at ``horizon_row``) -> vehicle ground point (X, Y, 1)."""
        c = self.calib
        k_inv = np.array(
            [
                [1.0 / c.fx, 0.0, (0.5 - c.cx) / c.fx],
                [0.0, 1.0 / c.fy, (0.5 + self.horizon_row - c.cy) / c.fy],
                [0.0, 0.0, 1.0],
            ]
        )
        rot = np.stack([self._x_ax, self._y_ax, self._z_ax], axis=1)  # ray = rot @ [xn, yn, 1]
        g = np.array([[-c.height_m, 0.0, self._cam[0]], [0.0, -c.height_m, 0.0], [0.0, 0.0, 1.0]])
        return g @ rot @ k_inv

    def _ground(self, x: float, y: float, yaw: float) -> np.ndarray:
        """Ground rows via one perspective warp per mip band (pixel -> vehicle ground -> world -> raster)."""
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        pose = np.array([[cy_, -sy_, x], [sy_, cy_, y], [0.0, 0.0, 1.0]])
        res = self.world.res_m
        to_raster = np.array(
            [[1.0 / res, 0.0, -self.world.x_min / res], [0.0, -1.0 / res, self.world.y_max / res], [0.0, 0.0, 1.0]]
        )
        base = to_raster @ pose @ self._hg
        out = np.empty(self._gx.shape + (3,), dtype=np.uint8)
        width = self.calib.width
        for r0, r1, lvl in self._bands:
            scale = float(2**lvl)
            mip = np.array(
                [[1.0 / scale, 0.0, 0.5 / scale - 0.5], [0.0, 1.0 / scale, 0.5 / scale - 0.5], [0.0, 0.0, 1.0]]
            )
            shift = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, float(r0)], [0.0, 0.0, 1.0]])
            m = mip @ base @ shift
            src = self.world.levels[lvl]
            out[r0:r1] = cv2.warpPerspective(
                src,
                m,
                (width, r1 - r0),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self._border,
            )
        if self._fog_end:
            band = out[: self._fog_end].astype(np.float32)
            out[: self._fog_end] = (band * (1 - self._fog) + HAZE_RGB * self._fog).astype(np.uint8)
        return out

    # ------------------------------------------------------------------------------------------------ boxes
    def _to_camera(self, pts_world: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
        c, s = math.cos(yaw), math.sin(yaw)
        dx = pts_world[:, 0] - x
        dy = pts_world[:, 1] - y
        q = np.stack(
            [dx * c + dy * s - self._cam[0], -dx * s + dy * c - self._cam[1], pts_world[:, 2] - self._cam[2]], axis=1
        )
        return np.stack([q @ self._x_ax, q @ self._y_ax, q @ self._z_ax], axis=1)

    @staticmethod
    def _clip_near(poly: np.ndarray) -> np.ndarray:
        """Sutherland-Hodgman clip of a camera-frame polygon against z >= _NEAR_Z."""
        out = []
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            ina, inb = a[2] >= _NEAR_Z, b[2] >= _NEAR_Z
            if ina:
                out.append(a)
            if ina != inb:
                t = (_NEAR_Z - a[2]) / (b[2] - a[2])
                out.append(a + t * (b - a))
        return np.asarray(out) if out else np.zeros((0, 3))

    def _project(self, cam_pts: np.ndarray) -> np.ndarray:
        u = self.calib.cx + self.calib.fx * cam_pts[:, 0] / cam_pts[:, 2]
        v = self.calib.cy + self.calib.fy * cam_pts[:, 1] / cam_pts[:, 2]
        return np.round(np.stack([u, v], axis=1) * 16).astype(np.int32)  # cv2 shift=4 sub-pixel coordinates

    def _fill(self, img: np.ndarray, poly_cam: np.ndarray, rgb: np.ndarray) -> None:
        clipped = self._clip_near(poly_cam)
        if len(clipped) < 3:
            return
        pts = self._project(clipped)
        lim = 16 * 4 * max(self.calib.width, self.calib.height)
        pts = np.clip(pts, -lim, lim)
        color = tuple(int(v) for v in np.clip(rgb, 0, 255))
        cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA, shift=4)

    def _draw_box(self, img: np.ndarray, box: Box3D, x: float, y: float, yaw: float) -> None:
        cb, sb = math.cos(box.yaw), math.sin(box.yaw)
        hl, hw = box.length / 2, box.width / 2
        local = ((hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw))  # CCW seen from above
        base = np.array([(box.x + lx * cb - ly * sb, box.y + lx * sb + ly * cb) for lx, ly in local])
        n_levels = 5 if box.kind == "obstacle" else 1  # obstacle: orange/white/orange/white/orange bands
        heights = np.linspace(0.0, box.height, n_levels + 1)
        grid = np.concatenate([np.column_stack([base, np.full(4, z)]) for z in heights])  # (4 * (n+1), 3)
        cam = self._to_camera(grid, x, y, yaw)
        if (cam[:, 2] < _NEAR_Z).all():
            return
        zc = np.maximum(cam[:, 2], _NEAR_Z)
        u = cam[:, 0] / zc
        if (u > 1.2 * self.calib.cx / self.calib.fx).all() or (u < -1.2 * self.calib.cx / self.calib.fx).all():
            return  # completely outside the horizontal field of view
        all_front = bool((cam[:, 2] >= _NEAR_Z).all())
        proj = self._project(cam) if all_front else None
        cam_pos = np.array([x + self._cam[0] * math.cos(yaw), y + self._cam[0] * math.sin(yaw), self._cam[2]])
        for i in range(4):
            a, b = base[i], base[(i + 1) % 4]
            edge = b - a
            normal = np.array([edge[1], -edge[0], 0.0]) / max(1e-9, float(np.linalg.norm(edge)))
            centre = np.array([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, box.height / 2])
            if normal @ (cam_pos - centre) <= 0:
                continue
            shade = 0.8 + 0.2 * max(0.0, float(normal @ _SUN))
            j = (i + 1) % 4
            for k in range(n_levels):
                idx = [4 * k + i, 4 * k + j, 4 * (k + 1) + j, 4 * (k + 1) + i]
                if box.kind == "obstacle":
                    rgb = (ORANGE_RGB if k % 2 == 0 else STRIPE_WHITE_RGB) * shade
                else:
                    rgb = np.array(box.color, dtype=float) * shade
                self._fill_idx(img, cam, proj, idx, rgb)
        if cam_pos[2] > box.height:  # top face visible only when the camera is above it
            top = [4 * n_levels + i for i in range(4)]
            top_rgb = ORANGE_RGB if box.kind == "obstacle" else np.array(box.roof_color, dtype=float)
            self._fill_idx(img, cam, proj, top, top_rgb)

    def _fill_idx(
        self, img: np.ndarray, cam: np.ndarray, proj: np.ndarray | None, idx: list[int], rgb: np.ndarray
    ) -> None:
        color = tuple(int(v) for v in np.clip(rgb, 0, 255))
        if proj is not None:
            pts = proj[idx]
            lim = 16 * 4 * max(self.calib.width, self.calib.height)
            cv2.fillPoly(img, [np.clip(pts, -lim, lim)], color, lineType=cv2.LINE_AA, shift=4)
        else:
            self._fill(img, cam[idx], rgb)

    def render(self, x: float, y: float, yaw: float, boxes: list[Box3D]) -> np.ndarray:
        """RGB uint8 image (H, W, 3) seen from the ego pose (rear-axle centre x, y; yaw CCW from +x)."""
        h, w = self.calib.height, self.calib.width
        img = np.empty((h, w, 3), dtype=np.uint8)
        img[: self.horizon_row] = self._sky
        img[self.horizon_row :] = self._ground(x, y, yaw)
        cam_x = x + self._cam[0] * math.cos(yaw)
        cam_y = y + self._cam[0] * math.sin(yaw)
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        candidates = []
        for box in boxes:
            d = np.array([box.x - cam_x, box.y - cam_y])
            dist = float(np.hypot(d[0], d[1]))
            reach = math.hypot(box.length, box.width) / 2
            if dist > _MAX_BOX_DISTANCE_M or float(d @ fwd) < -reach:
                continue
            candidates.append((dist, box))
        for _, box in sorted(candidates, key=lambda item: -item[0]):
            self._draw_box(img, box, x, y, yaw)
        return img
