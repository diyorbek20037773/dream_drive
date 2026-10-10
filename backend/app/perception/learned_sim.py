"""LearnedModelSimPerception: the *same* rendered LiteRoad RGB frames sent to the learned inference service.

Gated: SimulationSession only constructs this when ``LEARNED_SIM_MODE_ENABLED=true`` (the lead enables it only after
tests/scenarios/test_learned_sim_validation.py passes against a live service). Synthetic renders are out of the
YOLOPv2 training domain; failure to generalise is a reportable result, never hidden.

Mapping (no simulator truth is used):
- frame validity: identical black / near-uniform checks as the synthetic pipeline (CAMERA_BLACK / CAMERA_INVALID);
- lane: the model's binary lane-line mask -> flat-ground IPM with the static calibration -> robust line fits ->
  the nearest line left of the ego (left boundary) and right of the ego (right boundary) -> lane centre;
- obstacles: every model detection box (whatever class the verified class map names) -> bottom-edge IPM distance
  -> corridor test -> ``image_obstacle_cue`` with the same 24 m -> 8 m mapping as the synthetic pipeline;
- errors: InferenceTimeout -> PERCEPTION_TIMEOUT, any other client error -> INFERENCE_FAILED (frame quality missing).
Source ``learned_model`` with the model id and checkpoint SHA256 reported by the service for every frame.
"""

from __future__ import annotations

import base64
import dataclasses
import math
import time
from typing import Any

import cv2
import numpy as np

from app.perception.synthetic_rgb_cv import (
    CORRIDOR_HALF_WIDTH_M,
    CUE_FAR_M,
    CUE_NEAR_M,
    HALF_LANE_M,
    IMAGE_OFFSET_REF_M,
    LANE_WIDTH_M,
    ROI_MAX_FWD_M,
    ROI_MIN_FWD_M,
    CurveFit,
    GroundProjector,
    ObstacleBox,
    SyntheticPerceptionResult,
    SyntheticRgbPerception,
    ransac_curve,
    unavailable_result,
)
from app.schemas.reason_codes import ReasonCode
from app.simulators.base import CameraCalibration

SOURCE = "learned_model"
JPEG_QUALITY = 95
_MAX_POINTS = 4000


def _decode_mask(b64png: str | None, shape: tuple[int, int]) -> np.ndarray | None:
    if not b64png:
        return None
    buf = np.frombuffer(base64.b64decode(b64png), dtype=np.uint8)
    mask = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != shape:
        return None
    return mask > 127


class LearnedModelSimPerception:
    source = SOURCE

    def __init__(self, client: Any, calibration: CameraCalibration, conf_threshold: float = 0.1) -> None:
        self.client = client
        self.calib = calibration
        self.proj = GroundProjector(calibration)
        self._validator = SyntheticRgbPerception(calibration)
        self.conf_threshold = conf_threshold
        self.model_id: str | None = None
        self.model_sha256: str | None = None
        self.last_timing_ms: dict[str, float] = {}

    def _unavailable(self, quality: str, code: str, reason: str) -> SyntheticPerceptionResult:
        if self.model_id and self.model_sha256:
            base = unavailable_result(self.calib, quality, (code,), (code, reason), source=SOURCE)
            return dataclasses.replace(base, model_id=self.model_id, model_sha256=self.model_sha256)
        # Before the first successful inference no model identity is known; a learned_model perception record must
        # carry model id + SHA256, so the unusable frame is reported with source ``unavailable``.
        return unavailable_result(self.calib, quality, (code,), (code, reason), source="unavailable")

    def process(self, image: np.ndarray, frame_id: int = 0) -> SyntheticPerceptionResult:
        from app.perception.client import InferenceClientError, InferenceTimeout

        t0 = time.perf_counter()
        bad = self._validator._validate(image)  # noqa: SLF001 - shared pixel validity rules
        if bad is not None:
            code = bad.quality_codes[0] if bad.quality_codes else ReasonCode.CAMERA_INVALID.value
            return self._unavailable(bad.quality_status, code, "frame_invalid_before_inference")
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return self._unavailable("invalid", ReasonCode.CAMERA_INVALID.value, "jpeg_encode_failed")
        try:
            res = self.client.infer(
                buf.tobytes(),
                frame_id,
                content_type="image/jpeg",
                conf_threshold=self.conf_threshold,
                include_masks=True,
            )
        except InferenceTimeout:
            return self._unavailable("missing", ReasonCode.PERCEPTION_TIMEOUT.value, "inference_timeout")
        except InferenceClientError as exc:
            return self._unavailable("missing", ReasonCode.INFERENCE_FAILED.value, f"inference_error:{exc.code}")
        self.model_id, self.model_sha256 = res.model.model_id, res.model.sha256
        self.last_timing_ms = res.timing_ms.model_dump()
        return self._map(res, (time.perf_counter() - t0) * 1000.0)

    # ---------------------------------------------------------------------------------------------- mapping
    def _map(self, res: Any, elapsed_ms: float) -> SyntheticPerceptionResult:
        c, proj = self.calib, self.proj
        reasons: list[str] = ["synthetic_render_outside_model_training_domain"]
        lane_mask = _decode_mask(res.lane_mask_png, (c.height, c.width))
        centre: CurveFit | None = None
        lane_status = "unavailable"
        if lane_mask is None:
            reasons.append("lane_mask_missing")
        else:
            roi = lane_mask[proj.roi_top :]
            ys, xs = np.nonzero(roi)
            f = proj.fwd[proj.roi_top :][ys, xs]
            lft = proj.left[proj.roi_top :][ys, xs]
            keep = np.isfinite(f) & (f >= ROI_MIN_FWD_M) & (f <= ROI_MAX_FWD_M)
            f, lft = f[keep].astype(np.float64), lft[keep].astype(np.float64)
            if len(f) > _MAX_POINTS:
                step = int(math.ceil(len(f) / _MAX_POINTS))
                f, lft = f[::step], lft[::step]
            rng = np.random.default_rng(20261010)
            lines: list[CurveFit] = []
            for _ in range(4):
                fit = ransac_curve(f, lft, rng)
                if fit is None:
                    break
                lines.append(fit[0])
                f, lft = f[~fit[1]], lft[~fit[1]]
            left_b = min((ln for ln in lines if 0.3 <= ln.y_at(6.0) <= 3.5), key=lambda ln: ln.y_at(6.0), default=None)
            right_b = max(
                (ln for ln in lines if -3.5 <= ln.y_at(6.0) <= -0.3), key=lambda ln: ln.y_at(6.0), default=None
            )
            if left_b is not None and right_b is not None:
                width = left_b.y_at(6.0) - right_b.y_at(6.0)
                cl, cr = left_b.coeffs, right_b.coeffs
                coeffs = ((cl[0] + cr[0]) / 2, (cl[1] + cr[1]) / 2, (cl[2] + cr[2]) / 2)
                centre = CurveFit(
                    coeffs,
                    left_b.inliers + right_b.inliers,
                    min(left_b.x_min, right_b.x_min),
                    max(left_b.x_max, right_b.x_max),
                )
                lane_status = "valid" if abs(width - LANE_WIDTH_M) <= 0.7 else "uncertain"
                if lane_status == "uncertain":
                    reasons.append("lane_width_inconsistent")
            elif left_b is not None:
                centre, lane_status = left_b.shifted(-HALF_LANE_M), "uncertain"
                reasons.append("right_boundary_not_found")
            elif right_b is not None:
                centre, lane_status = right_b.shifted(HALF_LANE_M), "uncertain"
                reasons.append("left_boundary_not_found")
            else:
                reasons.append("no_lane_lines_from_model")
        lane_left = lane_heading = img_offset = None
        lane_points: list[tuple[float, float]] = []
        if centre is not None:
            lane_left = centre.y_at(0.0)
            lane_heading = math.atan(centre.slope_at(ROI_MIN_FWD_M))
            uv = proj.ground_to_pixel(IMAGE_OFFSET_REF_M, centre.y_at(IMAGE_OFFSET_REF_M))
            if uv is not None:
                img_offset = float(np.clip((uv[0] - c.cx) / (c.width / 2), -1.0, 1.0))
            for xf in np.arange(4.0, min(ROI_MAX_FWD_M, max(centre.x_max, 8.0)) + 1e-6, 2.0):
                uv = proj.ground_to_pixel(float(xf), centre.y_at(float(xf)))
                if uv is not None and 0 <= uv[0] < c.width and 0 <= uv[1] < c.height:
                    lane_points.append(uv)
        obstacles: list[ObstacleBox] = []
        cue = 0.0
        for det in res.detections:
            x1, y1, x2, y2 = (int(round(v)) for v in det.xyxy_pixels)
            x1, x2 = max(0, min(x1, c.width - 1)), max(0, min(x2, c.width))
            y1, y2 = max(0, min(y1, c.height - 1)), max(0, min(y2, c.height))
            if x2 <= x1 or y2 <= y1:
                continue
            vb = min(y2 - 1, c.height - 1)
            dist = float(proj.fwd[vb, (x1 + x2 - 1) // 2])
            if not math.isfinite(dist):
                continue
            la, lb = float(proj.left[vb, x1]), float(proj.left[vb, x2 - 1])
            lat = (min(la, lb), max(la, lb))
            path_y = centre.y_at(dist) if centre is not None else 0.0
            in_corr = lat[1] >= path_y - CORRIDOR_HALF_WIDTH_M and lat[0] <= path_y + CORRIDOR_HALF_WIDTH_M
            truncated = y2 >= c.height - 1
            obstacles.append(ObstacleBox((x1, y1, x2, y2), float(det.score), dist, lat, truncated, in_corr))
            if in_corr:
                o_cue = 1.0 if truncated else (CUE_FAR_M - dist) / (CUE_FAR_M - CUE_NEAR_M)
                cue = max(cue, float(np.clip(o_cue, 0.0, 1.0)))
        return SyntheticPerceptionResult(
            source=SOURCE,
            quality_status="valid",
            quality_codes=(),
            lane_status=lane_status,
            lane_centre=centre,
            lane_centre_left_m=lane_left,
            lane_heading_rad=lane_heading,
            image_lane_offset_normalized=img_offset,
            image_obstacle_cue=round(cue, 4),
            obstacles=tuple(obstacles),
            lane_points_px=tuple(lane_points),
            uncertainty_reasons=tuple(reasons),
            frame_width=c.width,
            frame_height=c.height,
            timing_ms=elapsed_ms,
            model_id=res.model.model_id,
            model_sha256=res.model.sha256,
        )
