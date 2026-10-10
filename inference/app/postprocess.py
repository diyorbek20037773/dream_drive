"""Decode YOLOPv2 outputs and map them back to original-image pixels.

- Detections: upstream ``split_for_trace_model`` + ``non_max_suppression`` (vendored, pinned commit), then the
  inverse letterbox: subtract the applied padding, divide by the per-axis content scale, clip to the image.
- Masks: crop the canvas to the content area (removing the applied padding), bilinearly resize the
  *probabilities* to the original (H, W) with ``align_corners=False``, then
  drivable = argmax over the 2 channels == 1 and lane = probability > 0.5.
  For a 1280x720 frame this is exactly upstream ``driving_area_mask``/``lane_line_mask`` (tested), which
  hard-code that frame size; this implementation works for any input size.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from app.preprocess import LetterboxGeometry
from app.vendored.yolopv2_utils import non_max_suppression, split_for_trace_model


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    score: float
    xyxy: tuple[float, float, float, float]
    model_id: str = "yolopv2"
    # Set when this box replaced another model's box in the ensemble class refinement (app.ensemble).
    refined_from_model: str | None = None

    def to_wire(self) -> dict[str, object]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "score": self.score,
            "xyxy_pixels": [round(v, 2) for v in self.xyxy],
            "model_id": self.model_id,
            "refined_from_model": self.refined_from_model,
        }


def class_name_for(class_id: int, class_map: Mapping[int, str]) -> str:
    """Verified names only; anything else is reported as unverified, never guessed."""
    return class_map.get(class_id, f"unverified_class_{class_id}")


def decode_detections(
    pred: Sequence[torch.Tensor], anchor_grid: Sequence[torch.Tensor], conf_thres: float, iou_thres: float
) -> torch.Tensor:
    """Raw heads -> (n, 6) float32 tensor [x1, y1, x2, y2, score, class] in canvas pixels (CPU)."""
    heads = [p.float() for p in pred]  # fp16 heads are decoded in fp32
    anchors = [a.float() for a in anchor_grid]
    decoded = split_for_trace_model(heads, anchors)
    det = non_max_suppression(decoded, conf_thres, iou_thres)[0]
    return det.detach().float().cpu()


def invert_boxes(boxes_canvas: np.ndarray, geometry: LetterboxGeometry) -> np.ndarray:
    """(n, 4) canvas xyxy -> original-image xyxy (float64), clipped to [0, W] x [0, H].

    The content area spans exactly ``content_width x content_height`` canvas pixels, which cv2.resize mapped
    onto the full ``W x H`` image, so the exact per-axis inverse scale is W / content_width (it differs from
    1 / r by at most the rounding of the resized size).
    """
    boxes = np.asarray(boxes_canvas, dtype=np.float64).reshape(-1, 4).copy()
    sx = geometry.image_width / geometry.content_width
    sy = geometry.image_height / geometry.content_height
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - geometry.pad_left) * sx
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - geometry.pad_top) * sy
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, geometry.image_width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, geometry.image_height)
    return boxes


def to_detections(det: torch.Tensor, geometry: LetterboxGeometry, class_map: Mapping[int, str]) -> list[Detection]:
    if det.numel() == 0:
        return []
    arr = det.numpy()
    boxes = invert_boxes(arr[:, :4], geometry)
    out: list[Detection] = []
    for (x1, y1, x2, y2), score, cls in zip(boxes, arr[:, 4], arr[:, 5], strict=True):
        if x2 <= x1 or y2 <= y1:
            continue  # box lies entirely in the padding: nothing of it is in the original image
        class_id = int(cls)
        out.append(
            Detection(
                class_id=class_id,
                class_name=class_name_for(class_id, class_map),
                score=float(min(max(score, 0.0), 1.0)),
                xyxy=(float(x1), float(y1), float(x2), float(y2)),
            )
        )
    out.sort(key=lambda d: d.score, reverse=True)
    return out


def _crop_content(t: torch.Tensor, geometry: LetterboxGeometry) -> torch.Tensor:
    if t.shape[-2:] != (geometry.canvas_height, geometry.canvas_width):
        raise ValueError(
            f"mask tensor {tuple(t.shape)} does not match canvas {geometry.canvas_height}x{geometry.canvas_width}"
        )
    top, left = geometry.pad_top, geometry.pad_left
    return t[:, :, top : top + geometry.content_height, left : left + geometry.content_width]


def invert_drivable(seg: torch.Tensor, geometry: LetterboxGeometry) -> np.ndarray:
    """seg [1, 2, Hc, Wc] probabilities -> bool [H, W] (argmax channel == 1)."""
    cropped = _crop_content(seg.float(), geometry)
    resized = F.interpolate(
        cropped, size=(geometry.image_height, geometry.image_width), mode="bilinear", align_corners=False
    )
    # torch.max (not argmax) to keep upstream's tie-breaking; ties resolve to channel 0 (not drivable).
    _, labels = torch.max(resized, 1)
    return (labels[0] == 1).cpu().numpy()


def invert_lane(ll: torch.Tensor, geometry: LetterboxGeometry) -> np.ndarray:
    """ll [1, 1, Hc, Wc] probabilities -> bool [H, W].

    Upstream uses ``torch.round(p) == 1``; torch.round rounds half to even, so for probabilities in [0, 1]
    that is exactly ``p > 0.5`` (0.5 itself is *not* lane).
    """
    cropped = _crop_content(ll.float(), geometry)
    resized = F.interpolate(
        cropped, size=(geometry.image_height, geometry.image_width), mode="bilinear", align_corners=False
    )
    return (resized[0, 0] > 0.5).cpu().numpy()


def encode_mask_png(mask: np.ndarray) -> str:
    """bool/uint8 HxW -> base64 PNG with values {0, 255} (8-bit grayscale)."""
    u8 = np.where(mask.astype(bool), np.uint8(255), np.uint8(0))
    ok, png = cv2.imencode(".png", u8, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(png.tobytes()).decode("ascii")


def decode_mask_png(b64: str) -> np.ndarray:
    """Inverse of encode_mask_png (used by tests and clients): base64 PNG -> uint8 HxW."""
    raw = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    mask = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2:
        raise ValueError("not a single-channel PNG mask")
    return mask
