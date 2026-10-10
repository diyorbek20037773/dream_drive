"""CUDA fp16/fp32 vs CPU fp32 agreement on the same frames (collected only when INFERENCE_DEVICE=cuda), for both
models of the ensemble (YOLOPv2 and RF-DETR Nano; RF-DETR runs fp16 on CUDA unless its fp16 load failed).

No labels exist for these frames: agreement with the CPU fp32 reference is a numerical-consistency check of the
GPU path, not an accuracy measurement.
"""

from __future__ import annotations

import statistics
from typing import Any

import cv2
import numpy as np

from app.preprocess import canvas_to_input, compute_geometry, letterbox

COCO_FIXTURE = "coco_val2017_000000190923.jpg"


def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def test_cuda_fp16_agrees_with_cpu_fp32(
    real_engine: Any, smoke_settings: Any, example_bytes: bytes, evidence: dict
) -> None:
    assert real_engine.device_name == "cuda"
    from app.model import YolopV2Engine

    cpu_settings = smoke_settings.model_copy(update={"inference_device": "cpu", "inference_precision": "fp32"})
    cpu_engine = YolopV2Engine(cpu_settings)
    assert cpu_engine.load(), cpu_engine.status()

    base = cv2.imdecode(np.frombuffer(example_bytes, np.uint8), cv2.IMREAD_COLOR)
    from conftest import COCO_IMAGE

    frames = {
        "example": base,
        "example_hflip": cv2.flip(base, 1),
        "example_dark": (base.astype(np.float32) * 0.5).astype(np.uint8),
        "coco_190923": cv2.imdecode(np.frombuffer(COCO_IMAGE.read_bytes(), np.uint8), cv2.IMREAD_COLOR),
    }
    per_frame: dict[str, Any] = {}
    matched_total = cpu_total = gpu_total = 0
    rf_matched = rf_cpu_total = rf_gpu_total = 0
    for name, img in frames.items():
        h, w = img.shape[:2]
        g = compute_geometry(w, h)
        x = canvas_to_input(letterbox(img, g))
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        ref = cpu_engine.infer(x, g, 0.1, 0.45, rgb)
        out = real_engine.infer(x, g, 0.1, 0.45, rgb)
        # RF-DETR compared on its own outputs at a 0.3 score threshold (borderline low scores may flip in fp16).
        rf_ref = [d for d in cpu_engine.rfdetr.predict(rgb, 0.3)]
        rf_out = [d for d in real_engine.rfdetr.predict(rgb, 0.3)]
        rf_hits = 0
        used_rf: set[int] = set()
        for rd in rf_ref:
            for j, od in enumerate(rf_out):
                if j not in used_rf and od.class_id == rd.class_id and _iou(list(rd.xyxy), list(od.xyxy)) >= 0.5:
                    used_rf.add(j)
                    rf_hits += 1
                    break
        rf_matched += rf_hits
        rf_cpu_total += len(rf_ref)
        rf_gpu_total += len(rf_out)
        ref_boxes = [list(d.xyxy) for d in ref.detections if d.model_id == "yolopv2"]
        gpu_boxes = [list(d.xyxy) for d in out.detections if d.model_id == "yolopv2"]
        used: set[int] = set()
        ious: list[float] = []
        for rb in ref_boxes:
            best, best_j = 0.0, -1
            for j, gb in enumerate(gpu_boxes):
                if j not in used and (v := _iou(rb, gb)) > best:
                    best, best_j = v, j
            if best >= 0.5:
                used.add(best_j)
                ious.append(best)
        matched_total += len(ious)
        cpu_total += len(ref_boxes)
        gpu_total += len(gpu_boxes)
        per_frame[name] = {
            "cpu_detections": len(ref_boxes),
            "gpu_detections": len(gpu_boxes),
            "matched_iou_ge_0_5": len(ious),
            "mean_matched_iou": round(statistics.fmean(ious), 4) if ious else None,
            "drivable_mask_iou": round(_mask_iou(ref.drivable, out.drivable), 5),
            "drivable_pixel_agreement": round(float((ref.drivable == out.drivable).mean()), 6),
            "lane_mask_iou": round(_mask_iou(ref.lane, out.lane), 5),
            "lane_pixel_agreement": round(float((ref.lane == out.lane).mean()), 6),
            "rfdetr_cpu_detections_ge_0_3": len(rf_ref),
            "rfdetr_gpu_detections_ge_0_3": len(rf_out),
            "rfdetr_matched_same_class_iou_ge_0_5": rf_hits,
        }
    evidence["cpu_fp32_vs_cuda"] = {
        "precision": real_engine.precision,
        "frames": per_frame,
        "detections_matched": matched_total,
        "cpu_detections_total": cpu_total,
        "gpu_detections_total": gpu_total,
        "rfdetr_precision": real_engine.rfdetr.precision,
        "rfdetr_detections_matched": rf_matched,
        "rfdetr_cpu_detections_total": rf_cpu_total,
        "rfdetr_gpu_detections_total": rf_gpu_total,
    }
    # Tolerances: fp16 may flip borderline boxes/pixels; gross disagreement means a broken GPU path.
    assert cpu_total > 0 and matched_total / cpu_total >= 0.8
    assert rf_cpu_total > 0 and rf_matched / rf_cpu_total >= 0.8
    for stats in per_frame.values():
        assert stats["drivable_pixel_agreement"] >= 0.99
        assert stats["lane_pixel_agreement"] >= 0.98
