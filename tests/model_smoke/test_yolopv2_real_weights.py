"""Two-model (YOLOPv2 + RF-DETR Nano) real-weight smoke + benchmark on the upstream example frame
(tests/fixtures/images/yolopv2_example.jpg). The RF-DETR class-mapping check is in test_rfdetr_real_weights.py.

Shape/identity/sanity checks only: there are no labels here, so nothing in this file is an accuracy claim.
"""

from __future__ import annotations

import hashlib
import os
import statistics
import time
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.manifest import load_manifest, model_path, verify_model_file
from app.postprocess import decode_mask_png
from app.preprocess import canvas_to_input, compute_geometry, letterbox

WARMUPS = 2
RUNS = max(10, int(os.environ.get("DD_SMOKE_RUNS", "10")))
UNION_CLASS_MAP = {
    "3": "vehicle",
    "101": "person",
    "102": "bicycle",
    "104": "motorcycle",
    "106": "bus",
    "108": "truck",
    "110": "traffic_light",
    "113": "stop_sign",
}


def _pct(values: list[float], q: float) -> float:
    return round(float(np.percentile(np.asarray(values), q)), 3)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "p50": _pct(values, 50),
        "p95": _pct(values, 95),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.fmean(values), 3),
    }


@pytest.fixture(scope="module")
def client(real_engine: Any, smoke_settings: Any) -> Any:
    with TestClient(create_app(smoke_settings, real_engine)) as c:
        yield c


def _check_result(body: dict[str, Any], width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    assert body["image"] == {"width": width, "height": height}
    for det in body["detections"]:
        x1, y1, x2, y2 = det["xyxy_pixels"]
        assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, det
        assert 0.0 <= det["score"] <= 1.0
        if det["model_id"] == "yolopv2":
            assert det["class_name"] == ("vehicle" if det["class_id"] == 3 else f"unverified_class_{det['class_id']}")
        else:
            assert det["model_id"] == "rfdetr_nano"
            assert UNION_CLASS_MAP[str(det["class_id"])] == det["class_name"] and det["class_id"] > 100
            assert det["refined_from_model"] in (None, "yolopv2")
    da = decode_mask_png(body["drivable_mask_png"])
    ll = decode_mask_png(body["lane_mask_png"])
    assert da.shape == ll.shape == (height, width)
    assert set(np.unique(da).tolist()) <= {0, 255} and set(np.unique(ll).tolist()) <= {0, 255}
    assert body["mask_stats"]["drivable_fraction"] == pytest.approx(float((da == 255).mean()), abs=1e-9)
    assert body["mask_stats"]["lane_fraction"] == pytest.approx(float((ll == 255).mean()), abs=1e-9)
    return da, ll


def test_checkpoint_identity_matches_manifest(real_engine: Any, smoke_settings: Any, evidence: dict) -> None:
    manifest = load_manifest(smoke_settings.model_manifest_path, smoke_settings.model_id)
    res = verify_model_file(model_path(smoke_settings.model_root, manifest), manifest)
    assert res.ok, res.detail
    assert res.actual_sha256 == manifest.sha256 == real_engine.verified_sha256
    assert res.actual_size == manifest.size_bytes
    assert real_engine.ready


def test_real_inference_on_upstream_example(
    client: Any, example_bytes: bytes, smoke_settings: Any, evidence: dict
) -> None:
    ready = client.get("/health/ready")
    assert ready.status_code == 200 and ready.json()["ready"] is True
    info = client.get("/v1/model").json()
    assert info["ready"] is True and info["sha256"] == evidence["model"]["manifest_sha256"]
    assert info["class_map"] == UNION_CLASS_MAP
    assert info["ensemble_sha256"] == evidence["ensemble"]["ensemble_sha256"] is not None
    assert [m["sha256"] for m in info["models"]] == [
        evidence["model"]["manifest_sha256"],
        evidence["rfdetr_model"]["manifest_sha256"],
    ]

    r = client.post("/v1/infer?frame_id=7", content=example_bytes, headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"]["sha256"] == evidence["model"]["manifest_sha256"]
    assert body["model"]["device"] == smoke_settings.inference_device
    img = cv2.imdecode(np.frombuffer(example_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    _check_result(body, w, h)
    assert [m["model_id"] for m in body["models"]] == ["yolopv2", "rfdetr_nano"]
    assert body["ensemble_sha256"] == evidence["ensemble"]["ensemble_sha256"]
    dets = body["detections"]
    yolo = [d for d in dets if d["model_id"] == "yolopv2"]
    assert len(yolo) > 0, "no YOLOPv2 detections on the upstream example frame"
    assert {d["class_id"] for d in yolo} == {3}
    assert all(d["class_name"] == "vehicle" for d in yolo)
    stats = body["mask_stats"]
    assert 0.0 < stats["drivable_fraction"] < 1.0
    assert 0.0 < stats["lane_fraction"] < 1.0
    evidence["input"] = {
        "file": "tests/fixtures/images/yolopv2_example.jpg",
        "sha256": hashlib.sha256(example_bytes).hexdigest(),
        "width": w,
        "height": h,
        "geometry": body["geometry"],
    }
    evidence["outputs_example"] = {
        "conf_threshold": 0.1,
        "iou_threshold": 0.45,
        "detections": len(dets),
        "detections_by_model": {m: sum(1 for d in dets if d["model_id"] == m) for m in ("yolopv2", "rfdetr_nano")},
        "refined_from_yolopv2": sum(1 for d in dets if d["refined_from_model"] == "yolopv2"),
        "class_ids": sorted({d["class_id"] for d in dets}),
        "class_counts": {
            str(c): sum(1 for d in dets if d["class_id"] == c) for c in sorted({d["class_id"] for d in dets})
        },
        "scores": [round(d["score"], 4) for d in dets],
        "drivable_fraction": round(stats["drivable_fraction"], 5),
        "lane_fraction": round(stats["lane_fraction"], 5),
        "timing_ms": body["timing_ms"],
    }


@pytest.mark.parametrize("variant", ["portrait_rot90", "crop_4x3", "downscaled_640x360"])
def test_real_inference_other_geometries(client: Any, example_bytes: bytes, variant: str, evidence: dict) -> None:
    """Exercises the letterbox/inversion path with real outputs on non-16:9 sizes (shape sanity, no quality claim)."""
    img = cv2.imdecode(np.frombuffer(example_bytes, np.uint8), cv2.IMREAD_COLOR)
    if variant == "portrait_rot90":
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif variant == "crop_4x3":
        img = img[:, 160:1120]  # 960x720
    else:
        img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    r = client.post("/v1/infer?frame_id=1", content=buf.tobytes(), headers={"Content-Type": "image/png"})
    assert r.status_code == 200, r.text
    body = r.json()
    h, w = img.shape[:2]
    _check_result(body, w, h)
    evidence.setdefault("outputs_geometry_variants", {})[variant] = {
        "width": w,
        "height": h,
        "geometry": body["geometry"],
        "detections": len(body["detections"]),
        "detections_by_model": {
            m: sum(1 for d in body["detections"] if d["model_id"] == m) for m in ("yolopv2", "rfdetr_nano")
        },
        "class_ids": sorted({d["class_id"] for d in body["detections"]}),
        "drivable_fraction": round(body["mask_stats"]["drivable_fraction"], 5),
        "lane_fraction": round(body["mask_stats"]["lane_fraction"], 5),
    }


def test_benchmark_warm_latency(real_engine: Any, client: Any, example_bytes: bytes, evidence: dict) -> None:
    import torch

    img = cv2.imdecode(np.frombuffer(example_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    g = compute_geometry(w, h)
    if real_engine.device_name == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Engine level: preprocess + both models + decode/NMS + mask inversion + refinement (no HTTP, no PNG encoding).
    engine_total, forward, post, pre, yolo_fwd, rf_fwd = [], [], [], [], [], []
    for i in range(WARMUPS + RUNS):
        t0 = time.perf_counter()
        x = canvas_to_input(letterbox(img, g))
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        t1 = time.perf_counter()
        out = real_engine.infer(x, g, 0.1, 0.45, rgb)
        t2 = time.perf_counter()
        if i >= WARMUPS:
            pre.append((t1 - t0) * 1000)
            forward.append(out.inference_ms)
            yolo_fwd.append(out.inference_ms_by_model["yolopv2"])
            rf_fwd.append(out.inference_ms_by_model["rfdetr_nano"])
            post.append(out.postprocess_ms)
            engine_total.append((t2 - t0) * 1000)

    # Service level: full POST /v1/infer through the ASGI app (decode, PNG masks, JSON), server-side timing.
    http_wall, server_total = [], []
    for i in range(WARMUPS + RUNS):
        t0 = time.perf_counter()
        r = client.post(f"/v1/infer?frame_id={i}", content=example_bytes, headers={"Content-Type": "image/jpeg"})
        t1 = time.perf_counter()
        assert r.status_code == 200
        if i >= WARMUPS:
            http_wall.append((t1 - t0) * 1000)
            server_total.append(r.json()["timing_ms"]["total"])

    bench: dict[str, Any] = {
        "warmup_runs": WARMUPS,
        "timed_runs": RUNS,
        "image": f"{w}x{h}",
        "canvas": f"{g.canvas_width}x{g.canvas_height}",
        "percentile_method": "numpy.percentile linear interpolation",
        "engine_preprocess_ms": _summary(pre),
        "engine_forward_ms": _summary(forward),
        "engine_forward_yolopv2_ms": _summary(yolo_fwd),
        "engine_rfdetr_predict_ms": _summary(rf_fwd),
        "rfdetr_predict_includes": "rfdetr resize 384x384 + normalise + forward + top-300 post-processing",
        "engine_decode_nms_mask_inversion_ms": _summary(post),
        "engine_total_ms": _summary(engine_total),
        "http_server_total_ms": _summary(server_total),
        "http_client_wall_ms": _summary(http_wall),
        "effective_fps_at_p50_http": round(1000.0 / _pct(http_wall, 50), 3),
        "torch_threads": torch.get_num_threads(),
        "rfdetr_precision": real_engine.rfdetr.precision,
    }
    if real_engine.device_name == "cuda":
        bench["cuda_max_memory_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        bench["cuda_max_memory_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    evidence["benchmark"] = bench
    assert all(v > 0 for v in forward)
