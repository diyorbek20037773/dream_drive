"""Manifest verification, readiness and HTTP API behaviour (no real weights needed).

(c) missing file / hash mismatch / unloadable bytes -> not ready, /v1/infer 503, never any predictions;
(d) request validation: bad bytes 400, too large 413, bad parameters 400, busy 503.
The "ready" cases use a tiny synthetic TorchScript test module (conftest.SyntheticYolopLike), never real weights.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from conftest import (
    REPO_MANIFEST,
    SPEC_RFDETR_CLASS_MAP,
    SYNTHETIC_BOX_CANVAS,
    rfdetr_entry,
    synthetic_rfdetr_builder,
    write_manifest,
)
from fastapi.testclient import TestClient

from app.ensemble import ensemble_sha256
from app.main import create_app
from app.manifest import ManifestError, load_manifest, verify_model_file
from app.model import YolopV2Engine
from app.postprocess import decode_mask_png
from app.preprocess import compute_geometry

REAL_SHA = "f2a8c8374203ae3e67ff9c184e931f763957de92a993b23269e4e721627f1f8c"
RF_SHA = "d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d"


def _sample(text: str, name: str, **labels: str) -> float | None:
    """Value of one Prometheus sample (label order independent)."""
    from prometheus_client.parser import text_string_to_metric_families

    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name == name and all(sample.labels.get(k) == v for k, v in labels.items()):
                return float(sample.value)
    return None


def _jpeg(width: int = 1280, height: int = 720) -> bytes:
    img = np.full((height, width, 3), 90, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _not_ready_client(settings: Any) -> tuple[TestClient, YolopV2Engine]:
    engine = YolopV2Engine(settings)
    client = TestClient(create_app(settings, engine))
    return client, engine


# ----------------------------------------------------------------------------------------------- manifest
def test_repo_manifest_is_resolved_and_valid() -> None:
    m = load_manifest(REPO_MANIFEST, "yolopv2")
    assert m.sha256 == REAL_SHA
    assert m.size_bytes == 156380200
    assert m.file_name == "yolopv2.pt"
    assert m.repository_revision == "9388c6aa55f3bf5a878679a1338bb090cff6a256"
    assert m.class_map == {3: "vehicle"}
    assert m.checkpoint_url.startswith("https://github.com/CAIC-AD/YOLOPv2/releases/download/")


def test_repo_manifest_rfdetr_entry_matches_spec() -> None:
    """docs/IMPLEMENTATION_SPEC.md section 2.1 values, exactly."""
    m = load_manifest(REPO_MANIFEST, "rfdetr_nano")
    assert m.sha256 == RF_SHA
    assert m.size_bytes == 366287238
    assert m.checkpoint_md5 == "fb6504cce7fbdc783f7a46991f07639f"
    assert m.file_name == "rf-detr-nano.pth"
    assert m.repository == "https://github.com/roboflow/rf-detr"
    assert m.repository_revision == "7433d5919d0f7e72cc6cb78663e0a731afd499e0"
    assert m.checkpoint_url == "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth"
    assert m.class_map_wire() == SPEC_RFDETR_CLASS_MAP
    assert m.class_id_offset == 100 and m.source_label_space == "coco2017_category_id"
    assert m.input_policy == "rfdetr_resize384_imagenet_norm"


def test_repo_manifest_ensemble_sha_is_derived_from_member_hashes() -> None:
    import json

    raw = json.loads(REPO_MANIFEST.read_text(encoding="utf-8"))
    expected = ensemble_sha256([("yolopv2", REAL_SHA), ("rfdetr_nano", RF_SHA)])
    assert raw["ensemble"]["ensemble_id"] == "yolopv2+rfdetr_nano"
    assert raw["ensemble"]["ensemble_sha256"] == expected


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"checkpoint_sha256": "abc"}, "checkpoint_sha256"),
        ({"checkpoint_sha256": None}, "checkpoint_sha256"),
        ({"checkpoint_size_bytes": 0}, "checkpoint_size_bytes"),
        ({"file_name": "../etc/passwd"}, "file_name"),
        ({"file_name": "sub/yolopv2.pt"}, "file_name"),
        ({"supported_class_map": None}, "supported_class_map"),
        ({"supported_class_map": {"car": "vehicle"}}, "supported_class_map"),
    ],
)
def test_manifest_validation_rejects_bad_entries(tmp_path: Path, mutation: dict[str, Any], message: str) -> None:
    import json

    raw = json.loads(REPO_MANIFEST.read_text(encoding="utf-8"))
    raw["models"][0].update(mutation)
    path = tmp_path / "m.yaml"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ManifestError, match=message):
        load_manifest(path, "yolopv2")


def test_manifest_missing_or_unknown_model(tmp_path: Path) -> None:
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "absent.yaml", "yolopv2")
    with pytest.raises(ManifestError):
        load_manifest(REPO_MANIFEST, "not_a_model")


def test_verify_model_file_outcomes(tmp_path: Path) -> None:
    payload = b"0123456789" * 100
    sha = hashlib.sha256(payload).hexdigest()
    manifest = load_manifest(
        write_manifest(tmp_path / "m.yaml", file_name="w.pt", sha256=sha, size=len(payload)), "yolopv2"
    )
    target = tmp_path / "w.pt"
    assert verify_model_file(target, manifest).reason == "model_missing"
    target.write_bytes(payload[:-1])
    assert verify_model_file(target, manifest).reason == "hash_mismatch"  # wrong size
    target.write_bytes(payload[:-1] + b"X")
    res = verify_model_file(target, manifest)
    assert res.reason == "hash_mismatch" and res.actual_sha256 != sha  # same size, different bytes
    target.write_bytes(payload)
    res = verify_model_file(target, manifest)
    assert res.ok and res.actual_sha256 == sha and res.actual_size == len(payload)


# ----------------------------------------------------------------------------------------------- (c) not ready
def _assert_not_ready_everywhere(client: TestClient, engine: YolopV2Engine, reason: str) -> None:
    assert engine.wait_settled(30)
    assert not engine.ready and engine.reason == reason
    ready = client.get("/health/ready")
    assert ready.status_code == 503
    assert ready.json()["ready"] is False and ready.json()["reason"] == reason
    assert client.get("/health/live").json() == {"status": "ok"}
    info = client.get("/v1/model")
    assert info.status_code == 200
    body = info.json()
    assert body["ready"] is False and body["reason"] == reason
    assert body["sha256"] is None and body["class_map"] == {}
    r = client.post("/v1/infer?frame_id=1", content=_jpeg(), headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 503
    assert r.json()["code"] == "model_not_ready"
    assert set(r.json()) == {"code", "message"}  # no detections/masks of any kind
    metrics = client.get("/metrics").text
    assert _sample(metrics, "dreamdrive_model_ready", model="yolopv2", device="cpu") == 0.0


def test_missing_weights_not_ready_and_503(make_settings: Callable[..., Any]) -> None:
    client, engine = _not_ready_client(make_settings())  # empty tmp MODEL_ROOT, real manifest
    with client:
        _assert_not_ready_everywhere(client, engine, "model_missing")


def test_hash_mismatch_not_ready_and_503(tmp_path: Path, make_settings: Callable[..., Any]) -> None:
    root = tmp_path / "models"
    root.mkdir()
    # Correct size, wrong bytes: must fail on SHA256, never be loaded.
    with (root / "yolopv2.pt").open("wb") as fh:
        fh.truncate(156380200)
    client, engine = _not_ready_client(make_settings(model_root=root))
    with client:
        _assert_not_ready_everywhere(client, engine, "hash_mismatch")
    assert "sha256" in engine.detail


def test_verified_but_unloadable_bytes_are_load_failed(tmp_path: Path, make_settings: Callable[..., Any]) -> None:
    root = tmp_path / "models"
    root.mkdir()
    payload = b"definitely not a torchscript archive" * 50
    (root / "yolopv2.pt").write_bytes(payload)
    manifest = write_manifest(
        tmp_path / "m.yaml", file_name="yolopv2.pt", sha256=hashlib.sha256(payload).hexdigest(), size=len(payload)
    )
    client, engine = _not_ready_client(make_settings(model_root=root, model_manifest_path=manifest))
    with client:
        _assert_not_ready_everywhere(client, engine, "load_failed")


def test_invalid_manifest_not_ready(tmp_path: Path, make_settings: Callable[..., Any]) -> None:
    bad = tmp_path / "m.yaml"
    bad.write_text("models: [", encoding="utf-8")
    client, engine = _not_ready_client(make_settings(model_manifest_path=bad))
    with client:
        _assert_not_ready_everywhere(client, engine, "manifest_invalid")


def test_fp16_on_cpu_is_device_unavailable(make_settings: Callable[..., Any]) -> None:
    client, engine = _not_ready_client(make_settings(inference_precision="fp16"))
    with client:
        _assert_not_ready_everywhere(client, engine, "device_unavailable")


def _yolo_only_ready_setup(
    tmp_path: Path, synthetic_checkpoint: tuple[Path, str, int], rf: dict[str, object]
) -> tuple[Path, Path]:
    """Model root with the verified synthetic YOLOPv2 file; the RF-DETR entry is whatever the test passes."""
    path, sha, size = synthetic_checkpoint
    root = tmp_path / "models"
    root.mkdir()
    (root / path.name).write_bytes(path.read_bytes())
    manifest = write_manifest(tmp_path / "m.yaml", file_name=path.name, sha256=sha, size=size, rfdetr=rf)
    return root, manifest


def test_rfdetr_missing_makes_service_not_ready(
    tmp_path: Path, synthetic_checkpoint: tuple[Path, str, int], make_settings: Callable[..., Any]
) -> None:
    """YOLOPv2 verified and loadable, RF-DETR absent: the service is NOT ready (no single-model fallback)."""
    root, manifest = _yolo_only_ready_setup(tmp_path, synthetic_checkpoint, rfdetr_entry())
    settings = make_settings(model_root=root, model_manifest_path=manifest)
    engine = YolopV2Engine(settings, rfdetr_builder=synthetic_rfdetr_builder)
    client = TestClient(create_app(settings, engine))
    with client:
        _assert_not_ready_everywhere(client, engine, "model_missing")
        info = client.get("/v1/model").json()
        members = {m["model_id"]: m for m in info["models"]}
        assert members["yolopv2"]["ready"] is True and members["yolopv2"]["reason"] is None
        assert members["rfdetr_nano"]["ready"] is False and members["rfdetr_nano"]["reason"] == "model_missing"
        assert members["rfdetr_nano"]["sha256"] is None and members["rfdetr_nano"]["class_map"] == {}
        assert info["ensemble_sha256"] is None
        ready = client.get("/health/ready").json()
        assert ready["models"]["rfdetr_nano"] == {"ready": False, "reason": "model_missing"}
        assert ready["detail"].startswith("rfdetr_nano:")
        metrics = client.get("/metrics").text
        assert _sample(metrics, "dreamdrive_model_ready", model="rfdetr_nano", device="cpu") == 0.0


def test_rfdetr_hash_mismatch_is_never_constructed(
    tmp_path: Path, synthetic_checkpoint: tuple[Path, str, int], make_settings: Callable[..., Any]
) -> None:
    payload = b"x" * 4096
    root, manifest = _yolo_only_ready_setup(
        tmp_path, synthetic_checkpoint, rfdetr_entry(sha256=hashlib.sha256(b"other").hexdigest(), size=len(payload))
    )
    (root / "rf-detr-nano.pth").write_bytes(payload)
    built: list[Path] = []

    def builder(path: Path, device: str, precision: str, md5: str | None) -> Any:
        built.append(path)
        return synthetic_rfdetr_builder(path, device, precision, md5)

    settings = make_settings(model_root=root, model_manifest_path=manifest)
    engine = YolopV2Engine(settings, rfdetr_builder=builder)
    client = TestClient(create_app(settings, engine))
    with client:
        _assert_not_ready_everywhere(client, engine, "hash_mismatch")
    assert built == []  # bytes that fail SHA256 are never handed to rfdetr


def test_rfdetr_publisher_md5_mismatch_is_hash_mismatch(
    tmp_path: Path, synthetic_checkpoint: tuple[Path, str, int], make_settings: Callable[..., Any]
) -> None:
    """SHA256 matches but the manifest's publisher MD5 does not: refused before construction."""
    payload = b"rf" * 1000
    root, manifest = _yolo_only_ready_setup(
        tmp_path,
        synthetic_checkpoint,
        rfdetr_entry(sha256=hashlib.sha256(payload).hexdigest(), size=len(payload), md5="0" * 32),
    )
    (root / "rf-detr-nano.pth").write_bytes(payload)
    built: list[str | None] = []

    def builder(path: Path, device: str, precision: str, md5: str | None) -> Any:
        built.append(md5)
        return synthetic_rfdetr_builder(path, device, precision, md5)

    settings = make_settings(model_root=root, model_manifest_path=manifest)
    engine = YolopV2Engine(settings, rfdetr_builder=builder)
    with TestClient(create_app(settings, engine)) as client:
        _assert_not_ready_everywhere(client, engine, "hash_mismatch")
    assert built == [] and "publisher md5" in engine.detail

    # With the correct MD5 the builder receives exactly the verified MD5.
    good = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    manifest = write_manifest(
        tmp_path / "m2.yaml",
        file_name=synthetic_checkpoint[0].name,
        sha256=synthetic_checkpoint[1],
        size=synthetic_checkpoint[2],
        rfdetr=rfdetr_entry(sha256=hashlib.sha256(payload).hexdigest(), size=len(payload), md5=good),
    )
    settings = make_settings(model_root=root, model_manifest_path=manifest)
    engine = YolopV2Engine(settings, rfdetr_builder=builder)
    assert engine.load(), engine.status()
    assert built == [good]


def test_rfdetr_load_error_is_load_failed(
    tmp_path: Path, synthetic_checkpoint: tuple[Path, str, int], make_settings: Callable[..., Any]
) -> None:
    payload = b"rf" * 1000
    root, manifest = _yolo_only_ready_setup(
        tmp_path, synthetic_checkpoint, rfdetr_entry(sha256=hashlib.sha256(payload).hexdigest(), size=len(payload))
    )
    (root / "rf-detr-nano.pth").write_bytes(payload)

    def broken(path: Path, device: str, precision: str, md5: str | None) -> Any:
        raise RuntimeError("cannot deserialise")

    settings = make_settings(model_root=root, model_manifest_path=manifest)
    engine = YolopV2Engine(settings, rfdetr_builder=broken)
    client = TestClient(create_app(settings, engine))
    with client:
        _assert_not_ready_everywhere(client, engine, "load_failed")


def test_rfdetr_class_map_must_match_rfdetr_coco_list(
    tmp_path: Path, synthetic_checkpoint: tuple[Path, str, int], make_settings: Callable[..., Any]
) -> None:
    """A manifest that names COCO category 10 anything but 'traffic light' is refused at load (no guessed maps)."""
    payload = b"rf" * 1000
    wrong = {**SPEC_RFDETR_CLASS_MAP, "110": "stop_sign"}
    root, manifest = _yolo_only_ready_setup(
        tmp_path,
        synthetic_checkpoint,
        rfdetr_entry(sha256=hashlib.sha256(payload).hexdigest(), size=len(payload), class_map=wrong),
    )
    (root / "rf-detr-nano.pth").write_bytes(payload)
    settings = make_settings(model_root=root, model_manifest_path=manifest)
    engine = YolopV2Engine(settings, rfdetr_builder=synthetic_rfdetr_builder)
    client = TestClient(create_app(settings, engine))
    with client:
        _assert_not_ready_everywhere(client, engine, "manifest_invalid")
    assert "traffic light" in engine.detail


def test_cuda_requested_without_cuda_is_device_unavailable(make_settings: Callable[..., Any]) -> None:
    import torch

    if torch.cuda.is_available():
        pytest.skip("this host has CUDA; the unavailable-device path cannot be exercised here")
    client, engine = _not_ready_client(make_settings(inference_device="cuda"))
    with client:
        assert engine.wait_settled(30)
        assert engine.reason == "device_unavailable"
        assert client.get("/health/ready").status_code == 503


# ----------------------------------------------------------------------------------------------- ready path
def test_ready_engine_full_response_shape(synthetic_ready_client: TestClient) -> None:
    client = synthetic_ready_client
    ready = client.get("/health/ready")
    assert ready.status_code == 200 and ready.json()["ready"] is True
    info = client.get("/v1/model").json()
    assert info["ready"] is True and info["class_map"] == {"3": "vehicle", **SPEC_RFDETR_CLASS_MAP}
    assert info["input_policy"] == "letterbox640_stride32" and info["device"] == "cpu"
    assert info["ensemble_id"] == "yolopv2+rfdetr_nano"
    members = {m["model_id"]: m for m in info["models"]}
    assert list(members) == ["yolopv2", "rfdetr_nano"]
    assert members["yolopv2"]["class_map"] == {"3": "vehicle"} and members["yolopv2"]["ready"] is True
    assert members["rfdetr_nano"]["class_map"] == SPEC_RFDETR_CLASS_MAP and members["rfdetr_nano"]["ready"] is True
    assert members["rfdetr_nano"]["input_policy"] == "rfdetr_resize384_imagenet_norm"
    expected_ensemble = ensemble_sha256(
        [("yolopv2", members["yolopv2"]["sha256"]), ("rfdetr_nano", members["rfdetr_nano"]["sha256"])]
    )
    assert info["ensemble_sha256"] == expected_ensemble

    # Portrait PNG: left half red (drivable in the synthetic module), green stripe rows 600..700 (lane).
    w, h = 720, 1280
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, : w // 2, 2] = 255  # red (BGR)
    img[600:700, :, 1] = 255  # green
    ok, png = cv2.imencode(".png", img)
    r = client.post(
        "/v1/infer?frame_id=42&conf_threshold=0.25", content=png.tobytes(), headers={"Content-Type": "image/png"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["schema_version"] == "1.0" and body["frame_id"] == 42
    assert body["image"] == {"width": w, "height": h}
    g = compute_geometry(w, h)
    assert body["geometry"] == g.to_wire()
    assert body["model"]["device"] == "cpu" and body["model"]["precision"] == "fp32"
    timing = dict(body["timing_ms"])
    by_model = timing.pop("inference_by_model")
    assert set(timing) == {"decode", "preprocess", "inference", "postprocess", "total"}
    assert all(v >= 0 for v in timing.values())
    assert set(by_model) == {"yolopv2", "rfdetr_nano"} and all(v >= 0 for v in by_model.values())
    assert timing["inference"] == pytest.approx(sum(by_model.values()), abs=0.01)

    assert body["ensemble_id"] == "yolopv2+rfdetr_nano" and body["ensemble_sha256"] == expected_ensemble
    assert [m["model_id"] for m in body["models"]] == ["yolopv2", "rfdetr_nano"]
    assert body["models"][0] == body["model"]

    # The synthetic YOLOPv2 detection (canvas xyxy 80,72,120,96) maps back to original pixels via the inverse
    # letterbox; the synthetic RF-DETR truck on exactly that box replaces it (class refinement).
    x1, y1, x2, y2 = SYNTHETIC_BOX_CANVAS
    expected = [(x1 - g.pad_left) * 2, (y1 - g.pad_top) * 2, (x2 - g.pad_left) * 2, (y2 - g.pad_top) * 2]
    dets = body["detections"]
    assert [d["class_id"] for d in dets] == [101, 110, 108]  # sorted by score: 0.90, 0.80, 0.70
    assert all(d["class_id"] != 3 for d in dets)  # the vehicle box was replaced
    truck = dets[2]
    assert truck["class_name"] == "truck" and truck["model_id"] == "rfdetr_nano"
    assert truck["refined_from_model"] == "yolopv2" and truck["score"] == pytest.approx(0.7)
    assert truck["xyxy_pixels"] == pytest.approx(expected, abs=0.01)
    person, light = dets[0], dets[1]
    assert person["class_name"] == "person" and person["model_id"] == "rfdetr_nano"
    assert person["xyxy_pixels"] == pytest.approx([0.1 * w, 0.2 * h, 0.2 * w, 0.6 * h], abs=0.01)
    assert light["class_name"] == "traffic_light" and light["refined_from_model"] is None
    # Disabled COCO classes (car 3, airplane 5), a box outside the frame and the 0.05 stop sign are not returned.
    low = client.post(
        "/v1/infer?frame_id=43&conf_threshold=0.01&include_masks=false",
        content=png.tobytes(),
        headers={"Content-Type": "image/png"},
    ).json()
    assert sorted(d["class_id"] for d in low["detections"]) == [101, 108, 110, 113]

    da = decode_mask_png(body["drivable_mask_png"])
    ll = decode_mask_png(body["lane_mask_png"])
    assert da.shape == ll.shape == (h, w)
    assert (da[:, : w // 2 - 2] == 255).all() and (da[:, w // 2 + 2 :] == 0).all()
    assert (ll[605:695] == 255).all() and (ll[:590] == 0).all() and (ll[710:] == 0).all()
    assert body["mask_stats"]["drivable_fraction"] == pytest.approx(0.5, abs=0.01)
    assert body["mask_stats"]["lane_fraction"] == pytest.approx(100 / 1280, abs=0.01)


def test_include_masks_false_omits_pngs_keeps_stats(synthetic_ready_client: TestClient) -> None:
    r = synthetic_ready_client.post(
        "/v1/infer?frame_id=0&include_masks=false", content=_jpeg(), headers={"Content-Type": "image/jpeg"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["drivable_mask_png"] is None and body["lane_mask_png"] is None
    assert 0.0 <= body["mask_stats"]["drivable_fraction"] <= 1.0


# ----------------------------------------------------------------------------------------------- (d) validation
@pytest.mark.parametrize(
    ("content", "ctype"),
    [
        (b"\x00\x01garbage", "image/jpeg"),
        (b"", "image/jpeg"),
        (b"\xff\xd8\xff\xe0" + b"\x00" * 100, "image/jpeg"),
        (b"plain text", "text/plain"),
    ],
)
def test_bad_bytes_are_400(synthetic_ready_client: TestClient, content: bytes, ctype: str) -> None:
    r = synthetic_ready_client.post("/v1/infer?frame_id=1", content=content, headers={"Content-Type": ctype})
    assert r.status_code == 400
    assert r.json()["code"] == "image_invalid"


def test_png_declared_as_jpeg_is_400(synthetic_ready_client: TestClient) -> None:
    ok, png = cv2.imencode(".png", np.zeros((64, 64, 3), np.uint8))
    r = synthetic_ready_client.post(
        "/v1/infer?frame_id=1", content=png.tobytes(), headers={"Content-Type": "image/jpeg"}
    )
    assert r.status_code == 400 and r.json()["code"] == "image_invalid"


def test_body_over_limit_is_413(
    synthetic_checkpoint: tuple[Path, str, int], synthetic_manifest: Path, make_settings: Callable[..., Any]
) -> None:
    settings = make_settings(
        model_root=synthetic_checkpoint[0].parent,
        model_manifest_path=synthetic_manifest,
        inference_max_image_bytes=4096,
    )
    engine = YolopV2Engine(settings, rfdetr_builder=synthetic_rfdetr_builder)
    with TestClient(create_app(settings, engine)) as client:
        assert engine.wait_settled(60) and engine.ready
        noise = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        big = cv2.imencode(".jpg", noise)[1].tobytes()
        assert len(big) > 4096
        r = client.post("/v1/infer?frame_id=1", content=big, headers={"Content-Type": "image/jpeg"})
        assert r.status_code == 413 and r.json()["code"] == "image_too_large"


def test_dimensions_over_limit_are_413(synthetic_ready_client: TestClient) -> None:
    ok, png = cv2.imencode(".png", np.zeros((16, 4097, 3), np.uint8))
    r = synthetic_ready_client.post(
        "/v1/infer?frame_id=1", content=png.tobytes(), headers={"Content-Type": "image/png"}
    )
    assert r.status_code == 413 and r.json()["code"] == "image_too_large"


@pytest.mark.parametrize(
    "query",
    [
        "",
        "?frame_id=-1",
        "?frame_id=abc",
        "?frame_id=1&conf_threshold=2",
        "?frame_id=1&iou_threshold=0",
        "?frame_id=1&include_masks=maybe",
    ],
)
def test_bad_parameters_are_400(synthetic_ready_client: TestClient, query: str) -> None:
    r = synthetic_ready_client.post(f"/v1/infer{query}", content=_jpeg(), headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 400
    assert r.json()["code"] == "request_invalid"


def test_unknown_route_uses_error_envelope(synthetic_ready_client: TestClient) -> None:
    r = synthetic_ready_client.get("/nope")
    assert r.status_code == 404 and r.json()["code"] == "not_found"


def test_busy_when_waiters_exceeded(synthetic_ready_client: TestClient) -> None:
    """max_waiters=0: while one forward pass holds the lock, a second request is refused with 503."""
    client = synthetic_ready_client
    engine: YolopV2Engine = client.app.state.engine  # type: ignore[attr-defined]
    results: dict[str, int] = {}
    engine._forward_lock.acquire()  # simulate a long-running forward pass
    try:
        t = threading.Thread(
            target=lambda: results.update(
                first=client.post(
                    "/v1/infer?frame_id=1", content=_jpeg(), headers={"Content-Type": "image/jpeg"}
                ).status_code
            )
        )
        t.start()
        deadline = time.monotonic() + 10
        while engine.in_flight < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert engine.in_flight == 1
        r = client.post("/v1/infer?frame_id=2", content=_jpeg(), headers={"Content-Type": "image/jpeg"})
        assert r.status_code == 503 and r.json()["code"] == "inference_busy"
        assert r.headers.get("retry-after") == "1"
    finally:
        engine._forward_lock.release()
    t.join(10)
    assert results["first"] == 200  # the waiter ran once the lock was released
    metrics = client.get("/metrics").text
    busy = _sample(
        metrics, "dreamdrive_inference_requests_total", model="yolopv2", device="cpu", outcome="inference_busy"
    )
    assert busy is not None and busy >= 1


def test_metrics_record_real_observations(synthetic_ready_client: TestClient) -> None:
    client = synthetic_ready_client
    assert (
        client.post("/v1/infer?frame_id=3", content=_jpeg(), headers={"Content-Type": "image/jpeg"}).status_code == 200
    )
    text = client.get("/metrics").text
    assert _sample(text, "dreamdrive_model_ready", model="yolopv2", device="cpu") == 1.0
    count = _sample(text, "dreamdrive_inference_duration_seconds_count", model="yolopv2", device="cpu", outcome="ok")
    assert count is not None and count >= 1
