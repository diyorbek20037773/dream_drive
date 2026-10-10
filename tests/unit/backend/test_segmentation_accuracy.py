"""Drivable-area accuracy plumbing (app/evaluation/segmentation_accuracy.py): KITTI GT colour decoding, ignore pixels,
pixel confusion and ratios, dataset-level pooling, manifest determinism and the Evaluation artifact shape.

No network and no KITTI data: a tiny synthetic ``data_road.zip`` lookalike is built in ``tmp_path`` and the inference
service is an ``httpx.MockTransport``. The mocked service only exercises plumbing; real numbers come from the real run
documented in docs/EVALUATION_SEGMENTATION.md.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import zipfile
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from app.evaluation import segmentation_accuracy as sa
from app.evaluation.registry import list_evaluations, load_evaluation, load_evaluation_details
from app.perception.client import InferenceClient, InferenceClientError, ensemble_sha256
from app.schemas.api import Evaluation

W, H = 20, 10
RGB_ROAD = (255, 0, 255)
RGB_NON_ROAD = (255, 0, 0)
RGB_IGNORE = (0, 0, 0)
RGB_ROAD_OUTSIDE_VALID = (0, 0, 255)
YOLO_SHA = "a" * 64
RFDETR_SHA = "b" * 64


def _gt_rgb() -> np.ndarray:
    """Rows 0-4 non-road, rows 5-9 road; rows 8-9 x cols 0-4 don't care; pixel (9,19) road outside valid area.

    Valid pixels: 100 non-road + 89 road = 189; ignored: 10 black + 1 blue = 11.
    """
    gt = np.zeros((H, W, 3), dtype=np.uint8)
    gt[:5] = RGB_NON_ROAD
    gt[5:] = RGB_ROAD
    gt[8:, :5] = RGB_IGNORE
    gt[9, 19] = RGB_ROAD_OUTSIDE_VALID
    return gt


def _png_rgb(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


def _mask_b64(mask: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def _pred(kind: str) -> np.ndarray:
    pred = np.zeros((H, W), dtype=bool)
    if kind == "perfect":
        pred[5:] = True  # also covers the don't-care pixels: they must not count
    elif kind == "partial":
        pred[3:, :10] = True
    elif kind != "empty":
        raise AssertionError(kind)
    return pred


def _zip(tmp_path: Path, name: str = "data_road.zip") -> Path:
    path = tmp_path / name
    rng = np.random.default_rng(0)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("data_road/", b"")
        zf.writestr(sa.IMAGE_DIR, b"")
        zf.writestr(sa.GT_DIR, b"")
        for stem in ("uu_000000", "um_000000", "umm_000000"):  # written out of order on purpose
            cat, idx = stem.split("_")
            zf.writestr(f"{sa.IMAGE_DIR}{stem}.png", _png_rgb(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)))
            zf.writestr(f"{sa.GT_DIR}{cat}_road_{idx}.png", _png_rgb(_gt_rgb()))
        lane = np.zeros((H, W, 3), dtype=np.uint8)
        lane[:, :] = RGB_NON_ROAD
        zf.writestr(f"{sa.GT_DIR}um_lane_000000.png", _png_rgb(lane))
        zf.writestr("data_road/training/calib/um_000000.txt", b"P0: 0\n")
    return path


# ---------------------------------------------------------------------------------------------------------------------
# GT decoding and pixel counting
# ---------------------------------------------------------------------------------------------------------------------
def test_gt_decoding_follows_devkit_planes() -> None:
    road, valid = sa.decode_gt_png(_png_rgb(_gt_rgb()))
    assert road.shape == valid.shape == (H, W)
    assert int(valid.sum()) == 189 and int((road & valid).sum()) == 89
    assert not valid[9, 19] and road[9, 19]  # blue only: road but outside the valid area
    assert not valid[8, 0] and not road[8, 0]
    hist = sa.gt_colour_histogram(_png_rgb(_gt_rgb()))
    assert hist == {"#000000": 10, "#0000ff": 1, "#ff0000": 100, "#ff00ff": 89}
    assert sa._gt_counts(hist) == {"road_px": 89, "non_road_px": 100, "ignore_px": 11, "road_outside_valid_px": 1}


def test_gt_rejects_grey_or_invalid_png() -> None:
    ok, buf = cv2.imencode(".png", np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="8-bit colour"):
        sa.decode_gt_png(buf.tobytes())
    with pytest.raises(ValueError, match="decoded"):
        sa.decode_gt_png(b"not a png")


def test_perfect_prediction_ignores_dont_care_pixels() -> None:
    road, valid = sa.decode_gt_png(_png_rgb(_gt_rgb()))
    c = sa.confusion(_pred("perfect"), road, valid)
    assert (c.tp, c.fp, c.fn, c.tn) == (89, 0, 0, 100)
    assert c.iou == c.precision == c.recall == c.f1 == 1.0
    assert c.valid == 189


def test_empty_prediction() -> None:
    road, valid = sa.decode_gt_png(_png_rgb(_gt_rgb()))
    c = sa.confusion(_pred("empty"), road, valid)
    assert (c.tp, c.fp, c.fn, c.tn) == (0, 0, 89, 100)
    assert c.iou == 0.0 and c.recall == 0.0 and c.f1 == 0.0
    assert c.precision is None  # undefined, never reported as 0 or 1


def test_partial_prediction_exact_ratios() -> None:
    road, valid = sa.decode_gt_png(_png_rgb(_gt_rgb()))
    c = sa.confusion(_pred("partial"), road, valid)
    # rows 3-4 x cols 0-9 = 20 FP; rows 5-9 x cols 0-9 = 50 minus 10 don't care = 40 TP; 89 - 40 = 49 FN
    assert (c.tp, c.fp, c.fn) == (40, 20, 49)
    assert c.iou == pytest.approx(40 / 109)
    assert c.precision == pytest.approx(40 / 60)
    assert c.recall == pytest.approx(40 / 89)
    assert c.f1 == pytest.approx(80 / 149)


def test_empty_union_is_undefined_and_shapes_must_match() -> None:
    nothing = np.zeros((H, W), dtype=bool)
    c = sa.confusion(nothing, nothing, np.ones((H, W), dtype=bool))
    assert c.union == 0 and c.iou is None and c.f1 is None and c.tn == H * W
    with pytest.raises(ValueError, match="shape mismatch"):
        sa.confusion(np.zeros((H, W + 1), dtype=bool), nothing, nothing)


def test_dataset_level_pooling_differs_from_mean_of_images() -> None:
    road, valid = sa.decode_gt_png(_png_rgb(_gt_rgb()))
    a = sa.confusion(_pred("perfect"), road, valid)
    b = sa.confusion(_pred("partial"), road, valid)
    pooled = a + b
    assert (pooled.tp, pooled.fp, pooled.fn) == (129, 20, 49)
    assert pooled.iou == pytest.approx(129 / 198)
    assert pooled.iou != pytest.approx((1.0 + 40 / 109) / 2)


def test_pred_mask_decoding_is_strict() -> None:
    mask = _pred("partial")
    decoded, non_binary = sa.decode_pred_mask(_mask_b64(mask), W, H)
    assert np.array_equal(decoded, mask) and non_binary == 0
    with pytest.raises(ValueError, match="expected"):
        sa.decode_pred_mask(_mask_b64(mask), W + 1, H)
    with pytest.raises(ValueError, match="missing"):
        sa.decode_pred_mask(None, W, H)
    with pytest.raises(ValueError, match="base64"):
        sa.decode_pred_mask("@@not-base64@@", W, H)
    grey = np.full((H, W), 100, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", grey)
    decoded, non_binary = sa.decode_pred_mask(base64.b64encode(buf.tobytes()).decode(), W, H)
    assert non_binary == H * W and not decoded.any()


def test_metric_rules_for_missing_scopes() -> None:
    m = sa.build_metrics({"overall": sa.PixelCounts(0, 0, 5, 5)}, {"overall": 1}, [0.0], 0, [], [], [], "cpu")
    assert m["drivable_precision_overall"].status == "not_evaluated"
    assert m["drivable_precision_overall"].value is None and "undefined" in m["drivable_precision_overall"].note
    assert m["drivable_iou_overall"].value == 0.0 and m["drivable_iou_overall"].sample_count == 1
    assert m["drivable_iou_um"].status == "not_evaluated" and m["drivable_iou_um"].sample_count == 0
    assert m["kitti_bev_maxf"].status == "not_evaluated"
    assert m["ego_lane_iou"].status == "unsupported"
    assert "inference_latency_p50_ms" not in m


# ---------------------------------------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------------------------------------
def test_manifest_is_deterministic_and_hashes_members(tmp_path: Path) -> None:
    zpath = _zip(tmp_path)
    out1, out2 = tmp_path / "m1.json", tmp_path / "m2.json"
    m = sa.build_manifest(zpath, out1, expected_archive_sha256=None)
    sa.build_manifest(zpath, out2, expected_archive_sha256=None)
    assert out1.read_bytes() == out2.read_bytes()
    assert out1.read_bytes() == sa.manifest_bytes(m)
    assert [e["id"] for e in m["images"]] == ["um_000000", "umm_000000", "uu_000000"]
    assert m["counts"]["by_category"] == {"um": 1, "umm": 1, "uu": 1}
    assert m["counts"]["gt_lane_files_not_evaluated"] == 1
    assert m["source"]["archive_sha256"] == hashlib.sha256(zpath.read_bytes()).hexdigest()
    assert m["source"]["licence_id"] == "CC-BY-NC-SA-3.0"
    with zipfile.ZipFile(zpath) as zf:
        for e in m["images"]:
            assert e["image"]["sha256"] == hashlib.sha256(zf.read(e["image"]["member"])).hexdigest()
            assert e["gt_road"]["sha256"] == hashlib.sha256(zf.read(e["gt_road"]["member"])).hexdigest()
            assert (e["width"], e["height"]) == (W, H)
            assert e["gt_road"]["road_px"] == 89 and e["gt_road"]["ignore_px"] == 11
    assert m["images"][0]["gt_lane_not_evaluated"]["member"] == f"{sa.GT_DIR}um_lane_000000.png"
    assert m["images"][1]["gt_lane_not_evaluated"] is None
    assert m["gt_colour_totals_rgb"]["all"] == {"#000000": 30, "#0000ff": 3, "#ff0000": 300, "#ff00ff": 267}
    text = out1.read_text(encoding="utf-8")
    assert "created" not in text and "utc" not in text.lower()  # no timestamps -> reproducible bytes


def test_manifest_rejects_wrong_archive_hash_and_missing_gt(tmp_path: Path) -> None:
    zpath = _zip(tmp_path)
    with pytest.raises(ValueError, match="expected"):
        sa.build_manifest(zpath, expected_archive_sha256="0" * 64)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr(f"{sa.IMAGE_DIR}um_000001.png", _png_rgb(np.zeros((H, W, 3), dtype=np.uint8)))
    with pytest.raises(ValueError, match="missing road ground truth"):
        sa.build_manifest(bad, expected_archive_sha256=None)


# ---------------------------------------------------------------------------------------------------------------------
# End-to-end with a mocked inference service
# ---------------------------------------------------------------------------------------------------------------------
PRED_BY_ID = {"um_000000": "perfect", "umm_000000": "empty", "uu_000000": "partial"}
ORDER = ["um_000000", "umm_000000", "uu_000000"]


def _fake_service(fail_all: bool = False) -> httpx.MockTransport:
    models = [
        {"model_id": "yolopv2", "sha256": YOLO_SHA, "device": "cpu", "precision": "fp32"},
        {"model_id": "rfdetr_nano", "sha256": RFDETR_SHA, "device": "cpu", "precision": "fp32"},
    ]
    ens = ensemble_sha256([("yolopv2", YOLO_SHA), ("rfdetr_nano", RFDETR_SHA)])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health/ready":
            return httpx.Response(200, json={"ready": True})
        if request.url.path == "/v1/model":
            return httpx.Response(
                200,
                json={
                    "model_id": "yolopv2",
                    "ready": True,
                    "sha256": YOLO_SHA,
                    "device": "cpu",
                    "precision": "fp32",
                    "ensemble_id": "yolopv2+rfdetr_nano",
                    "ensemble_sha256": ens,
                    "models": [
                        {"model_id": "yolopv2", "ready": True, "sha256": YOLO_SHA},
                        {"model_id": "rfdetr_nano", "ready": True, "sha256": RFDETR_SHA},
                    ],
                },
            )
        if fail_all:
            return httpx.Response(400, json={"code": "image_invalid", "message": "nope"})
        # The service must receive the original-size image as JPEG.
        assert request.headers["content-type"] == "image/jpeg"
        img = cv2.imdecode(np.frombuffer(request.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[:2] == (H, W)
        frame_id = int(request.url.params["frame_id"])
        mask = _pred(PRED_BY_ID[ORDER[frame_id]])
        return httpx.Response(
            200,
            json={
                "schema_version": "1.0",
                "frame_id": frame_id,
                "model": models[0],
                "models": models,
                "ensemble_id": "yolopv2+rfdetr_nano",
                "ensemble_sha256": ens,
                "image": {"width": W, "height": H},
                "geometry": {
                    "canvas_width": 32,
                    "canvas_height": 32,
                    "scale": 1.0,
                    "pad_left": 6,
                    "pad_top": 11,
                    "pad_right": 6,
                    "pad_bottom": 11,
                },
                "detections": [],
                "drivable_mask_png": _mask_b64(mask),
                "lane_mask_png": _mask_b64(np.zeros((H, W), dtype=bool)),
                "mask_stats": {"drivable_fraction": float(mask.mean()), "lane_fraction": 0.0},
                "timing_ms": {
                    "decode": 1,
                    "preprocess": 1,
                    "inference": 5,
                    "postprocess": 1,
                    "total": 10.0 + frame_id,
                    "inference_by_model": {"yolopv2": 4.0 + frame_id, "rfdetr_nano": 1.0},
                },
            },
        )

    return httpx.MockTransport(handler)


def test_run_writes_registry_loadable_evaluation(tmp_path: Path) -> None:
    zpath = _zip(tmp_path)
    manifest = tmp_path / "manifest.json"
    sa.build_manifest(zpath, manifest, expected_archive_sha256=None)
    out_dir, evidence = tmp_path / "evaluations", tmp_path / "evidence"
    client = InferenceClient("http://inference.test", 5.0, transport=_fake_service())
    progress: list[tuple[int, int, str]] = []
    doc = sa.run_drivable_accuracy(
        zpath,
        manifest,
        client,
        out_dir,
        "c" * 40,
        lambda d, t, label: progress.append((d, t, label)),
        evidence_dir=evidence,
        run_label="unit_test_mock",
    )
    assert progress[-1] == (3, 3, "uu_000000")
    assert doc["kind"] == "video_ml_accuracy" and doc["source"] == "learned_model" and doc["sample_count"] == 3
    assert doc["model_sha256"] == YOLO_SHA  # the drivable mask comes from YOLOPv2 only
    det = doc["details"]
    assert det["ensemble_sha256_reported"] == ensemble_sha256([("yolopv2", YOLO_SHA), ("rfdetr_nano", RFDETR_SHA)])
    assert det["dataset"]["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert det["dataset"]["archive_sha256"] == hashlib.sha256(zpath.read_bytes()).hexdigest()
    m = doc["metrics"]
    # pooled: TP 89+0+40, FP 0+0+20, FN 0+89+49
    assert m["drivable_iou_overall"]["value"] == round(129 / 287, 4)
    assert m["drivable_precision_overall"]["value"] == round(129 / 149, 4)
    assert m["drivable_recall_overall"]["value"] == round(129 / 267, 4)
    assert m["drivable_f1_overall"]["value"] == round(258 / 416, 4)
    assert m["drivable_iou_overall"]["sample_count"] == 3 and m["drivable_iou_overall"]["status"] == "measured"
    assert m["drivable_iou_um"]["value"] == 1.0 and m["drivable_iou_um"]["sample_count"] == 1
    assert m["drivable_iou_umm"]["value"] == 0.0
    assert m["drivable_precision_umm"]["status"] == "not_evaluated"  # undefined for an empty prediction
    assert m["drivable_iou_uu"]["value"] == round(40 / 109, 4)
    assert m["drivable_iou_per_image_mean"]["value"] == round((1.0 + 0.0 + 40 / 109) / 3, 4)
    assert m["drivable_iou_per_image_p50"]["value"] == round(40 / 109, 4)
    assert m["inference_latency_p50_ms"]["value"] == 11.0
    assert m["yolopv2_forward_p50_ms"]["value"] == 5.0
    assert m["lane_marking_accuracy"]["status"] == "not_evaluated"
    assert any("BDD100K" in lim for lim in doc["limitations"])
    assert any("unit_test_mock" in lim for lim in doc["limitations"])

    # Registry round trip + API schema validation (details stripped by the registry).
    listed = list_evaluations(out_dir)
    assert [e["id"] for e in listed] == [doc["id"]]
    Evaluation.model_validate(load_evaluation(out_dir, doc["id"]))
    assert load_evaluation_details(out_dir, doc["id"])["totals"]["overall"]["tp"] == 129

    # Evidence: per-image CSV, prediction masks, three visuals.
    with (evidence / f"{doc['id']}.per_image.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["id"] for r in rows] == ORDER
    assert [int(r["fp"]) for r in rows] == [0, 0, 20]
    masks = sorted(p.name for p in (evidence / f"{doc['id']}_pred_masks").iterdir())
    assert masks == [f"{i}.png" for i in ORDER]
    vis = det["evidence"]["visuals"]
    assert {k: v["id"] for k, v in vis.items()} == {"worst": "umm_000000", "median": "uu_000000", "best": "um_000000"}
    for v in vis.values():
        img = cv2.imread(str(evidence / v["file"]))
        assert img is not None and img.shape[1] == W and img.shape[0] == 3 * (H + 26)
    json.dumps(doc)  # serialisable


def test_run_rejects_archive_hash_mismatch(tmp_path: Path) -> None:
    zpath = _zip(tmp_path)
    manifest = tmp_path / "manifest.json"
    sa.build_manifest(zpath, manifest, expected_archive_sha256=None)
    (tmp_path / "x").mkdir()
    other = _zip(tmp_path / "x")
    with zipfile.ZipFile(other, "a") as zf:
        zf.writestr("extra.txt", b"changed")
    client = InferenceClient("http://inference.test", 5.0, transport=_fake_service())
    with pytest.raises(ValueError, match="archive_sha256"):
        sa.run_drivable_accuracy(other, manifest, client, tmp_path / "out", "c" * 40)


def test_run_refuses_without_ready_service_or_with_failures(tmp_path: Path) -> None:
    zpath = _zip(tmp_path)
    manifest = tmp_path / "manifest.json"
    sa.build_manifest(zpath, manifest, expected_archive_sha256=None)
    down = InferenceClient(
        "http://inference.test", 5.0, transport=httpx.MockTransport(lambda r: httpx.Response(503, json={}))
    )
    with pytest.raises(InferenceClientError):
        sa.run_drivable_accuracy(zpath, manifest, down, tmp_path / "out", "c" * 40)
    failing = InferenceClient("http://inference.test", 5.0, transport=_fake_service(fail_all=True))
    with pytest.raises(RuntimeError, match="no image could be evaluated"):
        sa.run_drivable_accuracy(zpath, manifest, failing, tmp_path / "out", "c" * 40, visuals=False)
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").glob("*.json"))


def test_committed_manifest_matches_the_documented_archive() -> None:
    path = Path(__file__).resolve().parents[3] / "datasets" / "manifests" / "kitti_road_training_v1.json"
    m = json.loads(path.read_text(encoding="utf-8"))
    assert m["manifest_id"] == sa.MANIFEST_ID
    assert m["source"]["archive_sha256"] == sa.ARCHIVE_SHA256
    assert m["source"]["archive_size_bytes"] == sa.ARCHIVE_SIZE_BYTES
    assert m["counts"]["images"] == 289 and len(m["images"]) == 289
    assert m["counts"]["by_category"] == {"um": 95, "umm": 96, "uu": 98}
    assert set(m["gt_colour_totals_rgb"]["all"]) <= {"#000000", "#0000ff", "#ff0000", "#ff00ff"}
    assert path.read_bytes() == sa.manifest_bytes(m)  # canonical serialisation
