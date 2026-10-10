"""Detector-accuracy plumbing (app/evaluation/detector_accuracy.py): subset rule, determinism, class mapping, metric
status rules and the Evaluation artifact shape. No network: the inference service is an ``httpx.MockTransport`` and
images come from a pre-seeded cache. The mocked service is for plumbing only; real numbers come from the real run.
"""

from __future__ import annotations

import base64
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import pytest

from app.evaluation import detector_accuracy as da
from app.evaluation.coco_ap import ClassResult
from app.evaluation.registry import list_evaluations, load_evaluation, load_evaluation_details
from app.perception.client import InferenceClient
from app.schemas.api import Evaluation

CATS = [
    (1, "person"),
    (2, "bicycle"),
    (3, "car"),
    (4, "motorcycle"),
    (6, "bus"),
    (8, "truck"),
    (10, "traffic light"),
    (13, "stop sign"),
    (18, "dog"),
]


def _coco() -> dict[str, Any]:
    images = [
        {
            "id": i,
            "file_name": f"{i:012d}.jpg",
            "coco_url": f"http://images.cocodataset.org/val2017/{i:012d}.jpg",
            "flickr_url": f"http://farm.example/{i}.jpg",
            "width": 64,
            "height": 48,
            "license": 1 if i == 7 else 2,
        }
        for i in (5, 3, 9, 7, 11, 13)
    ]
    anns = [
        # image 3: person + car -> qualifies
        {"id": 1, "image_id": 3, "category_id": 1, "bbox": [2, 2, 10, 20], "area": 150, "iscrowd": 0},
        {"id": 2, "image_id": 3, "category_id": 3, "bbox": [30, 20, 20, 15], "area": 250, "iscrowd": 0},
        # image 5: crowd person only + car -> excluded (crowd never qualifies)
        {"id": 3, "image_id": 5, "category_id": 1, "bbox": [0, 0, 30, 30], "area": 600, "iscrowd": 1},
        {"id": 4, "image_id": 5, "category_id": 3, "bbox": [30, 20, 20, 15], "area": 250, "iscrowd": 0},
        # image 7: traffic light + bus (+ a crowd person region) -> qualifies
        {"id": 5, "image_id": 7, "category_id": 10, "bbox": [5, 5, 4, 10], "area": 30, "iscrowd": 0},
        {"id": 6, "image_id": 7, "category_id": 6, "bbox": [10, 10, 40, 30], "area": 1000, "iscrowd": 0},
        {"id": 7, "image_id": 7, "category_id": 1, "bbox": [0, 30, 20, 18], "area": 300, "iscrowd": 1},
        # image 9: traffic light only -> excluded (no vehicle)
        {"id": 8, "image_id": 9, "category_id": 10, "bbox": [5, 5, 4, 10], "area": 30, "iscrowd": 0},
        # image 11: stop sign + truck + bicycle -> qualifies
        {"id": 9, "image_id": 11, "category_id": 13, "bbox": [1, 1, 8, 8], "area": 50, "iscrowd": 0},
        {"id": 10, "image_id": 11, "category_id": 8, "bbox": [20, 10, 40, 30], "area": 1000, "iscrowd": 0},
        {"id": 11, "image_id": 11, "category_id": 2, "bbox": [10, 30, 10, 10], "area": 60, "iscrowd": 0},
        # image 13: dog + car -> excluded (no signal category)
        {"id": 12, "image_id": 13, "category_id": 18, "bbox": [1, 1, 8, 8], "area": 50, "iscrowd": 0},
        {"id": 13, "image_id": 13, "category_id": 3, "bbox": [20, 10, 40, 30], "area": 1000, "iscrowd": 0},
    ]
    return {
        "images": images,
        "annotations": anns,
        "categories": [{"id": i, "name": n} for i, n in CATS],
        "licenses": [
            {"id": 1, "name": "Attribution License", "url": "http://creativecommons.org/licenses/by/2.0/"},
            {
                "id": 2,
                "name": "Attribution-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-sa/2.0/",
            },
        ],
    }


def _zip(tmp_path: Path, coco: dict[str, Any]) -> Path:
    path = tmp_path / "annotations_trainval2017.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(da.ANNOTATION_MEMBER, json.dumps(coco))
    return path


def test_subset_rule_crowd_and_order() -> None:
    body = da.select_street_subset(_coco(), limit=300)
    assert [im["id"] for im in body["images"]] == [3, 7, 11]
    assert body["candidate_count"] == 3 and body["image_count"] == 3
    counts = body["class_instance_counts"]
    assert counts["person"] == {"non_crowd": 1, "crowd": 1, "images": 1}
    assert counts["traffic light"]["non_crowd"] == 1 and counts["stop sign"]["non_crowd"] == 1
    assert counts["car"]["non_crowd"] == 1 and counts["bus"]["non_crowd"] == 1 and counts["truck"]["non_crowd"] == 1
    first = body["images"][0]
    assert first["license_id"] == 2 and first["license_name"] == "Attribution-ShareAlike License"
    assert first["license_url"].startswith("http://creativecommons.org/") and first["flickr_url"]
    assert {lic["id"]: lic["image_count"] for lic in body["licenses"]} == {1: 1, 2: 2}


def test_subset_limit_takes_lowest_ids() -> None:
    assert [im["id"] for im in da.select_street_subset(_coco(), limit=2)["images"]] == [3, 7]
    with pytest.raises(ValueError):
        da.select_street_subset(_coco(), limit=0)


def test_manifest_is_deterministic_and_hash_checked(tmp_path: Path) -> None:
    zpath = _zip(tmp_path, _coco())
    zsha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    da.build_manifest(zpath, a, expected_archive_sha256=zsha)
    da.build_manifest(zpath, b, expected_archive_sha256=zsha)
    assert a.read_bytes() == b.read_bytes()
    doc = json.loads(a.read_text(encoding="utf-8"))
    assert doc["source"]["archive_sha256"] == zsha
    assert doc["source"]["member_sha256"] == hashlib.sha256(json.dumps(_coco()).encode()).hexdigest()
    with pytest.raises(ValueError, match="sha256 mismatch"):
        da.build_manifest(zpath, tmp_path / "c.json")  # default expects the real COCO archive hash


def test_committed_manifest_matches_rule() -> None:
    path = Path(__file__).resolve().parents[3] / "datasets" / "manifests" / "coco_val2017_street_v1.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["source"]["archive_sha256"] == da.ANNOTATION_ARCHIVE_SHA256
    ids = [im["id"] for im in doc["images"]]
    assert ids == sorted(ids) and len(ids) == len(set(ids)) == doc["image_count"] <= 300
    assert all(im["coco_url"].startswith("http://images.cocodataset.org/val2017/") for im in doc["images"])
    assert all(im["license_name"] and im["license_url"] for im in doc["images"])


def test_ground_truth_merges_vehicle_categories() -> None:
    gt = da.ground_truth_by_class(_coco(), [3, 7, 11])
    assert gt["vehicle"][3][0] == [[30.0, 20.0, 20.0, 15.0]]  # car
    assert gt["vehicle"][7][0] == [[10.0, 10.0, 40.0, 30.0]]  # bus
    assert gt["vehicle"][11][0] == [[20.0, 10.0, 40.0, 30.0]]  # truck
    assert gt["bus"][7][0] == [[10.0, 10.0, 40.0, 30.0]] and gt["bus"][3][0] == []
    assert gt["person"][7] == ([[0.0, 30.0, 20.0, 18.0]], [True])  # crowd kept as an ignore region
    assert gt["traffic_light"][7][0] == [[5.0, 5.0, 4.0, 10.0]]


def test_map_detections_by_class_name() -> None:
    dets = [
        {"class_name": "bus", "score": 0.8, "xyxy_pixels": [10, 10, 50, 40]},
        {"class_name": "vehicle", "score": 0.7, "xyxy_pixels": [0, 0, 10, 10]},
        {"class_name": "traffic_light", "score": 0.6, "xyxy_pixels": [1, 2, 3, 6]},
        {"class_name": "unverified_class_7", "score": 0.9, "xyxy_pixels": [0, 0, 1, 1]},
    ]
    m = da.map_detections(dets)
    assert m["bus"] == [([10.0, 10.0, 40.0, 30.0], 0.8)]
    assert [s for _, s in m["vehicle_any"]] == [0.8, 0.7]
    assert m["vehicle"] == [([0.0, 0.0, 10.0, 10.0], 0.7)]
    assert m["traffic_light"] == [([1.0, 2.0, 2.0, 4.0], 0.6)]
    assert m["person"] == [] and sum(len(v) for v in m.values()) == 5


def test_spec_class_names_and_coco_ids() -> None:
    # spec 2.1: DreamDrive id = 100 + COCO id; mapping is by class_name.
    expected = {
        "person": "person",
        "bicycle": "bicycle",
        "motorcycle": "motorcycle",
        "bus": "bus",
        "truck": "truck",
        "traffic_light": "traffic light",
        "stop_sign": "stop sign",
    }
    for name, coco_name in expected.items():
        ec = da.EVAL_CLASS_BY_NAME[name]
        assert ec.pred_class_names == (name,) and ec.gt_category_names == (coco_name,) and ec.in_enabled_map
    assert da.EVAL_CLASS_BY_NAME["vehicle"].gt_category_names == ("car", "bus", "truck")
    assert not da.EVAL_CLASS_BY_NAME["vehicle"].in_enabled_map


def test_supported_classes_and_ensemble_hash() -> None:
    single = {"model_id": "yolopv2", "class_map": {"3": "vehicle"}}
    names = da.supported_class_names(single)
    assert names == {"vehicle"}
    assert da.class_supported(da.EVAL_CLASS_BY_NAME["vehicle"], names)
    assert not da.class_supported(da.EVAL_CLASS_BY_NAME["vehicle_any"], names)
    assert not da.class_supported(da.EVAL_CLASS_BY_NAME["person"], names)
    two = {
        "class_map": {"3": "vehicle"},
        "models": [{"model_id": "rfdetr_nano", "class_map": {"101": "person", "110": "traffic_light"}}],
    }
    assert da.supported_class_names(two) == {"vehicle", "person", "traffic_light"}
    a, b = "a" * 64, "b" * 64
    expected = hashlib.sha256(f"yolopv2:{a}\nrfdetr_nano:{b}\n".encode()).hexdigest()
    models = [{"model_id": "rfdetr_nano", "sha256": b}, {"model_id": "yolopv2", "sha256": a}]
    assert da.ensemble_sha256(models) == expected
    assert da.ensemble_sha256([{"model_id": "yolopv2", "sha256": a}]) is None


def test_metrics_unsupported_is_not_zero() -> None:
    res = ClassResult(0.4, 0.6, 0.35, 0.7, 0.5, 10, 0, 4, 12)
    results = {ec.name: res for ec in da.EVAL_CLASSES}
    supported = {ec.name: ec.name == "vehicle" for ec in da.EVAL_CLASSES}
    m = da.build_metrics(results, supported, 5, [10.0, 20.0], [12.0, 25.0], "cpu")
    assert m["ap50_vehicle"].status == "measured" and m["ap50_vehicle"].value == 0.6
    assert m["ap50_vehicle"].sample_count == 10
    assert m["ap50_person"].status == "unsupported" and m["ap50_person"].value is None
    assert m["map50_enabled_classes"].status == "unsupported"
    assert m["lane_accuracy"].status == "not_evaluated" and m["drivable_area_iou"].value is None
    all_on = da.build_metrics(results, dict.fromkeys(supported, True), 5, [10.0], [12.0], "cpu")
    assert all_on["map50_enabled_classes"].value == 0.6 and all_on["map50_enabled_classes"].sample_count == 70
    assert all_on["map50_95_enabled_classes"].value == 0.4


# ---------------------------------------------------------------------------------------------------------------------
# End-to-end plumbing with a mocked service (no network)
# ---------------------------------------------------------------------------------------------------------------------
def _seed_cache(cache: Path, coco: dict[str, Any], ids: list[int]) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    index = {}
    for im in coco["images"]:
        if im["id"] not in ids:
            continue
        img = np.full((im["height"], im["width"], 3), 90, dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        data = buf.tobytes()
        (cache / im["file_name"]).write_bytes(data)
        index[im["file_name"]] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    (cache / "cache_index.json").write_text(json.dumps({"images": index}), encoding="utf-8")


def _fake_service(coco: dict[str, Any], order: list[int]) -> httpx.MockTransport:
    """Returns GT boxes as detections: person/traffic_light/bus exact, vehicle for car; frame_id -> image id."""
    names = {1: "person", 3: "vehicle", 6: "bus", 10: "traffic_light"}
    png = base64.b64encode(b"\x89PNG").decode()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health/ready":
            return httpx.Response(200, json={"ready": True})
        frame_id = int(request.url.params["frame_id"])
        img_id = order[frame_id]
        dets = [
            {
                "class_id": 100 + a["category_id"] if a["category_id"] != 3 else 3,
                "class_name": names[a["category_id"]],
                "score": 0.9,
                "xyxy_pixels": [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]],
            }
            for a in coco["annotations"]
            if a["image_id"] == img_id and a["category_id"] in names and not a["iscrowd"]
        ]
        sha = "f" * 64
        return httpx.Response(
            200,
            json={
                "schema_version": "1.0",
                "frame_id": frame_id,
                "model": {"model_id": "yolopv2", "sha256": sha, "device": "cpu", "precision": "fp32"},
                "image": {"width": 64, "height": 48},
                "geometry": {
                    "canvas_width": 64,
                    "canvas_height": 64,
                    "scale": 1.0,
                    "pad_left": 0,
                    "pad_top": 8,
                    "pad_right": 0,
                    "pad_bottom": 8,
                },
                "detections": dets,
                "drivable_mask_png": png,
                "lane_mask_png": png,
                "mask_stats": {"drivable_fraction": 0.1, "lane_fraction": 0.01},
                "timing_ms": {"decode": 1, "preprocess": 1, "inference": 5, "postprocess": 1, "total": 8.0 + frame_id},
            },
        )

    return httpx.MockTransport(handler)


def test_run_writes_registry_loadable_evaluation(tmp_path: Path) -> None:
    coco = _coco()
    zpath = _zip(tmp_path, coco)
    manifest = tmp_path / "manifest.json"
    da.build_manifest(zpath, manifest, expected_archive_sha256=hashlib.sha256(zpath.read_bytes()).hexdigest())
    ids = [3, 7, 11]
    _seed_cache(tmp_path / "cache", coco, ids)
    out_dir, evidence = tmp_path / "evaluations", tmp_path / "evidence"
    info = {
        "model_id": "yolopv2",
        "ready": True,
        "class_map": {"3": "vehicle"},
        "models": [{"model_id": "rfdetr_nano", "class_map": {"101": "person", "106": "bus", "110": "traffic_light"}}],
    }
    client = InferenceClient("http://inference.test", 5.0, transport=_fake_service(coco, ids))
    progress: list[tuple[int, int]] = []
    doc = da.run_detector_accuracy(
        manifest,
        zpath,
        tmp_path / "cache",
        client,
        out_dir,
        "a" * 40,
        lambda d, t, _l: progress.append((d, t)),
        evidence_dir=evidence,
        run_label="unit_test_mock",
        model_info=info,
    )
    assert progress[-1] == (3, 3)
    assert doc["kind"] == "video_ml_accuracy" and doc["source"] == "learned_model" and doc["sample_count"] == 3
    m = doc["metrics"]
    assert m["ap50_person"] == {
        "value": 1.0,
        "unit": "ratio",
        "sample_count": 1,
        "status": "measured",
        "note": m["ap50_person"]["note"],
    }
    assert m["ap50_traffic_light"]["value"] == 1.0 and m["ap50_bus"]["value"] == 1.0
    # vehicle GT = car+bus+truck; the mock reports only the car as 'vehicle' -> recall 1/3.
    assert m["ap50_vehicle"]["sample_count"] == 3 and m["recall50_vehicle"]["value"] == pytest.approx(0.3333, abs=1e-4)
    assert m["ap50_stop_sign"]["status"] == "unsupported" and m["ap50_stop_sign"]["value"] is None
    assert m["map50_enabled_classes"]["status"] == "unsupported"
    assert m["inference_latency_p50_ms"]["value"] == 9.0 and m["inference_latency_p50_ms"]["sample_count"] == 3
    assert doc["code_commit"] == "a" * 40 and doc["model_sha256"] == "f" * 64
    assert any("unit_test_mock" in lim for lim in doc["limitations"])

    listed = list_evaluations(out_dir)
    assert [e["id"] for e in listed] == [doc["id"]] and "details" not in listed[0]
    Evaluation.model_validate(load_evaluation(out_dir, doc["id"]))
    details = load_evaluation_details(out_dir, doc["id"])
    assert details["dataset"]["images_evaluated"] == 3 and details["protocol"]["jpeg_quality"] == 95
    preds = json.loads((evidence / f"{doc['id']}.predictions.json").read_text(encoding="utf-8"))
    assert preds["image_ids"] == ids and preds["numpy_results"]["person"]["ap50"] == pytest.approx(1.0)
    assert not list(out_dir.glob("*.predictions.json"))


def test_run_refuses_tampered_cache(tmp_path: Path) -> None:
    coco = _coco()
    zpath = _zip(tmp_path, coco)
    manifest = tmp_path / "manifest.json"
    da.build_manifest(zpath, manifest, expected_archive_sha256=hashlib.sha256(zpath.read_bytes()).hexdigest())
    ids = [3, 7, 11]
    _seed_cache(tmp_path / "cache", coco, ids)
    for i in ids:  # every cached file modified: re-download is attempted, which the mock transport cannot serve
        (tmp_path / "cache" / f"{i:012d}.jpg").write_bytes(b"tampered")
    cache = da.ImageCache(tmp_path / "cache", timeout_s=1.0, retries=1)
    cache._http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"other")))
    with pytest.raises(ValueError, match="pinned"):
        cache.get(
            {"file_name": "000000000003.jpg", "coco_url": "http://images.cocodataset.org/val2017/000000000003.jpg"}
        )
    with pytest.raises(ValueError, match="non-COCO"):
        cache.get({"file_name": "000000000099.jpg", "coco_url": "http://evil.example/x.jpg"})
