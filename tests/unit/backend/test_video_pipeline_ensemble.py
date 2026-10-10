"""run_video_analysis against a FAKE two-model (YOLOPv2 + RF-DETR-N) inference client, and the HTTP client's
validation of the spec 2.1 ensemble fields. Unit level only: the fake returns fixed boxes; the traffic-light COLOUR is
NOT supplied by the fake - the pipeline must read it from the decoded pixels with the HSV heuristic. Real two-model
inference is exercised by the documented end-to-end run (artifacts/qa/spike), never by this fake.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import cv2
import httpx
import jsonschema
import numpy as np
import pytest
from test_video_synth import make_clip

from app.config import Settings
from app.perception.client import (
    InferenceClient,
    InferenceError,
    ensemble_sha256,
    merged_class_map,
    perception_identity,
)
from app.schemas.inference import InferenceResult, ModelInfo
from app.video import errors
from app.video.errors import VideoError
from app.video.pipeline import model_label, run_video_analysis

ROOT = Path(__file__).resolve().parents[3]
CONTRACTS = ROOT / "contracts"
YOLO_SHA = "f2a8c8374203ae3e67ff9c184e931f763957de92a993b23269e4e721627f1f8c"
RFDETR_SHA = "d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d"
ENS_SHA = ensemble_sha256([("yolopv2", YOLO_SHA), ("rfdetr_nano", RFDETR_SHA)])
COMMIT = "0123456789abcdef0123456789abcdef01234567"
SIZE = (640, 360)
LIGHT_BOX = [310.0, 40.0, 334.0, 106.0]  # upper centre
PERSON_BOX = [20.0, 150.0, 60.0, 300.0]  # sidewalk, outside the corridor
VEHICLE_BOX = [280.0, 215.0, 360.0, 265.0]  # in the corridor, not near
RFDETR_MAP = {"101": "person", "102": "bicycle", "104": "motorcycle", "106": "bus", "108": "truck"}
RFDETR_MAP |= {"110": "traffic_light", "113": "stop_sign"}
LIT = {"red": (255, 40, 40), "green": (40, 230, 130)}


def _png_b64(mask: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def draw_scene(index: int, _t: float, img: np.ndarray) -> None:
    """Signal head (green lit for samples 0-4, red lit afterwards) and a flat 'vehicle' patch."""
    x1, y1, x2, y2 = (int(v) for v in LIGHT_BOX)
    img[y1:y2, x1:x2] = 25
    lit = "green" if index < 15 else "red"  # clip frames at 30 fps, samples at 5 fps -> sample k = frame 6k
    for k, name in enumerate(("red", "yellow", "green")):
        centre = ((x1 + x2) // 2, y1 + int((y2 - y1) * (k + 0.5) / 3))
        cv2.circle(img, centre, 7, LIT[name] if name == lit else (55, 55, 55), -1)
    vx1, vy1, vx2, vy2 = (int(v) for v in VEHICLE_BOX)
    img[vy1:vy2, vx1:vx2] = (40, 60, 160)


class FakeEnsembleInferenceClientForTests:
    """Deterministic stand-in for a two-model InferenceClient. Test-only; never used by application code."""

    def __init__(self, *, single_model_results: bool = False) -> None:
        self.single_model_results = single_model_results
        self.calls: list[int] = []

    def model_info(self) -> ModelInfo:
        return ModelInfo.model_validate(
            {
                "model_id": "yolopv2",
                "ready": True,
                "sha256": YOLO_SHA,
                "class_map": {"3": "vehicle", **RFDETR_MAP},
                "device": "cpu",
                "precision": "fp32",
                "repository_revision": "9388c6aa55f3bf5a878679a1338bb090cff6a256",
                "ensemble_id": "yolopv2+rfdetr_nano",
                "ensemble_sha256": ENS_SHA,
                "models": [
                    {"model_id": "yolopv2", "ready": True, "sha256": YOLO_SHA, "class_map": {"3": "vehicle"}},
                    {
                        "model_id": "rfdetr_nano",
                        "ready": True,
                        "sha256": RFDETR_SHA,
                        "class_map": RFDETR_MAP,
                        "repository_revision": "7433d5919d0f7e72cc6cb78663e0a731afd499e0",
                    },
                ],
            }
        )

    def infer(
        self,
        image_bytes: bytes,
        frame_id: int,
        content_type: str = "image/jpeg",
        conf_threshold: float = 0.1,
        include_masks: bool = True,
    ) -> InferenceResult:
        self.calls.append(frame_id)
        h, w = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR).shape[:2]
        # Odd samples: the second detector relabels the YOLOPv2 vehicle box as a truck (class refinement).
        vehicle = (
            {"class_id": 108, "class_name": "truck", "model_id": "rfdetr_nano", "refined_from_model": "yolopv2"}
            if frame_id % 2
            else {"class_id": 3, "class_name": "vehicle", "model_id": "yolopv2"}
        )
        dets = [
            {**vehicle, "score": 0.86, "xyxy_pixels": VEHICLE_BOX},
            {
                "class_id": 101,
                "class_name": "person",
                "model_id": "rfdetr_nano",
                "score": 0.8,
                "xyxy_pixels": PERSON_BOX,
            },
            {
                "class_id": 110,
                "class_name": "traffic_light",
                "model_id": "rfdetr_nano",
                "score": 0.7,
                "xyxy_pixels": LIGHT_BOX,
            },
        ]
        drivable = np.zeros((h, w), dtype=bool)
        drivable[int(h * 0.6) :, int(w * 0.15) : int(w * 0.85)] = True
        lane = np.zeros((h, w), dtype=bool)
        lane[int(h * 0.55) :, int(w * 0.3) : int(w * 0.3) + 3] = True
        lane[int(h * 0.55) :, int(w * 0.7) : int(w * 0.7) + 3] = True
        yolo = {"model_id": "yolopv2", "sha256": YOLO_SHA, "device": "cpu", "precision": "fp32"}
        rfdetr = {"model_id": "rfdetr_nano", "sha256": RFDETR_SHA, "device": "cpu", "precision": "fp32"}
        body: dict[str, Any] = {
            "frame_id": frame_id,
            "model": yolo,
            "image": {"width": w, "height": h},
            "geometry": {
                "canvas_width": 640,
                "canvas_height": 384,
                "scale": 1.0,
                "pad_left": 0,
                "pad_top": 12,
                "pad_right": 0,
                "pad_bottom": 12,
            },
            "detections": dets,
            "drivable_mask_png": _png_b64(drivable),
            "lane_mask_png": _png_b64(lane),
            "mask_stats": {"drivable_fraction": float(drivable.mean()), "lane_fraction": float(lane.mean())},
            "timing_ms": {"decode": 1.0, "preprocess": 1.0, "inference": 20.0, "postprocess": 1.0, "total": 23.0},
        }
        if self.single_model_results:
            body["detections"] = [
                {**dets[0], "class_id": 3, "class_name": "vehicle", "model_id": "yolopv2", "refined_from_model": None}
            ]
        else:
            body |= {"models": [yolo, rfdetr], "ensemble_id": "yolopv2+rfdetr_nano", "ensemble_sha256": ENS_SHA}
        return InferenceResult.model_validate(body)


def _schema(name: str) -> jsonschema.Draft202012Validator:
    schema = json.loads((CONTRACTS / name).read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER)


def _run(tmp_path: Path, client: Any) -> tuple[dict[str, Any], Path]:
    clip = tmp_path / "scene.mp4"
    make_clip(clip, n_frames=60, fps=30.0, size=SIZE, draw=draw_scene)
    out_dir = tmp_path / "artifacts" / "jobs" / "job-ens"
    settings = Settings(git_commit=COMMIT, analysis_fps=5.0, app_env="test")
    summary = run_video_analysis("job-ens", clip, out_dir, client, settings, None, None, allow_pyav_probe_fallback=True)
    return summary, out_dir


def test_ensemble_identity_light_heuristic_and_reports(tmp_path: Path) -> None:
    summary, out_dir = _run(tmp_path, FakeEnsembleInferenceClientForTests())
    assert summary["analyzed_frames"] == 10
    timeline = json.loads((out_dir / "timeline.json").read_text(encoding="utf-8"))
    model = timeline["model"]
    assert model["model_id"] == "yolopv2+rfdetr_nano" and model["sha256"] == ENS_SHA and model["ensemble"] is True
    assert [(m["model_id"], m["checkpoint_sha256"]) for m in model["models"]] == [
        ("yolopv2", YOLO_SHA),
        ("rfdetr_nano", RFDETR_SHA),
    ]
    assert model["models"][1]["repository_revision"] == "7433d5919d0f7e72cc6cb78663e0a731afd499e0"
    assert timeline["light_state_heuristic"]["method"] == "hsv_heuristic"

    ws_validator = _schema("world-state.schema.json")
    extras = {"pts_s", "pts_estimated", "source_frame_index", "output_frame_index", "timing_ms", "lane_points"}
    extras |= {"measurements", "detection_sources", "traffic_lights", "light_state", "light_state_method"}
    for i, frame in enumerate(timeline["frames"]):
        ws = {k: v for k, v in frame.items() if k not in extras}
        ws_validator.validate(ws)
        assert ws["perception"]["model_id"] == "yolopv2+rfdetr_nano" and ws["perception"]["model_sha256"] == ENS_SHA
        dets, sources = ws["perception"]["detections"], frame["detection_sources"]
        assert len(dets) == len(sources) == 3
        assert [s["model_id"] for s in sources] == [
            ("rfdetr_nano" if i % 2 else "yolopv2"),
            "rfdetr_nano",
            "rfdetr_nano",
        ]
        assert sources[0]["refined_from_model"] == ("yolopv2" if i % 2 else None)
        (light,) = frame["traffic_lights"]
        assert light["detection_index"] == 2 and light["relevant"] is True
        assert light["light_state_method"] == "hsv_heuristic"
        expected = "green" if i < 3 else "red"  # sample i is clip frame 6 i; the light turns red at frame 15
        assert light["light_state"] == expected and frame["light_state"] == expected
        assert frame["light_state_method"] == "hsv_heuristic"
        codes = ws["decision"]["reason_codes"]
        assert "TRAFFIC_LIGHT_VISIBLE" in codes
        if expected == "red":
            assert ws["decision"]["state"] == "controlled_stop" and codes[0] == "TRAFFIC_LIGHT_RED_HEURISTIC"
        else:
            assert "TRAFFIC_LIGHT_GREEN_HEURISTIC" in codes and "TRAFFIC_LIGHT_RED_HEURISTIC" not in codes
        assert not any(c.startswith("PERSON") for c in codes)  # the person is on the sidewalk

    # One vehicle track survives the vehicle/truck relabels; the person has its own track.
    tracks = {t["class_name"]: t for t in summary["tracks"]}
    assert tracks["truck"]["class_counts"] == {"truck": 5, "vehicle": 5} and tracks["truck"]["group"] == "vehicle"
    assert tracks["person"]["group"] == "person" and tracks["person"]["track_id"] != tracks["truck"]["track_id"]
    assert summary["class_counts"] == {"vehicle": 5, "truck": 5, "person": 10, "traffic_light": 10}

    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    _schema("run-report.schema.json").validate(report)
    assert report["model_sha256"] == ENS_SHA
    ident = report["limitations"][0]
    assert YOLO_SHA in ident and RFDETR_SHA in ident and ENS_SHA in ident
    assert any("HSV heuristic" in item and "not model output" in item for item in report["limitations"])
    assert any("General traffic signs other than stop signs" in item for item in report["limitations"])
    m = report["metrics"]
    assert m["detections_person"]["value"] == 10 and m["detections_traffic_light"]["value"] == 10
    assert m["tracks_person"]["value"] == 1 and m["tracks_truck"]["value"] == 1
    assert m["detections_from_yolopv2"]["value"] == 5 and m["detections_from_rfdetr_nano"]["value"] == 25
    assert m["refined_detections"]["value"] == 5
    assert m["traffic_light_hsv_red"]["value"] == 7 and m["traffic_light_hsv_green"]["value"] == 3
    assert "not model output" in m["traffic_light_hsv_red"]["note"]

    html_text = (out_dir / "report.html").read_text(encoding="utf-8")
    for needle in ("yolopv2+rfdetr_nano", YOLO_SHA, RFDETR_SHA, "HSV heuristic", "Tracks per class"):
        assert needle in html_text
    assert "<script" not in html_text.lower()


def test_single_model_results_from_a_two_model_service_fail(tmp_path: Path) -> None:
    with pytest.raises(VideoError) as exc:
        _run(tmp_path, FakeEnsembleInferenceClientForTests(single_model_results=True))
    assert exc.value.code == errors.MODEL_CHANGED


def test_hud_label_and_identity_helpers() -> None:
    client = FakeEnsembleInferenceClientForTests()
    jpeg = cv2.imencode(".jpg", np.zeros((360, 640, 3), np.uint8))[1].tobytes()
    ident = perception_identity(client.infer(jpeg, 0))
    assert ident.is_ensemble and ident.model_id == "yolopv2+rfdetr_nano" and ident.sha256 == ENS_SHA
    assert model_label(ident) == "YOLOPv2 + RF-DETR-N"
    single = perception_identity(FakeEnsembleInferenceClientForTests(single_model_results=True).infer(jpeg, 0))
    assert not single.is_ensemble and single.model_id == "yolopv2" and single.sha256 == YOLO_SHA
    assert model_label(single) == "YOLOPv2"
    assert merged_class_map(client.model_info())["110"] == "traffic_light"


# --------------------------------------------------------------------------------------------------------------
# HTTP client validation of the ensemble wire fields (httpx MockTransport; no service)
# --------------------------------------------------------------------------------------------------------------
def _result_body(**overrides: Any) -> dict[str, Any]:
    yolo = {"model_id": "yolopv2", "sha256": YOLO_SHA, "device": "cpu", "precision": "fp32"}
    rfdetr = {"model_id": "rfdetr_nano", "sha256": RFDETR_SHA, "device": "cpu", "precision": "fp32"}
    body: dict[str, Any] = {
        "schema_version": "1.0",
        "frame_id": 0,
        "model": yolo,
        "models": [yolo, rfdetr],
        "ensemble_id": "yolopv2+rfdetr_nano",
        "ensemble_sha256": ENS_SHA,
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
        "detections": [
            {
                "class_id": 101,
                "class_name": "person",
                "score": 0.9,
                "xyxy_pixels": [1, 1, 10, 20],
                "model_id": "rfdetr_nano",
            },
            {
                "class_id": 108,
                "class_name": "truck",
                "score": 0.8,
                "xyxy_pixels": [20, 10, 40, 30],
                "model_id": "rfdetr_nano",
                "refined_from_model": "yolopv2",
            },
        ],
        "drivable_mask_png": None,
        "lane_mask_png": None,
        "mask_stats": {"drivable_fraction": 0.0, "lane_fraction": 0.0},
        "timing_ms": {
            "decode": 1,
            "preprocess": 1,
            "inference": 1,
            "postprocess": 1,
            "total": 4,
            "inference_by_model": {"yolopv2": 0.6, "rfdetr_nano": 0.4},
        },
    }
    body.update(overrides)
    return body


def _client(body: dict[str, Any]) -> InferenceClient:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    return InferenceClient("http://inference:8001", 5.0, transport=transport)


def test_client_accepts_a_consistent_ensemble_result() -> None:
    result = _client(_result_body()).infer(b"jpeg", 0, include_masks=False)
    assert [m.model_id for m in result.models] == ["yolopv2", "rfdetr_nano"]
    assert result.detections[1].refined_from_model == "yolopv2"
    assert perception_identity(result).sha256 == ENS_SHA


@pytest.mark.parametrize(
    "overrides",
    [
        {"ensemble_sha256": "0" * 64},  # does not match the members
        {"ensemble_sha256": None},  # two models without an ensemble identity
        {"ensemble_id": None},
        {"models": [{"model_id": "rfdetr_nano", "sha256": RFDETR_SHA, "device": "cpu", "precision": "fp32"}]},
        {"models": [], "ensemble_id": None, "ensemble_sha256": ENS_SHA},
    ],
)
def test_client_rejects_inconsistent_ensemble_identity(overrides: dict[str, Any]) -> None:
    with pytest.raises(InferenceError) as exc:
        _client(_result_body(**overrides)).infer(b"jpeg", 0, include_masks=False)
    assert exc.value.code == "invalid_response"


def test_client_rejects_detections_from_models_that_did_not_run() -> None:
    body = _result_body()
    body["detections"][0]["model_id"] = "yolov8"
    with pytest.raises(InferenceError):
        _client(body).infer(b"jpeg", 0, include_masks=False)
    body = _result_body()
    body["detections"][1]["refined_from_model"] = "other"
    with pytest.raises(InferenceError):
        _client(body).infer(b"jpeg", 0, include_masks=False)


def test_client_keeps_single_model_responses_working() -> None:
    body = _result_body(models=[], ensemble_id=None, ensemble_sha256=None)
    body["detections"] = [{"class_id": 3, "class_name": "vehicle", "score": 0.9, "xyxy_pixels": [1, 1, 10, 10]}]
    result = _client(body).infer(b"jpeg", 0, include_masks=False)
    ident = perception_identity(result)
    assert not ident.is_ensemble and ident.model_id == "yolopv2" and ident.sha256 == YOLO_SHA


def test_model_info_class_maps_are_merged_and_conflicts_rejected() -> None:
    info = FakeEnsembleInferenceClientForTests().model_info().model_dump()
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=info))
    parsed = InferenceClient("http://inference:8001", 5.0, transport=transport).model_info()
    assert merged_class_map(parsed) == {"3": "vehicle", **RFDETR_MAP}
    info["models"][1]["class_map"] = {"3": "person"}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=info))
    with pytest.raises(InferenceError):
        InferenceClient("http://inference:8001", 5.0, transport=transport).model_info()
    info = FakeEnsembleInferenceClientForTests().model_info().model_dump()
    info["ensemble_sha256"] = "1" * 64
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=info))
    with pytest.raises(InferenceError):
        InferenceClient("http://inference:8001", 5.0, transport=transport).model_info()
