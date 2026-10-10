"""Contract tests: Pydantic dumps validate against contracts/*.json (Draft 2020-12 + format checker), and the
semantic rules that JSON Schema cannot express (xyxy order, exclusive throttle/brake, ...) are enforced by Pydantic.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pytest
from pydantic import ValidationError

from app.perception.video_decisions import decide
from app.perception.world_state import ModelRef, TrackedDetection, build_video_world_state
from app.reporting.video_report import VideoRunFacts, build_run_report
from app.schemas.events import EventEnvelope
from app.schemas.report import Metric, RunReport
from app.schemas.world_state import Command, Detection, WorldState

CONTRACTS = Path(__file__).resolve().parents[3] / "contracts"
SHA = "f2a8c8374203ae3e67ff9c184e931f763957de92a993b23269e4e721627f1f8c"
COMMIT = "f85c397" + "0" * 33
NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def _validator(name: str) -> jsonschema.Draft202012Validator:
    schema = json.loads((CONTRACTS / name).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER)


WS = _validator("world-state.schema.json")
RR = _validator("run-report.schema.json")
EV = _validator("event.schema.json")


def _errors(validator: jsonschema.Draft202012Validator, doc: Any) -> list[str]:
    return [e.message for e in validator.iter_errors(doc)]


# ---------------------------------------------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------------------------------------------
def _video_world_state() -> WorldState:
    h, w = 360, 640
    image = np.random.default_rng(0).integers(60, 200, size=(h, w, 3), dtype=np.uint8)
    lane = np.zeros((h, w), dtype=bool)
    lane[200:, 200:204] = True
    lane[200:, 440:444] = True
    drivable = np.zeros((h, w), dtype=bool)
    drivable[200:, 100:540] = True
    dets = [
        TrackedDetection(3, "vehicle", 0.91, [250.0, 200.0, 390.0, 330.0], 1),
        TrackedDetection(5, "unverified_class_5", 0.4, [0.0, 0.0, 640.4, 12.0], None),  # clipped to the image
    ]
    decision = decide(image, drivable, lane, dets)
    return build_video_world_state(
        job_id="job-1",
        frame_id=3,
        pts_s=0.6,
        acquired_utc=NOW,
        width=w,
        height=h,
        decision=decision,
        detections=dets,
        model=ModelRef("yolopv2", SHA),
    )


def _simulation_world_state() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "run_id": "run-1",
        "mode": "simulation",
        "frame": {
            "id": 12,
            "timestamp_seconds": 0.6,
            "clock": "simulation_time",
            "acquired_utc": NOW.isoformat(),
            "width": 640,
            "height": 360,
            "quality_status": "valid",
        },
        "perception": {
            "source": "synthetic_rgb_cv",
            "model_id": None,
            "model_sha256": None,
            "detections": [
                {
                    "track_id": None,
                    "class_id": 0,
                    "class_name": "obstacle",
                    "score": 0.8,
                    "xyxy_pixels": [300, 180, 340, 220],
                }
            ],
            "lane_mask_ref": None,
            "drivable_mask_ref": None,
            "lane_status": "valid",
            "image_lane_offset_normalized": -0.1,
            "image_obstacle_cue": 0.4,
            "ttc_seconds": None,
            "uncertainty_reasons": [],
        },
        "ego": {"source": "ego_telemetry", "x_m": 10.0, "y_m": -1.75, "yaw_rad": 0.0, "speed_mps": 3.2},
        "decision": {"state": "slow", "reason_codes": ["OBSTACLE_AHEAD_SLOW"], "action_kind": "simulator_control"},
        "command": {
            "steer": 0.05,
            "throttle": 0.0,
            "brake": 0.3,
            "input_frame_id": 12,
            "source": "dreamdrive_controller",
            "safety_override": False,
            "reason_codes": ["OBSTACLE_AHEAD_SLOW"],
        },
    }


def _video_facts() -> VideoRunFacts:
    return VideoRunFacts(
        job_id="job-1",
        code_commit=COMMIT,
        generated_utc=NOW,
        input_sha256="1" * 64,
        input_bytes=1000,
        video={"duration_s": 10.0, "width": 1280, "height": 720, "fps_avg": 30.0, "rotation": 0, "codec": "h264"},
        probe_tool="ffprobe",
        analysis_fps=5.0,
        model_id="yolopv2",
        model_sha256=SHA,
        model_device="cpu",
        model_precision="fp32",
        model_revision="9388c6aa55f3bf5a878679a1338bb090cff6a256",
        class_map={"3": "vehicle"},
        decoded_frames=300,
        sampled_frames=50,
        analyzed_frames=50,
        failed_frames=0,
        estimated_pts_frames=0,
        output_frames=50,
        roundtrip_ms=[400.0 + i for i in range(50)],
        server_inference_ms=[380.0] * 50,
        decode_ms=1000.0,
        postprocess_ms=2000.0,
        encode_ms=3000.0,
        total_wall_ms=34000.0,
        vehicle_detections=100,
        lane_valid_frames=45,
        decision_counts={"observe": 50},
        reason_counts={"LANE_VALID": 45},
        class_counts={"vehicle": 100},
        tracks=[],
        limitations=["analytical only"],
    )


def _simulation_report() -> dict[str, Any]:
    report = RunReport(
        run_id="run-1",
        kind="simulation",
        status="arrived",
        perception_source="synthetic_rgb_cv",
        code_commit=COMMIT,
        model_sha256=None,
        input_manifest_sha256=None,
        generated_utc=NOW,
        metrics={
            "route_completion": Metric.measured(1.0, "ratio", 1),
            "collisions": Metric.measured(0, "count", 1, "evaluator truth, after the run"),
            "lane_iou": Metric.missing("ratio", "synthetic CV has no lane IoU labels"),
        },
        limitations=["synthetic LiteRoad environment only"],
        evaluation_truth_used=True,
    )
    return report.model_dump(mode="json")


# ---------------------------------------------------------------------------------------------------------------
# Positive validation
# ---------------------------------------------------------------------------------------------------------------
def test_video_world_state_matches_contract() -> None:
    doc = _video_world_state().model_dump(mode="json")
    assert _errors(WS, doc) == []
    assert doc["perception"]["detections"][1]["xyxy_pixels"] == [0.0, 0.0, 640.0, 12.0]


def test_simulation_world_state_matches_contract() -> None:
    doc = _simulation_world_state()
    assert _errors(WS, doc) == []
    assert WorldState.model_validate(doc).model_dump(mode="json")["command"]["brake"] == 0.3


def test_video_run_report_matches_contract() -> None:
    doc = build_run_report(_video_facts())
    assert _errors(RR, doc) == []
    assert doc["metrics"]["inference_roundtrip_latency_p50"]["value"] == pytest.approx(424.5)
    assert doc["metrics"]["detection_map50"] == {
        "value": None,
        "unit": "ratio",
        "sample_count": 0,
        "status": "not_evaluated",
        "note": "no labeled ground truth for this upload",
    }


def test_simulation_run_report_matches_contract() -> None:
    assert _errors(RR, _simulation_report()) == []


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "job_state", "job_id": "j", "state": "running", "error_code": None},
        {
            "kind": "job_progress",
            "job_id": "j",
            "progress": 0.5,
            "stage": "analyzing",
            "frames_done": 5,
            "frames_total": 10,
        },
        {"kind": "run_state", "run_id": "r", "state": "arrived", "outcome": "arrived"},
        {"kind": "frame_ready", "run_id": "r", "frame_id": 3, "sim_time_s": 0.15},
        {
            "kind": "decision",
            "run_id": "r",
            "frame_id": 3,
            "state": "slow",
            "reason_codes": ["OBSTACLE_AHEAD_SLOW"],
            "command": {"steer": 0.0, "throttle": 0.0, "brake": 0.4},
        },
        {"kind": "capability_changed", "video_analysis_available": False, "reason": "model_missing"},
        {"kind": "error", "code": "internal", "message": "worker crashed"},
    ],
)
def test_event_envelopes_match_contract(payload: dict[str, Any]) -> None:
    env = EventEnvelope(
        event_id="1700000000000-0", timestamp_utc=NOW, resource_id="res-1", type=payload["kind"], payload=payload
    )
    doc = env.model_dump(mode="json")
    assert _errors(EV, doc) == []
    assert EventEnvelope.model_validate(doc) == env


# ---------------------------------------------------------------------------------------------------------------
# Negative / semantic rules
# ---------------------------------------------------------------------------------------------------------------
def test_format_checker_is_active() -> None:
    doc = _simulation_report()
    doc["generated_utc"] = "yesterday afternoon"
    assert any("date-time" in m for m in _errors(RR, doc))


def test_unordered_xyxy_rejected_semantically() -> None:
    # JSON Schema only bounds each coordinate; ordering is a semantic rule enforced by the Pydantic model.
    doc = _simulation_world_state()
    doc["perception"]["detections"][0]["xyxy_pixels"] = [340, 180, 300, 220]
    assert _errors(WS, doc) == []
    with pytest.raises(ValidationError, match="ordered"):
        WorldState.model_validate(doc)
    with pytest.raises(ValidationError):
        Detection(class_id=3, class_name="vehicle", score=0.5, xyxy_pixels=[-1, 0, 5, 5])


def test_exclusive_throttle_brake_rejected_semantically() -> None:
    doc = _simulation_world_state()
    doc["command"]["throttle"] = 0.2
    assert _errors(WS, doc) == []  # schema bounds pass
    with pytest.raises(ValidationError, match="throttle and brake"):
        WorldState.model_validate(doc)
    with pytest.raises(ValidationError):
        Command(steer=0, throttle=0.1, brake=0.1, input_frame_id=1, safety_override=False, reason_codes=["X"])


def test_video_mode_rejects_command_and_ego() -> None:
    doc = _video_world_state().model_dump(mode="json")
    doc["command"] = _simulation_world_state()["command"]
    assert _errors(WS, doc)
    with pytest.raises(ValidationError):
        WorldState.model_validate(doc)
    doc2 = _video_world_state().model_dump(mode="json")
    doc2["decision"]["action_kind"] = "simulator_control"
    assert _errors(WS, doc2)
    with pytest.raises(ValidationError):
        WorldState.model_validate(doc2)


def test_learned_model_requires_identity() -> None:
    doc = _video_world_state().model_dump(mode="json")
    doc["perception"]["model_sha256"] = None
    assert _errors(WS, doc)
    with pytest.raises(ValidationError):
        WorldState.model_validate(doc)


def test_dynamic_ground_truth_fields_are_rejected() -> None:
    doc = _simulation_world_state()
    doc["perception"]["obstacle_truth_m"] = [12.0, 0.5]
    assert _errors(WS, doc)
    with pytest.raises(ValidationError):
        WorldState.model_validate(doc)
    report = _simulation_report()
    report["runtime_used_dynamic_ground_truth"] = True
    assert _errors(RR, report)
    with pytest.raises(ValidationError):
        RunReport.model_validate(report)


@pytest.mark.parametrize(
    "metric",
    [
        {"value": None, "unit": "ms", "sample_count": 3, "status": "measured", "note": ""},
        {"value": 1.0, "unit": "ms", "sample_count": 0, "status": "measured", "note": ""},
        {"value": 0.5, "unit": "ratio", "sample_count": 0, "status": "not_evaluated", "note": "no labels"},
        {"value": None, "unit": "ratio", "sample_count": 0, "status": "unsupported", "note": ""},
    ],
)
def test_measured_metric_rules(metric: dict[str, Any]) -> None:
    report = _simulation_report()
    report["metrics"]["bad"] = metric
    assert _errors(RR, report)
    with pytest.raises(ValidationError):
        RunReport.model_validate(report)


def test_pydantic_required_fields_match_contracts() -> None:
    ws_schema = json.loads((CONTRACTS / "world-state.schema.json").read_text(encoding="utf-8"))
    py = WorldState.model_json_schema()
    assert set(py["required"]) <= set(ws_schema["required"])
    assert set(ws_schema["properties"]) == set(py["properties"])
    for sub in ("frame", "perception"):
        contract = ws_schema["properties"][sub]
        model = py["$defs"]["FrameInfo" if sub == "frame" else "Perception"]
        assert set(contract["properties"]) == set(model["properties"])
    rr_schema = json.loads((CONTRACTS / "run-report.schema.json").read_text(encoding="utf-8"))
    assert set(rr_schema["properties"]) == set(RunReport.model_json_schema()["properties"])


# ---------------------------------------------------------------------------------------------------------------
# Two-model ensemble (YOLOPv2 + RF-DETR-N, docs/IMPLEMENTATION_SPEC.md section 2.1)
# ---------------------------------------------------------------------------------------------------------------
RFDETR_SHA = "d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d"


def test_ensemble_world_state_and_report_match_contracts() -> None:
    from dataclasses import replace

    from app.perception.client import ensemble_sha256

    ens_sha = ensemble_sha256([("yolopv2", SHA), ("rfdetr_nano", RFDETR_SHA)])
    h, w = 360, 640
    image = np.random.default_rng(1).integers(60, 200, size=(h, w, 3), dtype=np.uint8)
    dets = [
        TrackedDetection(108, "truck", 0.8, [250.0, 200.0, 390.0, 330.0], 1, "rfdetr_nano", "yolopv2"),
        TrackedDetection(101, "person", 0.7, [20.0, 150.0, 60.0, 300.0], 2, "rfdetr_nano"),
        TrackedDetection(110, "traffic_light", 0.6, [310.0, 40.0, 334.0, 106.0], 3, "rfdetr_nano"),
    ]
    ws = build_video_world_state(
        job_id="job-ens",
        frame_id=0,
        pts_s=0.0,
        acquired_utc=NOW,
        width=w,
        height=h,
        decision=decide(image, None, None, dets),
        detections=dets,
        model=ModelRef("yolopv2+rfdetr_nano", ens_sha),
    )
    doc = ws.model_dump(mode="json")
    assert _errors(WS, doc) == []
    assert doc["perception"]["model_id"] == "yolopv2+rfdetr_nano" and doc["perception"]["model_sha256"] == ens_sha
    assert "model_id" not in doc["perception"]["detections"][0]  # provenance lives in timeline extras only

    facts = replace(
        _video_facts(),
        model_id="yolopv2+rfdetr_nano",
        model_sha256=ens_sha,
        model_revision=None,
        class_map={"3": "vehicle", "101": "person", "110": "traffic_light"},
        models=[
            {"model_id": "yolopv2", "checkpoint_sha256": SHA},
            {"model_id": "rfdetr_nano", "checkpoint_sha256": RFDETR_SHA},
        ],
        class_counts={"vehicle": 80, "truck": 20, "person": 12, "traffic_light": 9},
        detection_counts_by_model={"yolopv2": 80, "rfdetr_nano": 41},
        refined_detection_counts={"yolopv2->truck": 20},
        track_counts_by_class={"vehicle": 3, "person": 2},
        light_state_counts={"red": 4, "unknown": 5},
        frame_light_state_counts={"red": 4},
    )
    report = build_run_report(facts)
    assert _errors(RR, report) == []
    assert report["model_sha256"] == ens_sha
    assert SHA in report["limitations"][0] and RFDETR_SHA in report["limitations"][0]
    metrics = report["metrics"]
    assert metrics["detections_person"]["value"] == 12 and metrics["tracks_vehicle"]["value"] == 3
    assert metrics["traffic_light_hsv_red"]["status"] == "measured" and metrics["refined_detections"]["value"] == 20
    assert metrics["traffic_light_state_known_ratio"]["value"] == pytest.approx(4 / 9, abs=1e-6)
