"""Reporting threshold (DETECTION_REPORT_MIN_SCORE): low-score boxes stay in timeline.json and in a separate
low-score bucket, but are not counted as findings in the job summary, report.json/html or the track list; the job's
model identity is the ensemble identity when the service runs two models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_video_pipeline_ensemble import (
    ENS_SHA,
    FakeEnsembleInferenceClientForTests,
    _run,
    _schema,
)

from app.config import Settings
from app.jobs.video_job import model_identity
from app.perception.world_state import TrackedDetection
from app.schemas.api import JobSummary
from app.schemas.inference import InferenceResult
from app.video.pipeline import report_tracks, split_class_counts

STOP_BOX = [520.0, 60.0, 560.0, 100.0]
FAR_PERSON_BOX = [600.0, 140.0, 620.0, 200.0]


class LowScoreEnsembleClient(FakeEnsembleInferenceClientForTests):
    """Adds a 0.10 stop sign and a 0.22 person (the kind of boxes ByteTrack's second association needs)."""

    def infer(self, image_bytes: bytes, frame_id: int, *args: Any, **kwargs: Any) -> InferenceResult:
        res = super().infer(image_bytes, frame_id, *args, **kwargs)
        body = res.model_dump(mode="json")
        body["detections"] += [
            {
                "class_id": 113,
                "class_name": "stop_sign",
                "model_id": "rfdetr_nano",
                "score": 0.1,
                "xyxy_pixels": STOP_BOX,
            },
            {
                "class_id": 101,
                "class_name": "person",
                "model_id": "rfdetr_nano",
                "score": 0.22,
                "xyxy_pixels": FAR_PERSON_BOX,
            },
        ]
        return InferenceResult.model_validate(body)


def _det(name: str, score: float, track: int | None = None) -> TrackedDetection:
    return TrackedDetection(101, name, score, [0.0, 0.0, 10.0, 10.0], track)


def test_split_class_counts_uses_the_threshold_inclusively() -> None:
    dets = [_det("person", 0.22), _det("person", 0.4), _det("person", 0.91), _det("stop_sign", 0.1)]
    high, low = split_class_counts(dets, 0.4)
    assert dict(high) == {"person": 2}
    assert dict(low) == {"person": 1, "stop_sign": 1}


def test_report_tracks_drops_low_max_score_tracks_and_counts_them() -> None:
    tracks = [
        {"track_id": 1, "class_name": "person", "max_score": 0.22},
        {"track_id": 2, "class_name": "vehicle", "max_score": 0.4},
        {"track_id": 3, "class_name": "truck", "max_score": 0.87},
    ]
    kept, dropped = report_tracks(tracks, 0.4)
    assert [t["track_id"] for t in kept] == [2, 3]
    assert dropped == 1


def test_setting_default_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings().detection_report_min_score == 0.4
    monkeypatch.setenv("DETECTION_REPORT_MIN_SCORE", "0.55")
    assert Settings().detection_report_min_score == 0.55


def test_pipeline_reports_low_score_detections_separately(tmp_path: Path) -> None:
    summary, out_dir = _run(tmp_path, LowScoreEnsembleClient())
    # Findings are unchanged by the low-score boxes; the low-score bucket shows them, nothing is hidden.
    assert summary["class_counts"] == {"vehicle": 5, "truck": 5, "person": 10, "traffic_light": 10}
    assert summary["class_counts_low_score"] == {"stop_sign": 10, "person": 10}
    assert summary["detection_report_min_score"] == 0.4
    assert all(t["max_score"] >= 0.4 for t in summary["tracks"])
    assert summary["low_score_tracks_dropped"] >= 0
    parsed = JobSummary.model_validate(summary)
    assert parsed.class_counts_low_score["stop_sign"] == 10

    timeline = json.loads((out_dir / "timeline.json").read_text(encoding="utf-8"))
    for frame in timeline["frames"]:
        scores = sorted(d["score"] for d in frame["perception"]["detections"])
        assert len(scores) == 5 and scores[0] == pytest.approx(0.1) and scores[1] == pytest.approx(0.22)
        # The 0.10 stop sign never drives a decision.
        assert "STOP_SIGN_VISIBLE" not in frame["decision"]["reason_codes"]

    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    _schema("run-report.schema.json").validate(report)
    m = report["metrics"]
    assert m["detections_person"]["value"] == 10 and "score >= 0.4" in m["detections_person"]["note"]
    assert "detections_stop_sign" not in m
    assert m["low_score_detections_stop_sign"]["value"] == 10
    assert m["low_score_detections_person"]["value"] == 10
    assert any(item.startswith("Reporting threshold 0.4") for item in report["limitations"])

    html_text = (out_dir / "report.html").read_text(encoding="utf-8")
    assert "Low-confidence detections" in html_text and "DETECTION_REPORT_MIN_SCORE" in html_text


def test_old_summaries_without_threshold_fields_still_validate() -> None:
    old = {
        "analyzed_frames": 1,
        "decoded_frames": 1,
        "coverage_ratio": 1.0,
        "duration_s": 1.0,
        "effective_analyzed_fps": 1.0,
        "timing_ms": {"decode": 1.0, "postprocess": 1.0, "encode": 1.0, "total_wall": 1.0},
        "decision_counts": {},
        "reason_counts": {},
        "tracks": [],
        "class_counts": {"person": 3},
    }
    parsed = JobSummary.model_validate(old)
    assert parsed.detection_report_min_score is None and parsed.class_counts_low_score == {}


def test_job_model_identity_is_the_ensemble_identity() -> None:
    ident = model_identity(FakeEnsembleInferenceClientForTests())
    assert ident == {"model_id": "yolopv2+rfdetr_nano", "sha256": ENS_SHA, "device": "cpu", "precision": "fp32"}
