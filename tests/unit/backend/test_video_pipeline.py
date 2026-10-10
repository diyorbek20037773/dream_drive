"""run_video_analysis end to end with a FAKE inference client (unit level only).

`FakeInferenceClientForTests` is deterministic and derives its box from the red rectangle in the received JPEG, so
the test also proves the pipeline sends the right (decoded, rotated) pixels. Real-model inference is exercised by
the model-smoke/integration gates and the documented end-to-end run, never by this fake.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import av
import cv2
import jsonschema
import numpy as np
import pytest
from test_video_synth import make_clip

from app.config import Settings
from app.perception.client import InferenceError, InferenceTimeout, InferenceUnavailable
from app.schemas.inference import InferenceResult, ModelInfo
from app.video import errors
from app.video.errors import JobCancelled, VideoError
from app.video.pipeline import OUTPUT_NAMES, decode_mask_png, run_video_analysis

ROOT = Path(__file__).resolve().parents[3]
CONTRACTS = ROOT / "contracts"
FAKE_SHA = "a" * 64
COMMIT = "0123456789abcdef0123456789abcdef01234567"
WS_EXTRA_KEYS = {
    "pts_s",
    "pts_estimated",
    "source_frame_index",
    "output_frame_index",
    "timing_ms",
    "lane_points",
    "measurements",
    # spec section 2.1 (aligned with perception.detections / HSV light heuristic, method "hsv_heuristic")
    "detection_sources",
    "traffic_lights",
    "light_state",
    "light_state_method",
}


def _png_b64(mask: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", (mask.astype(np.uint8) * 255))
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


class FakeInferenceClientForTests:
    """Deterministic stand-in for InferenceClient. Test-only; never used by application code."""

    def __init__(
        self,
        *,
        ready: bool = True,
        fail_frames: set[int] | None = None,
        fail_with: type[Exception] = InferenceTimeout,
        change_model_at: int | None = None,
    ) -> None:
        self.ready_flag = ready
        self.fail_frames = fail_frames or set()
        self.fail_with = fail_with
        self.change_model_at = change_model_at
        self.calls: list[int] = []
        self.sizes: list[tuple[int, int]] = []

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id="yolopv2",
            ready=self.ready_flag,
            reason=None if self.ready_flag else "model_missing",
            sha256=FAKE_SHA,
            class_map={"3": "vehicle"},
            device="cpu",
            precision="fp32",
            repository_revision="9388c6aa55f3bf5a878679a1338bb090cff6a256",
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
        if frame_id in self.fail_frames:
            raise self.fail_with("inference_timeout" if self.fail_with is InferenceTimeout else "fake_failure")
        bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        self.sizes.append((w, h))
        red = (bgr[:, :, 2] > 200) & (bgr[:, :, 1] < 60) & (bgr[:, :, 0] < 60)
        dets: list[dict[str, Any]] = []
        ys, xs = np.nonzero(red)
        if xs.size:
            dets.append(
                {
                    "class_id": 3,
                    "class_name": "vehicle",
                    "score": 0.88,
                    "xyxy_pixels": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                }
            )
        dets.append({"class_id": 3, "class_name": "vehicle", "score": 0.15, "xyxy_pixels": [1.0, 1.0, 9.0, 9.0]})
        drivable = np.zeros((h, w), dtype=bool)
        drivable[int(h * 0.6) :, int(w * 0.15) : int(w * 0.85)] = True
        lane = np.zeros((h, w), dtype=bool)
        lane[int(h * 0.55) :, int(w * 0.3) : int(w * 0.3) + 3] = True
        lane[int(h * 0.55) :, int(w * 0.7) : int(w * 0.7) + 3] = True
        sha = "b" * 64 if self.change_model_at is not None and frame_id >= self.change_model_at else FAKE_SHA
        return InferenceResult.model_validate(
            {
                "frame_id": frame_id,
                "model": {"model_id": "yolopv2", "sha256": sha, "device": "cpu", "precision": "fp32"},
                "image": {"width": w, "height": h},
                "geometry": {
                    "canvas_width": 640,
                    "canvas_height": 384,
                    "scale": 1.0,
                    "pad_left": 0,
                    "pad_top": 0,
                    "pad_right": 0,
                    "pad_bottom": 0,
                },
                "detections": dets,
                "drivable_mask_png": _png_b64(drivable) if include_masks else None,
                "lane_mask_png": _png_b64(lane) if include_masks else None,
                "mask_stats": {"drivable_fraction": float(drivable.mean()), "lane_fraction": float(lane.mean())},
                "timing_ms": {"decode": 1.0, "preprocess": 1.0, "inference": 10.0, "postprocess": 1.0, "total": 13.0},
            }
        )


def _settings(**kw: Any) -> Settings:
    base: dict[str, Any] = {"git_commit": COMMIT, "analysis_fps": 5.0, "app_env": "test"}
    base.update(kw)
    return Settings(**base)


def _schema(name: str) -> jsonschema.Draft202012Validator:
    schema = json.loads((CONTRACTS / name).read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER)


def _run(tmp_path: Path, clip: Path, client: Any, **kw: Any) -> tuple[dict[str, Any], Path, list]:
    out_dir = tmp_path / "artifacts" / "jobs" / "job1"
    progress: list[tuple[int, int, str]] = []
    summary = run_video_analysis(
        "job1",
        clip,
        out_dir,
        client,
        kw.pop("settings", _settings()),
        lambda d, t, s: progress.append((d, t, s)),
        kw.pop("cancel_check", None),
        allow_pyav_probe_fallback=True,
        **kw,
    )
    return summary, out_dir, progress


def _no_partials(tmp_path: Path) -> bool:
    jobs = tmp_path / "artifacts" / "jobs"
    return not jobs.exists() or not [p for p in jobs.iterdir() if ".partial-" in p.name]


def test_end_to_end_outputs_and_contracts(tmp_path: Path) -> None:
    clip = tmp_path / "in.mp4"
    make_clip(clip, n_frames=60, fps=30.0, size=(320, 180))  # 2.0 s
    client = FakeInferenceClientForTests()
    summary, out_dir, progress = _run(tmp_path, clip, client)

    assert sorted(p.name for p in out_dir.iterdir()) == sorted(OUTPUT_NAMES)
    assert _no_partials(tmp_path)
    assert client.calls == list(range(10))
    assert set(client.sizes) == {(320, 180)}

    # Summary = JobResults.summary shape (spec 3.4)
    assert set(summary) == {
        "analyzed_frames",
        "decoded_frames",
        "coverage_ratio",
        "duration_s",
        "effective_analyzed_fps",
        "timing_ms",
        "decision_counts",
        "reason_counts",
        "tracks",
        "class_counts",
        "audio",
    }
    assert set(summary["timing_ms"]) == {
        "decode",
        "inference_p50",
        "inference_p95",
        "postprocess",
        "encode",
        "total_wall",
    }
    assert summary["analyzed_frames"] == 10 and summary["decoded_frames"] == 60
    assert summary["coverage_ratio"] == 1.0
    assert summary["audio"] == "removed"
    assert summary["effective_analyzed_fps"] > 0
    assert sum(summary["decision_counts"].values()) == 10
    assert summary["class_counts"] == {"vehicle": 20}
    assert summary["tracks"][0]["track_id"] == 1 and summary["tracks"][0]["frames"] == 10

    # annotated.mp4: H.264, one frame per analyzed sample at the analysis rate
    with av.open(str(out_dir / "annotated.mp4")) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264" and len(container.streams.audio) == 0
        times = [f.time for f in container.decode(stream)]
    assert len(times) == 10 and times[-1] == pytest.approx(1.8, abs=1e-3)

    # timeline.json: every frame is a contract-valid WorldState plus the documented extra keys
    timeline = json.loads((out_dir / "timeline.json").read_text(encoding="utf-8"))
    assert timeline["schema_version"] == "1.0" and timeline["job_id"] == "job1"
    assert timeline["model"]["sha256"] == FAKE_SHA and timeline["analysis_fps"] == 5.0
    ws_validator = _schema("world-state.schema.json")
    pts = []
    for i, frame in enumerate(timeline["frames"]):
        assert WS_EXTRA_KEYS <= set(frame)
        ws = {k: v for k, v in frame.items() if k not in WS_EXTRA_KEYS}
        ws_validator.validate(ws)
        assert ws["mode"] == "video_analysis" and ws["ego"] is None and ws["command"] is None
        assert ws["decision"]["action_kind"] == "analytical_only"
        assert ws["perception"]["source"] == "learned_model" and ws["perception"]["model_sha256"] == FAKE_SHA
        assert ws["perception"]["ttc_seconds"] is None
        assert ws["frame"]["clock"] == "video_pts" and ws["frame"]["id"] == i
        assert frame["output_frame_index"] == i
        pts.append(frame["pts_s"])
        # low-score untracked decoy is kept in the record (honest model output) but has no track id
        decoy = [d for d in ws["perception"]["detections"] if d["score"] < 0.5]
        assert decoy and decoy[0]["track_id"] is None
    assert pts == pytest.approx([k / 5.0 for k in range(10)], abs=1e-4)

    # report.json: run-report contract
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    _schema("run-report.schema.json").validate(report)
    assert report["kind"] == "video_analysis" and report["status"] == "succeeded"
    assert report["perception_source"] == "learned_model" and report["model_sha256"] == FAKE_SHA
    assert report["code_commit"] == COMMIT
    import hashlib

    assert report["input_manifest_sha256"] == hashlib.sha256(clip.read_bytes()).hexdigest()
    m = report["metrics"]
    assert m["analyzed_frames"]["status"] == "measured" and m["analyzed_frames"]["value"] == 10
    assert (
        m["inference_roundtrip_latency_p95"]["unit"] == "ms"
        and m["inference_roundtrip_latency_p95"]["sample_count"] == 10
    )
    for name in ("detection_map50", "lane_iou", "tracking_idf1"):
        assert m[name]["status"] == "not_evaluated" and m[name]["value"] is None
        assert m[name]["note"] == "no labeled ground truth for this upload"
    assert m["time_to_collision"]["status"] == "unsupported"
    assert report["runtime_used_dynamic_ground_truth"] is False and report["evaluation_truth_used"] is False

    # report.html: self-contained, no scripts or external URLs
    html_text = (out_dir / "report.html").read_text(encoding="utf-8")
    assert "<script" not in html_text.lower()
    assert "http://" not in html_text and "https://" not in html_text
    assert "data:image/jpeg;base64," in html_text
    assert "Cheklovlar" in html_text and "Limitations" in html_text

    assert cv2.imread(str(out_dir / "poster.jpg")) is not None

    stages = [s for _, _, s in progress]
    assert stages[0] == "probing" and stages[-1] == "done"
    assert {"analyzing", "encoding", "writing_report"} <= set(stages)
    done = [d for d, _, s in progress if s == "analyzing"]
    assert done == sorted(done)


def test_existing_job_dir_with_input_is_preserved(tmp_path: Path) -> None:
    out_dir = tmp_path / "artifacts" / "jobs" / "job1"
    out_dir.mkdir(parents=True)
    clip = out_dir / "input"
    make_clip(clip, n_frames=15, fps=30.0)
    before = clip.read_bytes()
    summary = run_video_analysis(
        "job1", clip, out_dir, FakeInferenceClientForTests(), _settings(), None, None, allow_pyav_probe_fallback=True
    )
    assert summary["analyzed_frames"] == 3
    assert sorted(p.name for p in out_dir.iterdir()) == sorted([*OUTPUT_NAMES, "input"])
    assert clip.read_bytes() == before
    assert _no_partials(tmp_path)


def test_vfr_timestamps_reach_the_timeline(tmp_path: Path) -> None:
    clip = tmp_path / "vfr.mp4"
    make_clip(clip, times_s=[0.0, 0.033, 0.066, 0.4, 0.433, 1.2, 1.233])
    _, out_dir, _ = _run(tmp_path, clip, FakeInferenceClientForTests())
    timeline = json.loads((out_dir / "timeline.json").read_text(encoding="utf-8"))
    assert [f["pts_s"] for f in timeline["frames"]] == pytest.approx([0.0, 0.4, 1.2], abs=1e-4)
    assert [f["frame"]["timestamp_seconds"] for f in timeline["frames"]] == pytest.approx([0.0, 0.4, 1.2], abs=1e-4)


def test_rotation_is_applied_before_inference(tmp_path: Path) -> None:
    clip = tmp_path / "rot.mp4"
    make_clip(clip, n_frames=10, size=(320, 180), rotation_ccw=90)
    client = FakeInferenceClientForTests()
    _, out_dir, _ = _run(tmp_path, clip, client)
    assert set(client.sizes) == {(180, 320)}
    timeline = json.loads((out_dir / "timeline.json").read_text(encoding="utf-8"))
    assert timeline["video"]["rotation"] == 270
    assert (timeline["frames"][0]["frame"]["width"], timeline["frames"][0]["frame"]["height"]) == (180, 320)
    with av.open(str(out_dir / "annotated.mp4")) as c:
        f0 = next(c.decode(video=0))
        assert (f0.width, f0.height) == (180, 320)


def test_cancellation_removes_partial_outputs(tmp_path: Path) -> None:
    clip = tmp_path / "in.mp4"
    make_clip(clip, n_frames=60, fps=30.0)
    client = FakeInferenceClientForTests()
    checks = {"n": 0}

    def cancel() -> bool:
        checks["n"] += 1
        return checks["n"] > 3

    with pytest.raises(JobCancelled):
        _run(tmp_path, clip, client, cancel_check=cancel)
    assert client.calls == [0, 1, 2]
    assert not (tmp_path / "artifacts" / "jobs" / "job1").exists()
    assert _no_partials(tmp_path)


def test_build_identity_is_required(tmp_path: Path) -> None:
    clip = tmp_path / "in.mp4"
    make_clip(clip, n_frames=10)
    with pytest.raises(VideoError) as exc:
        _run(tmp_path, clip, FakeInferenceClientForTests(), settings=_settings(git_commit="unknown"))
    assert exc.value.code == errors.BUILD_IDENTITY_MISSING
    assert not (tmp_path / "artifacts").exists()


def test_model_not_ready_is_reported_not_mocked(tmp_path: Path) -> None:
    clip = tmp_path / "in.mp4"
    make_clip(clip, n_frames=10)
    client = FakeInferenceClientForTests(ready=False)
    with pytest.raises(VideoError) as exc:
        _run(tmp_path, clip, client)
    assert exc.value.code == errors.INFERENCE_UNAVAILABLE
    assert client.calls == []


def test_consecutive_inference_failures_fail_the_job(tmp_path: Path) -> None:
    clip = tmp_path / "in.mp4"
    make_clip(clip, n_frames=60, fps=30.0)
    with pytest.raises(VideoError) as exc:
        _run(tmp_path, clip, FakeInferenceClientForTests(fail_frames={2, 3, 4}))
    assert exc.value.code == errors.INFERENCE_FAILED
    assert _no_partials(tmp_path) and not (tmp_path / "artifacts" / "jobs" / "job1").exists()
    with pytest.raises(VideoError) as exc2:
        _run(tmp_path, clip, FakeInferenceClientForTests(fail_frames={0, 1, 2}, fail_with=InferenceUnavailable))
    assert exc2.value.code == errors.INFERENCE_UNAVAILABLE


def test_isolated_failure_is_recorded_honestly(tmp_path: Path) -> None:
    clip = tmp_path / "in.mp4"
    make_clip(clip, n_frames=66, fps=30.0)  # 11 samples; 1 failure = 9 % <= 10 %
    summary, out_dir, _ = _run(tmp_path, clip, FakeInferenceClientForTests(fail_frames={4}, fail_with=InferenceError))
    assert summary["analyzed_frames"] == 10
    assert summary["coverage_ratio"] == pytest.approx(10 / 11, abs=1e-6)
    assert summary["reason_counts"]["INFERENCE_FAILED"] == 1
    timeline = json.loads((out_dir / "timeline.json").read_text(encoding="utf-8"))
    failed = timeline["frames"][4]
    assert failed["perception"]["source"] == "unavailable"
    assert failed["perception"]["model_id"] is None and failed["perception"]["detections"] == []
    assert failed["decision"]["reason_codes"] == ["INFERENCE_FAILED"]
    assert failed["timing_ms"]["inference_server"] is None
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["metrics"]["inference_failed_frames"]["value"] == 1
    assert any("failed inference" in item for item in report["limitations"])


def test_model_identity_change_fails(tmp_path: Path) -> None:
    clip = tmp_path / "in.mp4"
    make_clip(clip, n_frames=30, fps=30.0)
    with pytest.raises(VideoError) as exc:
        _run(tmp_path, clip, FakeInferenceClientForTests(change_model_at=2))
    assert exc.value.code == errors.MODEL_CHANGED


def test_too_long_video_rejected_before_inference(tmp_path: Path) -> None:
    clip = tmp_path / "in.mp4"
    make_clip(clip, n_frames=60, fps=30.0)
    client = FakeInferenceClientForTests()
    with pytest.raises(VideoError) as exc:
        _run(tmp_path, clip, client, settings=_settings(max_video_seconds=1.0))
    assert exc.value.code == errors.VIDEO_TOO_LONG
    assert client.calls == []


def test_corrupt_upload_fails_cleanly(tmp_path: Path) -> None:
    clip = tmp_path / "in.mp4"
    clip.write_bytes(b"not a video at all" * 100)
    with pytest.raises(VideoError) as exc:
        _run(tmp_path, clip, FakeInferenceClientForTests())
    assert exc.value.code == errors.VIDEO_INVALID


def test_truncated_video_is_reported_as_limitation(tmp_path: Path) -> None:
    from test_video_decode import _corrupt_tail

    good = tmp_path / "good.mp4"
    make_clip(good, n_frames=60, fps=30.0)
    clip = tmp_path / "tail.mp4"
    _corrupt_tail(good, clip, 0.5)
    summary, out_dir, _ = _run(tmp_path, clip, FakeInferenceClientForTests())
    assert 0 < summary["decoded_frames"] < 60
    timeline = json.loads((out_dir / "timeline.json").read_text(encoding="utf-8"))
    assert timeline["sampling"]["stopped_at_decode_error"] is True
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert any("corrupt packet" in item for item in report["limitations"])


def test_mask_decoding() -> None:
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:10, 3:7] = True
    decoded = decode_mask_png(_png_b64(mask), 30, 20)
    assert decoded is not None and np.array_equal(decoded, mask)
    assert decode_mask_png(_png_b64(mask), 31, 20) is None  # wrong size is rejected, not resized
    assert decode_mask_png("!!notbase64!!", 30, 20) is None
    assert decode_mask_png(None, 30, 20) is None
