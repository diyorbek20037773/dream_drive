"""LearnedModelSimPerception mapping logic and the learned-mode gate.

These tests use a clearly fake, in-process client that returns hand-built masks/boxes so the *mapping* (IPM, lane
boundaries, corridor cue, error handling, identity tagging) can be checked deterministically. They prove nothing about
the real model; the real-service validation lives in tests/scenarios/test_learned_sim_validation.py.
"""

from __future__ import annotations

import base64
import hashlib

import cv2
import numpy as np
import pytest

from app.agents.runner import SessionConfig, SimulationSession, UnsupportedMode
from app.config import Settings
from app.perception.client import InferenceError, InferenceTimeout
from app.perception.learned_sim import LearnedModelSimPerception
from app.schemas.inference import InferenceResult
from app.simulators.lite.renderer import LiteRenderer, default_calibration
from app.simulators.lite.world import Box3D, get_world

FAKE_SHA = hashlib.sha256(b"fake-test-model-not-a-checkpoint").hexdigest()


def _png_b64(mask: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


class FakeClient:
    """Derives 'model outputs' from the pixels with fixed colour rules (test double, not inference)."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.calls = 0
        self.fail = fail

    def infer(
        self,
        image_bytes: bytes,
        frame_id: int,
        content_type: str = "image/jpeg",
        conf_threshold: float = 0.1,
        include_masks: bool = True,
    ) -> InferenceResult:
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(int)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        lane = ((r > 200) & (g > 200) & (b > 200)) | ((r > 200) & (g > 150) & (b < 90))
        orange = (r > 200) & (g > 70) & (g < 150) & (b < 70)
        dets = []
        if orange.any():
            ys, xs = np.nonzero(orange)
            lane[max(0, ys.min() - 2) : ys.max() + 3, max(0, xs.min() - 2) : xs.max() + 3] = False  # stripes
            dets.append(
                {
                    "class_id": 3,
                    "class_name": "vehicle",
                    "score": 0.6,
                    "xyxy_pixels": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                }
            )
        h, w = lane.shape
        return InferenceResult.model_validate(
            {
                "frame_id": frame_id,
                "model": {"model_id": "fake_test_model", "sha256": FAKE_SHA, "device": "cpu", "precision": "fp32"},
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
                "drivable_mask_png": None,
                "lane_mask_png": _png_b64(lane),
                "mask_stats": {"drivable_fraction": 0.0, "lane_fraction": float(lane.mean())},
                "timing_ms": {"decode": 0.0, "preprocess": 0.0, "inference": 0.0, "postprocess": 0.0, "total": 0.0},
            }
        )


@pytest.fixture(scope="module")
def renderer() -> LiteRenderer:
    return LiteRenderer(get_world(), default_calibration())


def test_mapping_lane_and_obstacle(renderer: LiteRenderer) -> None:
    perc = LearnedModelSimPerception(FakeClient(), default_calibration())
    img = renderer.render(20.0, -1.75, 0.0, [*get_world().static_boxes, Box3D(32.5, -1.75, 0, 1, 1, 1.2, "obstacle")])
    res = perc.process(img, 4)
    assert res.source == "learned_model" and res.model_id == "fake_test_model" and res.model_sha256 == FAKE_SHA
    assert res.lane_status in ("valid", "uncertain") and abs(res.lane_centre_left_m) < 0.3
    assert len(res.obstacles) == 1 and res.obstacles[0].in_corridor
    assert res.obstacles[0].distance_m == pytest.approx(12.0, abs=1.0)
    assert res.image_obstacle_cue == pytest.approx((24.0 - res.obstacles[0].distance_m) / 16.0, abs=1e-3)
    assert "synthetic_render_outside_model_training_domain" in res.uncertainty_reasons
    p = res.to_perception()
    assert p.source == "learned_model" and p.model_sha256 == FAKE_SHA


def test_black_frame_never_reaches_the_model(renderer: LiteRenderer) -> None:
    client = FakeClient()
    res = LearnedModelSimPerception(client, default_calibration()).process(np.zeros((360, 640, 3), np.uint8))
    assert client.calls == 0 and res.quality_status == "missing" and "CAMERA_BLACK" in res.quality_codes
    assert res.source == "unavailable", "no model identity known yet -> not labelled learned_model"


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (InferenceTimeout("inference_timeout"), "PERCEPTION_TIMEOUT"),
        (InferenceError("inference_error"), "INFERENCE_FAILED"),
    ],
)
def test_client_errors_map_to_safety_codes(renderer: LiteRenderer, exc: Exception, code: str) -> None:
    res = LearnedModelSimPerception(FakeClient(fail=exc), default_calibration()).process(
        renderer.render(10.0, -1.75, 0.0, [])
    )
    assert res.quality_status == "missing" and res.quality_codes == (code,)


def test_learned_mode_gate_and_session_tagging(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SessionConfig(run_id="gate", start_node="A", destination_node="C", perception_source="learned_model")
    with pytest.raises(UnsupportedMode):
        SimulationSession(cfg, inference_client=FakeClient())
    import app.agents.runner as runner_mod

    monkeypatch.setattr(runner_mod, "get_settings", lambda: Settings(learned_sim_mode_enabled=True))
    with pytest.raises(UnsupportedMode, match="inference client"):
        SimulationSession(cfg)
    s = SimulationSession(cfg, inference_client=FakeClient())
    r = s.start()
    for _ in range(10):
        r = s.tick()
    assert r.world_state.perception.source == "learned_model"
    assert r.world_state.perception.model_sha256 == FAKE_SHA
    assert r.live["perception"]["source"] == "learned_model"
    timeout_session = SimulationSession(cfg, inference_client=FakeClient(fail=InferenceTimeout("inference_timeout")))
    timeout_session.start()
    r = timeout_session.tick()
    assert r.live["decision"]["state"] == "safety_stop" and "PERCEPTION_TIMEOUT" in r.live["decision"]["reason_codes"]
    assert r.live["command"]["brake"] == 1.0


def test_learned_mode_stale_frame_records_stay_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Frames rejected before perception (stale) still produce schema-valid learned-mode world states."""
    import app.agents.runner as runner_mod

    monkeypatch.setattr(runner_mod, "get_settings", lambda: Settings(learned_sim_mode_enabled=True))
    cfg = SessionConfig(
        run_id="stale", start_node="A", destination_node="C", scenario="stale_frame", perception_source="learned_model"
    )
    s = SimulationSession(cfg, inference_client=FakeClient())
    r = s.start()
    stale = []
    while not r.terminal:
        r = s.tick()
        if r.live["perception"]["quality_status"] == "stale":
            stale.append(r)
    assert r.outcome == "stopped_frame_stale" and stale
    assert all(x.world_state.perception.source == "learned_model" for x in stale)
    assert all(x.world_state.perception.model_sha256 == FAKE_SHA for x in stale)
