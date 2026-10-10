"""Live validation of learned-model perception on LiteRoad renders (gate for LEARNED_SIM_MODE_ENABLED).

Runs against a real inference service (env ``DD_INFERENCE_URL``, e.g. http://inference:8001). Skipped when the
variable is unset, unless ``DD_REQUIRE_LIVE=1`` (then it fails, so `make qa` cannot pass without it).

Checks (each recorded as pass/fail with measurements, never assumed):
1. lane: straight-road renders at known lateral offsets -> perceived lane-centre offset error;
2. obstacle: closed-loop ``obstacle_stop`` (seed 1) -> must stop before the obstacle without collision;
3. missing camera: closed-loop ``camera_blackout`` (seed 1) -> safety stop on the first black frame.
The measured outcome is written to ``$DD_EVIDENCE_DIR`` (default ``<repo>/artifacts/learned_sim_validation``).

Assertions: the harness really ran the model (identity reported by the service, tagged on every perception record)
and the configuration is consistent: ``LEARNED_SIM_MODE_ENABLED=true`` is only acceptable when every check passed.
A failing check with the mode disabled is a valid, reportable result (domain shift), not a test failure.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.timeout(3600)

ROOT = Path(__file__).resolve().parents[2]
LANE_OFFSETS_M = (-0.5, -0.25, 0.0, 0.25, 0.5)
LANE_TOLERANCE_M = 0.4


def _live_url() -> str:
    url = os.environ.get("DD_INFERENCE_URL", "").strip()
    if not url:
        if os.environ.get("DD_REQUIRE_LIVE") == "1":
            pytest.fail("DD_REQUIRE_LIVE=1 but DD_INFERENCE_URL is not set: learned-sim validation cannot run")
        pytest.skip("DD_INFERENCE_URL not set (live learned-sim validation only runs against a real service)")
    return url


def _closed_loop(scenario: str, client: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import app.agents.runner as runner_mod
    from app.config import Settings

    # Enable the mode for this measurement only (the deployed default stays false) and allow slow CPU inference.
    monkeypatch.setattr(
        runner_mod, "get_settings", lambda: Settings(learned_sim_mode_enabled=True, max_simulation_wall_seconds=1500)
    )
    session = runner_mod.SimulationSession(
        runner_mod.SessionConfig(
            run_id=f"learned-validation-{scenario}",
            start_node="A",
            destination_node="C",
            scenario=scenario,
            seed=1,
            perception_source="learned_model",
            max_ticks=600,
            encode_jpeg=False,
        ),
        inference_client=client,
    )
    t0 = time.monotonic()
    r = session.start()
    records = [r.world_state.perception]
    first_black: dict[str, Any] | None = None
    while not r.terminal:
        r = session.tick()
        records.append(r.world_state.perception)
        if first_black is None and r.live["perception"]["quality_status"] == "missing":
            first_black = {
                "sim_time_s": r.sim_time_s,
                "state": r.live["decision"]["state"],
                "brake": r.live["command"]["brake"] if r.live["command"] else None,
            }
    summary = session.summary(time.monotonic() - t0)
    return {"summary": summary, "records": records, "first_black": first_black}


def test_learned_model_on_literoad_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    url = _live_url()
    from app.config import get_settings
    from app.perception.client import InferenceClient
    from app.perception.learned_sim import LearnedModelSimPerception
    from app.simulators.lite.renderer import LiteRenderer, default_calibration
    from app.simulators.lite.world import get_world

    settings = get_settings()
    client = InferenceClient(url, timeout_s=float(settings.sim_inference_timeout_seconds))
    info = client.model_info()
    assert client.ready() and info.ready, f"inference service at {url} is not ready: {info.reason}"
    assert info.sha256 and len(info.sha256) == 64

    outcome: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(UTC).isoformat(),
        "inference_url_host": url.split("//")[-1].split("/")[0],
        "model": {
            "model_id": info.model_id,
            "sha256": info.sha256,
            "device": info.device,
            "precision": info.precision,
            "class_map": info.class_map,
        },
        "checks": {},
    }

    # 1. lane geometry on straight renders at known lateral offsets (known from the test's own scene setup)
    calib = default_calibration()
    renderer = LiteRenderer(get_world(), calib)
    perception = LearnedModelSimPerception(client, calib)
    lane_rows = []
    for i, off in enumerate(LANE_OFFSETS_M):
        img = renderer.render(15.0, -1.75 + off, 0.0, list(get_world().static_boxes))
        res = perception.process(img, i)
        assert res.model_sha256 == info.sha256, "every learned perception record carries the served checkpoint"
        err = None if res.lane_centre_left_m is None else abs(res.lane_centre_left_m - (-off))
        lane_rows.append(
            {
                "ego_offset_m": off,
                "lane_status": res.lane_status,
                "lane_centre_left_m": res.lane_centre_left_m,
                "abs_error_m": err,
                "inference_ms": res.timing_ms,
                "uncertainty": list(res.uncertainty_reasons),
            }
        )
    good = sum(1 for r in lane_rows if r["abs_error_m"] is not None and r["abs_error_m"] <= LANE_TOLERANCE_M)
    outcome["checks"]["lane"] = {
        "passed": good >= 4,
        "frames": len(lane_rows),
        "within_tolerance": good,
        "tolerance_m": LANE_TOLERANCE_M,
        "rows": lane_rows,
    }

    # 2. visible obstacle: closed loop must stop before it
    obs = _closed_loop("obstacle_stop", client, monkeypatch)
    s = obs["summary"]
    outcome["checks"]["obstacle"] = {
        "passed": s["outcome"] == "stopped_before_obstacle" and s["evaluator"]["collisions"] == 0,
        "outcome": s["outcome"],
        "collisions": s["evaluator"]["collisions"],
        "ticks": s["ticks"],
        "final_obstacle_gap_m": s["evaluator"]["final_obstacle_gap_m"],
        "decision_counts": s["decision_counts"],
        "perception_ms_p50": s["timing_ms"]["perception_p50"],
        "wall_time_s": s["wall_time_s"],
    }

    # 3. missing camera: safety stop on the first black frame
    blk = _closed_loop("camera_blackout", client, monkeypatch)
    sb, fb = blk["summary"], blk["first_black"]
    outcome["checks"]["missing_camera"] = {
        "passed": sb["outcome"] == "stopped_camera_invalid"
        and fb is not None
        and fb["state"] == "safety_stop"
        and fb["brake"] == 1.0,
        "outcome": sb["outcome"],
        "first_black_frame": fb,
        "collisions": sb["evaluator"]["collisions"],
    }
    for rec in obs["records"] + blk["records"]:
        assert rec.source in ("learned_model", "unavailable")
        if rec.source == "learned_model":
            assert rec.model_sha256 == info.sha256
    outcome["all_passed"] = all(c["passed"] for c in outcome["checks"].values())
    outcome["learned_sim_mode_enabled"] = bool(settings.learned_sim_mode_enabled)

    out_dir = Path(os.environ.get("DD_EVIDENCE_DIR", ROOT / "artifacts" / "learned_sim_validation"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    text = json.dumps(outcome, indent=2, default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o))
    (out_dir / f"learned_sim_validation_{stamp}.json").write_text(text, encoding="utf-8")
    (out_dir / "latest.json").write_text(text, encoding="utf-8")
    print(json.dumps({k: v["passed"] for k, v in outcome["checks"].items()}))

    if settings.learned_sim_mode_enabled:
        assert outcome["all_passed"], (
            "LEARNED_SIM_MODE_ENABLED=true but the learned model failed LiteRoad validation: "
            + json.dumps({k: v["passed"] for k, v in outcome["checks"].items()})
        )
