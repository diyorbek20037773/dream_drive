"""Wire contracts of the simulation loop and the synthetic scenario-suite evaluation artifact."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from app.agents.runner import SessionConfig, SimulationSession

CONTRACTS_DIR = Path(__file__).resolve().parents[2] / "contracts"

pytestmark = pytest.mark.timeout(900)

LIVE_KEYS = {
    "frame_id",
    "sim_time_s",
    "wall_time_s",
    "realtime_factor",
    "ego",
    "decision",
    "command",
    "perception",
    "route_progress",
    "acquisition_age_s",
    "maneuver",
    "decision_basis",
}
LIVE_PERCEPTION_KEYS = {
    "source",
    "quality_status",
    "lane_status",
    "image_lane_offset_normalized",
    "image_obstacle_cue",
    "obstacle_boxes",
    "lane_points",
    "frame_width",
    "frame_height",
}
COMMIT = "0123456789abcdef0123456789abcdef01234567"  # test input only; never written to a real artifact store


def _schema(name: str) -> dict:
    return json.loads((CONTRACTS_DIR / name).read_text(encoding="utf-8"))


def _validator(name: str) -> jsonschema.Validator:
    schema = _schema(name)
    cls = jsonschema.validators.validator_for(schema)
    return cls(schema, format_checker=cls.FORMAT_CHECKER)


def test_world_state_and_live_shapes() -> None:
    ws_validator = _validator("world-state.schema.json")
    from app.schemas.api import RunLive

    s = SimulationSession(
        SessionConfig(run_id="contract", start_node="A", destination_node="C", scenario="obstacle_stop", seed=1)
    )
    results = [s.start()]
    while not results[-1].terminal:
        results.append(s.tick())
    for r in results[:: max(1, len(results) // 25)] + [results[-1]]:
        ws_validator.validate(r.world_state.model_dump(mode="json"))
        assert set(r.live) == LIVE_KEYS
        assert set(r.live["perception"]) == LIVE_PERCEPTION_KEYS
        RunLive.model_validate(r.live)
        json.dumps(r.live)  # JSON-serialisable
        assert r.live["ego"]["source"] == "ego_telemetry"
        assert r.live["perception"]["source"] == "synthetic_rgb_cv"
        assert 0.0 <= r.live["route_progress"] <= 1.0
        if r.live["command"] is not None:
            assert r.live["command"]["source"] == "dreamdrive_controller"
            assert r.live["command"]["input_frame_id"] == r.frame_id
            assert r.live["command"]["reason_codes"]
    report = s.report(COMMIT, 3.0)
    details = report.pop("details")
    _validator("run-report.schema.json").validate(report)
    assert report["kind"] == "simulation" and report["status"] == "stopped"
    assert report["evaluation_truth_used"] is True and report["runtime_used_dynamic_ground_truth"] is False
    assert report["model_sha256"] is None and report["perception_source"] == "synthetic_rgb_cv"
    assert report["metrics"]["collisions"]["value"] == 0
    assert report["metrics"]["ttc_seconds"]["status"] == "unsupported"
    assert details["outcome"] == "stopped_before_obstacle"
    assert details["decision_counts"]["controlled_stop"] > 0
    from app.schemas.api import RunReportResponse

    RunReportResponse.model_validate({**report, "details": details})


def test_simulator_catalog_validates() -> None:
    from app.schemas.api import Simulator
    from app.simulators import simulator_catalog

    cat = simulator_catalog()
    sims = [Simulator.model_validate(c) for c in cat]
    lite, carla = sims
    assert lite.available and lite.default_start == "A"
    assert {m.id: m.available for m in lite.perception_modes} == {"synthetic_rgb_cv": True, "learned_model": False}
    nodes = {n.id: n for n in lite.road_graph.nodes}
    assert nodes["Q"].reachable_from_default is False and nodes["C"].reachable_from_default is True
    assert not carla.available and carla.reason
    edge = cat[0]["road_graph"]["edges"][0]
    assert {"id", "from", "to", "polyline"} <= set(edge)


def test_scenario_suite_writes_evaluation(tmp_path: Path) -> None:
    from app.evaluation.registry import list_evaluations, load_evaluation, load_evaluation_details
    from app.evaluation.scenario_suite import run_scenario_suite
    from app.schemas.api import Evaluation

    progress: list[tuple[int, int, str]] = []
    doc = run_scenario_suite([5], tmp_path, COMMIT, lambda d, t, label: progress.append((d, t, label)))
    assert progress[-1][:2] == (5, 5)
    assert doc["kind"] == "synthetic_scenario_suite" and doc["source"] == "synthetic_rgb_cv"
    assert doc["sample_count"] == 5 and doc["state"] == "succeeded" and doc["model_sha256"] is None
    m = doc["metrics"]
    for name in (
        "normal_arrival_rate",
        "bend_arrival_rate",
        "obstacle_controlled_stop_rate",
        "camera_blackout_safety_stop_rate",
        "stale_frame_safety_stop_rate",
        "collisions_total",
        "lateral_deviation_mean_m",
        "realtime_factor_headless",
        "tick_wall_ms_p50",
        "obstacle_stop_gap_mean_m",
    ):
        assert m[name]["status"] == "measured" and m[name]["sample_count"] >= 1, name
    assert m["expected_outcome_rate"]["value"] == 1.0
    assert m["collisions_total"]["value"] == 0
    assert m["real_road_driving_accuracy"]["status"] == "unsupported"
    assert m["real_road_driving_accuracy"]["value"] is None
    listed = list_evaluations(tmp_path)
    assert len(listed) == 1 and "details" not in listed[0]
    loaded = load_evaluation(tmp_path, doc["id"])
    Evaluation.model_validate(loaded)
    assert load_evaluation(tmp_path, "../../etc/passwd") is None
    details = load_evaluation_details(tmp_path, doc["id"])
    assert len(details["runs"]) == 5 and all(r["expected_outcome_met"] for r in details["runs"])
    unknown = run_scenario_suite([5], tmp_path, "not-a-commit", scenarios=("camera_blackout",))
    assert unknown["code_commit"] is None, "an unknown commit is recorded as null, never invented"
