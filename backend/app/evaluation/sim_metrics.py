"""Aggregate LiteRoad scenario-run summaries into Evaluation metrics (measured values only, with sample counts)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from app.schemas.report import Metric

EXPECTED_OUTCOME = {
    "normal": "arrived",
    "bend": "arrived",
    "obstacle_stop": "stopped_before_obstacle",
    "camera_blackout": "stopped_camera_invalid",
    "stale_frame": "stopped_frame_stale",
}


def expected_outcome_met(run: dict[str, Any]) -> bool:
    exp = EXPECTED_OUTCOME.get(run["scenario"])
    return exp is not None and run.get("outcome") == exp and run["evaluator"]["collisions"] == 0


def _rate(runs: list[dict[str, Any]], scenario: str, note: str) -> Metric:
    sel = [r for r in runs if r["scenario"] == scenario]
    if not sel:
        return Metric.missing("ratio", f"no {scenario} runs in this suite")
    ok = sum(1 for r in sel if expected_outcome_met(r))
    return Metric.measured(round(ok / len(sel), 4), "ratio", len(sel), note)


def _pooled(values: Iterable[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None]


def suite_metrics(runs: list[dict[str, Any]], tick_ms: list[float], deviations: list[float]) -> dict[str, Metric]:
    """``runs`` are SimulationSession.summary() dicts; ``tick_ms``/``deviations`` pooled per-tick samples of the
    route-following scenarios (normal, bend)."""
    m: dict[str, Metric] = {}
    n = len(runs)
    m["normal_arrival_rate"] = _rate(runs, "normal", "arrived within the tick budget with zero collisions")
    m["bend_arrival_rate"] = _rate(runs, "bend", "arrived through the curved edges with zero collisions")
    m["obstacle_controlled_stop_rate"] = _rate(
        runs, "obstacle_stop", "stopped before the visible obstacle (stopped_before_obstacle) with zero collisions"
    )
    m["camera_blackout_safety_stop_rate"] = _rate(
        runs, "camera_blackout", "safety stop (stopped_camera_invalid) after the camera went black"
    )
    m["stale_frame_safety_stop_rate"] = _rate(runs, "stale_frame", "safety stop (stopped_frame_stale) on frozen frames")
    if n:
        m["expected_outcome_rate"] = Metric.measured(
            round(sum(expected_outcome_met(r) for r in runs) / n, 4), "ratio", n, "all suite runs"
        )
        m["collisions_total"] = Metric.measured(
            sum(r["evaluator"]["collisions"] for r in runs),
            "count",
            n,
            "evaluator truth (ego footprint 4.5 x 1.8 m vs obstacle boxes)",
        )
        m["safety_overrides_total"] = Metric.measured(sum(r["commands"]["safety_overrides"] for r in runs), "count", n)
        m["ticks_mean"] = Metric.measured(round(float(np.mean([r["ticks"] for r in runs])), 2), "ticks", n)
        sim_total = float(sum(r["sim_time_s"] for r in runs))
        wall_total = float(sum(r["wall_time_s"] for r in runs))
        m["sim_time_total_s"] = Metric.measured(round(sim_total, 3), "s", n)
        m["wall_time_total_s"] = Metric.measured(round(wall_total, 3), "s", n, "headless, no real-time pacing")
        if wall_total > 0:
            m["realtime_factor_headless"] = Metric.measured(
                round(sim_total / wall_total, 3),
                "ratio",
                n,
                "sim time / wall time without pacing (live runs are paced to <= 1.0)",
            )
        false_arrivals = sum(1 for r in runs if r["scenario"] == "obstacle_stop" and r.get("outcome") == "arrived")
        n_obs = sum(1 for r in runs if r["scenario"] == "obstacle_stop")
        if n_obs:
            m["obstacle_false_arrival_count"] = Metric.measured(false_arrivals, "count", n_obs)
    else:
        m["expected_outcome_rate"] = Metric.missing("ratio", "no runs executed")
    if deviations:
        arr = np.asarray(deviations)
        m["lateral_deviation_mean_m"] = Metric.measured(
            round(float(arr.mean()), 4), "m", arr.size, "normal + bend runs, rear axle vs truth lane centre"
        )
        m["lateral_deviation_p95_m"] = Metric.measured(round(float(np.percentile(arr, 95)), 4), "m", arr.size)
        m["lateral_deviation_max_m"] = Metric.measured(round(float(arr.max()), 4), "m", arr.size)
    else:
        m["lateral_deviation_mean_m"] = Metric.missing("m", "no route-following runs")
    gaps = _pooled(r["evaluator"]["final_obstacle_gap_m"] for r in runs if r["scenario"] == "obstacle_stop")
    if gaps:
        m["obstacle_stop_gap_mean_m"] = Metric.measured(
            round(float(np.mean(gaps)), 3), "m", len(gaps), "final ego-front-to-obstacle footprint gap"
        )
        m["obstacle_stop_gap_min_m"] = Metric.measured(round(float(np.min(gaps)), 3), "m", len(gaps))
    else:
        m["obstacle_stop_gap_mean_m"] = Metric.missing("m", "no obstacle_stop runs")
    reactions = []
    for r in runs:
        if r["scenario"] in ("camera_blackout", "stale_frame") and r["commands"]["first_safety_override"]:
            reactions.append(r["commands"]["first_safety_override"]["sim_time_s"])
    if reactions:
        m["camera_fault_first_override_sim_time_s_mean"] = Metric.measured(
            round(float(np.mean(reactions)), 3),
            "s",
            len(reactions),
            "camera fault injected at t=4.0 s; frozen frames carry the last good timestamp (3.95 s)",
        )
    if tick_ms:
        arr = np.asarray(tick_ms)
        m["tick_wall_ms_p50"] = Metric.measured(
            round(float(np.percentile(arr, 50)), 3),
            "ms",
            arr.size,
            "render + perception + control + JPEG-free headless tick",
        )
        m["tick_wall_ms_p95"] = Metric.measured(round(float(np.percentile(arr, 95)), 3), "ms", arr.size)
    m["real_road_driving_accuracy"] = Metric.missing(
        "ratio", "LiteRoad is synthetic; no claim about real roads or vehicles", status="unsupported"
    )
    m["learned_model_sim_performance"] = Metric.missing(
        "ratio", "learned-model simulator mode is evaluated separately (test_learned_sim_validation)"
    )
    return m
