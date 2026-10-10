"""Run report for LiteRoad simulation runs (contracts/run-report.schema.json, kind ``simulation``) + ``details``."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from app.schemas.report import Metric, RunReport

SIM_LIMITATIONS: tuple[str, ...] = (
    "Bounded synthetic LiteRoad world (flat ground, clear daylight, painted lanes, orange/white obstacles); results "
    "are not evidence about real roads or real vehicles.",
    "Perception source synthetic_rgb_cv is a classical colour/geometry pixel algorithm tuned to LiteRoad renders; it "
    "is not YOLOPv2 and inherits no learned-model accuracy claim.",
    "Obstacle distances are flat-ground inverse-perspective estimates from the static synthetic calibration, not "
    "metric depth; ttc_seconds is not computed.",
    "Ego localization is simulator telemetry (allowed runtime input); collisions, lateral deviation and route "
    "progress in this report are measured by the evaluator after each step and were never used for control.",
    "No traffic rules, junction right-of-way, other moving traffic or pedestrians are modelled.",
    "Simulator-only commands; nothing is sent to a physical vehicle.",
    "details.maneuver_timeline labels (STOP, BRAKE, SLOW_DOWN, ACCELERATE, CRUISE, TURN_LEFT, TURN_RIGHT, HOLD) "
    "describe the applied simulator command and ego telemetry with fixed thresholds (app.control.maneuver); they are a "
    "description of what the simulated vehicle did, not a prediction or a separate controller.",
)


def outcome_status(outcome: str) -> str:
    if outcome == "arrived":
        return "arrived"
    if outcome.startswith("stopped"):
        return "stopped"
    return "failed"


def build_simulation_report(
    *,
    run_id: str,
    outcome: str,
    perception_source: str,
    code_commit: str,
    metrics: dict[str, Metric],
    details: dict[str, Any],
    model_sha256: str | None = None,
    extra_limitations: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Validated RunReport dump plus ``details``. Raises ValueError for an invalid commit or model hash."""
    if perception_source == "synthetic_rgb_cv" and model_sha256 is not None:
        raise ValueError("synthetic_rgb_cv runs have no model checkpoint; model_sha256 must be None")
    try:
        report = RunReport(
            run_id=run_id,
            kind="simulation",
            status=outcome_status(outcome),  # type: ignore[arg-type]
            perception_source=perception_source,  # type: ignore[arg-type]
            code_commit=code_commit,
            model_sha256=model_sha256,
            input_manifest_sha256=None,
            generated_utc=datetime.now(UTC),
            metrics=metrics,
            limitations=[*SIM_LIMITATIONS, *extra_limitations],
            evaluation_truth_used=True,
            runtime_used_dynamic_ground_truth=False,
        )
    except ValidationError as exc:
        raise ValueError(f"run report validation failed: {exc.errors(include_url=False)}") from exc
    out = report.model_dump(mode="json")
    out["details"] = details
    return out
