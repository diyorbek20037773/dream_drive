"""Synthetic LiteRoad scenario suite: normal, bend, obstacle_stop, camera_blackout, stale_frame for each seed.

Runs headless (no real-time pacing, JPEG encoding skipped), measures every run with the evaluator and writes an
``Evaluation`` JSON (kind ``synthetic_scenario_suite``, source ``synthetic_rgb_cv``) to ``out_dir/<id>.json``.
"""

from __future__ import annotations

import os
import platform
import re
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.agents.runner import SessionConfig, SimulationSession
from app.evaluation.datasets import LITEROAD_SYNTHETIC_SCENARIOS
from app.evaluation.registry import write_evaluation
from app.evaluation.sim_metrics import EXPECTED_OUTCOME, expected_outcome_met, suite_metrics
from app.reporting.sim_report import SIM_LIMITATIONS
from app.simulators.lite.scenarios import SCENARIOS, SUITE_SCENARIOS

ProgressCb = Callable[[int, int, str], None]
_COMMIT = re.compile(r"^[a-f0-9]{40,64}$")
SUITE_TICK_BUDGET = 1200  # per run; normal/bend arrivals measured at < 900 ticks (docs/SIMULATION_EVIDENCE.md)


def run_single(
    scenario: str, seed: int, run_id: str, max_ticks: int = SUITE_TICK_BUDGET
) -> tuple[dict[str, Any], list[float], list[float]]:
    spec = SCENARIOS[scenario]
    start, dest = spec.suite_route
    session = SimulationSession(
        SessionConfig(
            run_id=run_id,
            start_node=start,
            destination_node=dest,
            scenario=scenario,
            seed=seed,
            max_ticks=max_ticks,
            encode_jpeg=False,
        )
    )
    t0 = time.monotonic()
    result = session.start()
    while not result.terminal:
        result = session.tick()
        session.note_wall_time(time.monotonic() - t0)
    summary = session.summary(time.monotonic() - t0)
    summary["expected_outcome"] = EXPECTED_OUTCOME[scenario]
    summary["expected_outcome_met"] = expected_outcome_met(summary)
    return summary, session.tick_wall_ms_samples, session.lateral_deviation_samples


def run_scenario_suite(
    seeds: list[int],
    out_dir: Path,
    code_commit: str,
    progress_cb: ProgressCb | None = None,
    *,
    eval_id: str | None = None,
    scenarios: tuple[str, ...] = SUITE_SCENARIOS,
) -> dict[str, Any]:
    """Run the suite and persist the Evaluation; returns the Evaluation dict (including ``details``).

    ``eval_id`` lets the job runner reuse the database id; otherwise a new uuid4 hex id is generated.
    An invalid / unknown ``code_commit`` is recorded as null rather than invented.
    """
    if not seeds:
        raise ValueError("at least one seed is required")
    for sc in scenarios:
        if sc not in SCENARIOS:
            raise ValueError(f"unknown scenario {sc!r}")
    eval_id = eval_id or uuid.uuid4().hex
    created = datetime.now(UTC)
    total = len(seeds) * len(scenarios)
    runs: list[dict[str, Any]] = []
    tick_ms: list[float] = []
    deviations: list[float] = []
    done = 0
    for seed in seeds:
        for sc in scenarios:
            summary, ticks, devs = run_single(sc, int(seed), f"{eval_id}-{sc}-{seed}")
            runs.append(summary)
            tick_ms.extend(ticks)
            if sc in ("normal", "bend"):
                deviations.extend(devs)
            done += 1
            if progress_cb is not None:
                progress_cb(done, total, f"{sc}:{seed}")
    metrics = suite_metrics(runs, tick_ms, deviations)
    doc: dict[str, Any] = {
        "id": eval_id,
        "kind": "synthetic_scenario_suite",
        "state": "succeeded",
        "created_utc": created.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "source": "synthetic_rgb_cv",
        "sample_count": len(runs),
        "dataset": dict(LITEROAD_SYNTHETIC_SCENARIOS),
        "metrics": {k: v.model_dump(mode="json") for k, v in metrics.items()},
        "limitations": [
            *SIM_LIMITATIONS,
            "Each scenario uses one fixed route; seeds vary the initial pose (+-0.3 m, +-0.04 rad) and obstacle "
            "placement (+-3 m along, +-0.3 m across). Wider operating domains are not evaluated.",
            "Wall-clock timings depend on the host CPU and concurrent load; they are measurements, not guarantees.",
        ],
        "code_commit": code_commit if _COMMIT.match(code_commit or "") else None,
        "model_sha256": None,
        "seeds": [int(s) for s in seeds],
        "error_code": None,
        "details": {
            "runs": [
                {k: v for k, v in r.items() if k != "route"} | {"route_length_m": r["route"]["length_m"]} for r in runs
            ],
            "scenarios": list(scenarios),
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
                "python": platform.python_version(),
            },
        },
    }
    write_evaluation(Path(out_dir), doc)
    return doc
