"""LiteRoad closed-loop scenario tests. Run inside the backend test image or with backend/.venv on a dev host.

Puts ``<root>/backend`` (repo) or ``/app/backend`` (image) on sys.path; identical relative layout. There is no
root-level conftest: the backend and inference packages are both named ``app`` and never share an interpreter.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

# Must be set before app.config.get_settings() is first called (it is cached).
os.environ.setdefault("APP_ENV", "test")

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

CONTRACTS_DIR = ROOT / "contracts"


@pytest.fixture
def run_session() -> Callable[..., tuple[Any, list[Any]]]:
    """Run a SimulationSession to a terminal state; returns (session, [TickResult, ...]) including start()."""
    from app.agents.runner import SessionConfig, SimulationSession
    from app.simulators.lite.scenarios import SCENARIOS

    def _run(
        scenario: str = "normal",
        seed: int = 1,
        start: str | None = None,
        dest: str | None = None,
        max_ticks: int = 2400,
        adapter: Any = None,
        encode_jpeg: bool = False,
        tick_cap: int | None = None,
    ) -> tuple[Any, list[Any]]:
        spec = SCENARIOS[scenario]
        cfg = SessionConfig(
            run_id=f"test-{scenario}-{seed}",
            start_node=start or spec.suite_route[0],
            destination_node=dest or spec.suite_route[1],
            scenario=scenario,
            seed=seed,
            max_ticks=max_ticks,
            encode_jpeg=encode_jpeg,
        )
        session = SimulationSession(cfg, adapter=adapter)
        results = [session.start()]
        while not results[-1].terminal and (tick_cap is None or len(results) <= tick_cap):
            results.append(session.tick())
        return session, results

    return _run
