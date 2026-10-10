"""Simulator catalogue for /simulators, /capabilities and run validation.

LiteRoad comes from ``app.agents.runner.SimulationSession`` (static road graph, scenario list, route planner).
The learned-model perception mode is offered only when LEARNED_SIM_MODE_ENABLED is set (scenario validation passed)
and the inference model is ready. CARLA is listed but unavailable with a reason: no CARLA adapter is part of this
build, so nothing would execute a CARLA run.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, get_args

from app.api.services import ModelStatus
from app.config import Settings
from app.schemas.api import (
    PerceptionModeInfo,
    RoadGraphInfo,
    ScenarioId,
    ScenarioInfo,
    Simulator,
)

logger = logging.getLogger("dreamdrive.api.catalog")

_SCENARIO_IDS = set(get_args(ScenarioId.__value__))


def load_runner() -> Any:
    """The runner module (imported lazily: it pulls in OpenCV/NumPy simulator code)."""
    from app.agents import runner

    return runner


def default_start(runner: Any) -> str | None:
    try:
        from app.simulators.lite.graph import DEFAULT_START

        return str(DEFAULT_START)
    except ImportError:
        graph = runner.SimulationSession.road_graph()
        return graph.nodes[0].id if graph.nodes else None


def _routable(session_cls: Any, start: str, dest: str) -> bool:
    try:
        session_cls.plan_route(start, dest)
    except Exception:  # noqa: BLE001 - RouteUnreachable (or any planner refusal) means not reachable
        return False
    return True


@lru_cache(maxsize=1)
def _lite_static() -> tuple[RoadGraphInfo, str | None, tuple[tuple[str, bool, str | None], ...]]:
    runner = load_runner()
    session_cls = runner.SimulationSession
    graph = session_cls.road_graph()
    start = default_start(runner)
    reachable: set[str] = set()
    if start is not None:
        for node in graph.nodes:
            if node.id == start:
                reachable.add(node.id)
                continue
            if _routable(session_cls, start, node.id):
                reachable.add(node.id)
    scenarios = tuple((str(s["id"]), bool(s.get("available", False)), s.get("reason")) for s in session_cls.scenarios())
    return RoadGraphInfo.from_graph(graph, reachable), start, scenarios


def learned_mode_reason(settings: Settings, model: ModelStatus) -> str | None:
    if not settings.learned_sim_mode_enabled:
        return "learned_mode_not_validated"
    if not model.ready:
        return "model_not_ready"
    return None


def lite_simulator(settings: Settings, model: ModelStatus) -> Simulator:
    try:
        graph, start, scenarios = _lite_static()
    except Exception:  # noqa: BLE001 - visible as unavailable, never hidden
        logger.exception("LiteRoad catalogue unavailable", extra={"code": "simulator_unavailable"})
        return Simulator(
            id="lite",
            name="LiteRoad",
            available=False,
            reason="simulator_module_unavailable",
            perception_modes=[
                PerceptionModeInfo(id="synthetic_rgb_cv", available=False, reason="simulator_unavailable"),
                PerceptionModeInfo(id="learned_model", available=False, reason="simulator_unavailable"),
            ],
            scenarios=[],
            road_graph=None,
            default_start=None,
        )
    learned_reason = learned_mode_reason(settings, model)
    return Simulator(
        id="lite",
        name="LiteRoad",
        available=True,
        reason=None,
        perception_modes=[
            PerceptionModeInfo(id="synthetic_rgb_cv", available=True, reason=None),
            PerceptionModeInfo(id="learned_model", available=learned_reason is None, reason=learned_reason),
        ],
        scenarios=[
            ScenarioInfo.model_validate(
                {"id": sid, "available": available, "reason": None if available else (reason or "unavailable")}
            )
            for sid, available, reason in scenarios
            if sid in _SCENARIO_IDS
        ],
        road_graph=graph,
        default_start=start,
    )


def carla_reason(settings: Settings) -> str:
    if not settings.carla_host:
        return "carla_host_not_configured"
    return "carla_adapter_not_available"


def carla_simulator(settings: Settings) -> Simulator:
    reason = carla_reason(settings)
    return Simulator(
        id="carla",
        name="CARLA",
        available=False,
        reason=reason,
        perception_modes=[
            PerceptionModeInfo(id="synthetic_rgb_cv", available=False, reason="simulator_unavailable"),
            PerceptionModeInfo(id="learned_model", available=False, reason="simulator_unavailable"),
        ],
        scenarios=[],
        road_graph=None,
        default_start=None,
    )


def simulators(settings: Settings, model: ModelStatus) -> list[Simulator]:
    return [lite_simulator(settings, model), carla_simulator(settings)]


def scenario_availability() -> dict[str, tuple[bool, str | None]]:
    _graph, _start, scenarios = _lite_static()
    return {sid: (available, reason) for sid, available, reason in scenarios}


def node_ids() -> set[str]:
    graph, _start, _scenarios = _lite_static()
    return {n.id for n in graph.nodes}


def reset_cache() -> None:
    _lite_static.cache_clear()
