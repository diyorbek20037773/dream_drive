"""Simulator adapters (LiteRoad default, CARLA optional/unavailable) and the API catalogue helper.

Imports are lazy so that importing ``app.simulators.base`` stays cheap and free of cycles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.config import Settings

LEARNED_SIM_DISABLED_REASON = (
    "Learned-model simulator perception is disabled until it passes the LiteRoad validation scenarios "
    "(lane, obstacle, missing camera); synthetic renders are outside the YOLOPv2 training domain."
)


def road_graph_wire(start: str | None = None) -> dict[str, Any]:
    """LiteRoad topology in the API ``road_graph`` shape (``from``/``to`` keys, reachability from ``start``)."""
    from app.simulators.lite.graph import DEFAULT_START, build_town_graph, reachable_from

    graph = build_town_graph()
    reach = reachable_from(graph, start or DEFAULT_START)
    return {
        "nodes": [
            {
                "id": n.id,
                "x": n.x,
                "y": n.y,
                "label_uz": n.label_uz,
                "label_en": n.label_en,
                "reachable_from_default": n.id in reach,
            }
            for n in graph.nodes
        ],
        "edges": [
            {"id": e.id, "from": e.a, "to": e.b, "polyline": [[float(x), float(y)] for x, y in e.polyline]}
            for e in graph.edges
        ],
        "lane_width_m": graph.lane_width_m,
        "units": "m",
        "frame": "ENU x-east y-north yaw CCW",
    }


def simulator_catalog(settings: Settings | None = None) -> list[dict[str, Any]]:
    """``Simulator`` objects (docs/IMPLEMENTATION_SPEC.md 3.4) for GET /api/v1/simulators."""
    from app.config import get_settings
    from app.simulators.carla_adapter import carla_availability
    from app.simulators.lite.graph import DEFAULT_START
    from app.simulators.lite.scenarios import SCENARIOS

    settings = settings or get_settings()
    learned_ok = bool(settings.learned_sim_mode_enabled)
    lite = {
        "id": "lite",
        "name": "LiteRoad (CPU synthetic simulator)",
        "available": True,
        "reason": None,
        "perception_modes": [
            {"id": "synthetic_rgb_cv", "available": True, "reason": None},
            {
                "id": "learned_model",
                "available": learned_ok,
                "reason": None if learned_ok else LEARNED_SIM_DISABLED_REASON,
            },
        ],
        "scenarios": [{"id": s.id, "available": s.available, "reason": s.reason} for s in SCENARIOS.values()],
        "road_graph": road_graph_wire(DEFAULT_START),
        "default_start": DEFAULT_START,
    }
    carla_ok, carla_reason = carla_availability(settings)
    carla = {
        "id": "carla",
        "name": "CARLA 0.9.16 (optional)",
        "available": carla_ok,
        "reason": None if carla_ok else carla_reason,
        "perception_modes": [
            {"id": "synthetic_rgb_cv", "available": False, "reason": "synthetic_rgb_cv is tuned to LiteRoad renders"},
            {"id": "learned_model", "available": False, "reason": carla_reason},
        ],
        "scenarios": [],
        "road_graph": None,
        "default_start": None,
    }
    return [lite, carla]
