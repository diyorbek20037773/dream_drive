"""LiteRoad scenario catalogue and deterministic per-seed scene setup.

Scene setup (initial pose jitter, obstacle placement, camera fault schedule) is *scene generation*: it configures the
private simulator world. Nothing returned here is handed to perception, FSM or controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.navigation.route import Route


@dataclass(frozen=True)
class ScenarioSpec:
    id: str
    label_en: str
    label_uz: str
    available: bool
    reason: str | None
    suite_route: tuple[str, str]  # start, destination used by the evaluation suite
    camera_fault: str | None = None  # "black" | "freeze"
    fault_start_s: float | None = None
    obstacle_ahead_m: float | None = None  # nominal distance along the route from the start pose
    obstacle_jitter_m: float = 0.0
    obstacle_lateral_jitter_m: float = 0.0
    pose_lateral_jitter_m: float = 0.3
    pose_heading_jitter_rad: float = 0.04


SCENARIOS: dict[str, ScenarioSpec] = {
    s.id: s
    for s in (
        ScenarioSpec("normal", "Normal route", "Oddiy marshrut", True, None, ("A", "C")),
        ScenarioSpec("bend", "Lane bends (S-curve)", "Yo‘l burilishlari (S-egri)", True, None, ("D", "I")),
        ScenarioSpec(
            "obstacle_stop",
            "Visible obstacle: controlled stop",
            "Ko‘rinadigan to‘siq: boshqariladigan to‘xtash",
            True,
            None,
            ("A", "C"),
            obstacle_ahead_m=25.0,
            obstacle_jitter_m=3.0,
            obstacle_lateral_jitter_m=0.3,
        ),
        ScenarioSpec(
            "camera_blackout",
            "Camera blackout at t=4 s",
            "Kamera t=4 s da o‘chadi",
            True,
            None,
            ("A", "C"),
            camera_fault="black",
            fault_start_s=4.0,
        ),
        ScenarioSpec(
            "stale_frame",
            "Camera frame freezes at t=4 s",
            "Kadr t=4 s da qotib qoladi",
            True,
            None,
            ("A", "C"),
            camera_fault="freeze",
            fault_start_s=4.0,
        ),
        ScenarioSpec(
            "pause_resume",
            "Pause/resume test helper",
            "Pauza/davom ettirish test yordamchisi",
            False,
            "Test-only helper scenario; pause and resume are available on every run through the run controls.",
            ("A", "C"),
        ),
    )
}

SUITE_SCENARIOS: tuple[str, ...] = ("normal", "bend", "obstacle_stop", "camera_blackout", "stale_frame")


@dataclass(frozen=True)
class ScenarioSetup:
    """Resolved per-seed scene configuration (simulator/evaluator private)."""

    scenario: str
    seed: int
    start_pose: tuple[float, float, float]  # x, y, yaw
    obstacles: tuple[tuple[float, float, float], ...]  # x, y, yaw of visible scenario obstacles
    obstacle_s_m: tuple[float, ...]  # their arc length along the route
    camera_fault: str | None
    fault_start_s: float | None


def resolve_setup(spec: ScenarioSpec, seed: int, route: Route) -> ScenarioSetup:
    """Deterministic scene parameters for (scenario, seed, route)."""
    rng = np.random.default_rng([seed, sum(map(ord, spec.id))])
    lat = float(rng.uniform(-spec.pose_lateral_jitter_m, spec.pose_lateral_jitter_m))
    dh = float(rng.uniform(-spec.pose_heading_jitter_rad, spec.pose_heading_jitter_rad))
    x0, y0 = route.point_at(0.0)
    h0 = route.heading_at(0.0)
    start = (x0 - lat * math.sin(h0), y0 + lat * math.cos(h0), math.remainder(h0 + dh, math.tau))
    obstacles: list[tuple[float, float, float]] = []
    s_list: list[float] = []
    if spec.obstacle_ahead_m is not None:
        s = spec.obstacle_ahead_m + float(rng.uniform(-spec.obstacle_jitter_m, spec.obstacle_jitter_m))
        s = min(s, route.length_m - 2.0)
        off = float(rng.uniform(-spec.obstacle_lateral_jitter_m, spec.obstacle_lateral_jitter_m))
        obstacles.append(point_on_route(route, s, off))
        s_list.append(s)
    return ScenarioSetup(spec.id, seed, start, tuple(obstacles), tuple(s_list), spec.camera_fault, spec.fault_start_s)


def point_on_route(route: Route, s_m: float, lateral_offset_m: float) -> tuple[float, float, float]:
    """World pose at arc length ``s_m`` shifted ``lateral_offset_m`` to the left (+) of the travel direction."""
    x, y = route.point_at(s_m)
    h = route.heading_at(s_m)
    return x - lateral_offset_m * math.sin(h), y + lateral_offset_m * math.cos(h), h
