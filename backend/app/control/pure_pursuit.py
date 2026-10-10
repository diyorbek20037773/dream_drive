"""Pure Pursuit path tracking on the lane-centre route (rear-axle reference, kinematic bicycle geometry)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.navigation.route import Route

WHEELBASE_M = 2.6
MAX_STEER_RAD = 0.5


def lookahead_distance(speed_mps: float) -> float:
    """clamp(2 + 0.6 v, 3, 8) metres."""
    return min(max(2.0 + 0.6 * max(speed_mps, 0.0), 3.0), 8.0)


@dataclass(frozen=True)
class PursuitTarget:
    s_m: float  # arc length of the target point on the route
    lookahead_m: float
    x_fwd: float  # target in the vehicle frame: forward (m)
    y_left: float  # target in the vehicle frame: left (m)


def target_point(route: Route, ego_x: float, ego_y: float, ego_yaw: float, s_ego: float, speed: float) -> PursuitTarget:
    """Route point ``lookahead`` metres ahead of the projected ego position, expressed in the vehicle frame.
    Beyond the route end the final heading is extended so the target never collapses onto the vehicle."""
    ld = lookahead_distance(speed)
    s_t = s_ego + ld
    if s_t <= route.length_m:
        tx, ty = route.point_at(s_t)
    else:
        ex, ey = route.point_at(route.length_m)
        h = route.heading_at(route.length_m)
        extra = s_t - route.length_m
        tx, ty = ex + extra * math.cos(h), ey + extra * math.sin(h)
    dx, dy = tx - ego_x, ty - ego_y
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    return PursuitTarget(s_m=s_t, lookahead_m=ld, x_fwd=dx * c + dy * s, y_left=-dx * s + dy * c)


def pure_pursuit_steer(
    x_fwd: float, y_left: float, wheelbase: float = WHEELBASE_M, max_steer_rad: float = MAX_STEER_RAD
) -> float:
    """Normalized steering command (+ = left) for a target point in the vehicle frame. Not clamped here:
    ``app.control.command.finalize_command`` clamps to [-1, 1] and records CONTROL_CLAMPED."""
    ld2 = x_fwd * x_fwd + y_left * y_left
    if ld2 < 1e-6:
        return 0.0
    curvature = 2.0 * y_left / ld2
    return math.atan(wheelbase * curvature) / max_steer_rad
