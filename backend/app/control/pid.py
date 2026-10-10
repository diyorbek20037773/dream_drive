"""Speed PID producing a desired longitudinal acceleration (m/s^2), with anti-windup and output clamp."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpeedPID:
    kp: float = 1.5  # (m/s^2) per (m/s) of speed error
    ki: float = 0.4
    kd: float = 0.0
    integral_limit: float = 2.0  # bound on the integral state (m/s * s)
    out_min: float = -6.0  # strongest braking the vehicle can deliver
    out_max: float = 2.5  # strongest acceleration

    def __post_init__(self) -> None:
        self.integral = 0.0
        self._prev_error: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self._prev_error = None

    def update(self, target: float, measured: float, dt: float) -> float:
        """One step. Conditional integration: the integral only grows while the output is not saturated in the
        direction of the error (classic anti-windup), and it is additionally clamped to ``integral_limit``."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        error = target - measured
        deriv = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
        self._prev_error = error
        candidate = min(max(self.integral + error * dt, -self.integral_limit), self.integral_limit)
        unsat = self.kp * error + self.ki * candidate + self.kd * deriv
        saturating = (unsat > self.out_max and error > 0) or (unsat < self.out_min and error < 0)
        if not saturating:
            self.integral = candidate
        out = self.kp * error + self.ki * self.integral + self.kd * deriv
        return min(max(out, self.out_min), self.out_max)


def accel_to_pedals(
    accel_mps2: float, speed_mps: float, max_accel: float = 2.5, max_brake: float = 6.0, rolling_drag: float = 0.3
) -> tuple[float, float]:
    """Map a desired acceleration to mutually exclusive (throttle, brake) in [0, 1].

    Rolling drag (0.3 m/s^2 while moving) is compensated as feed-forward on throttle and counted as free
    deceleration on braking, matching the LiteRoad longitudinal model.
    """
    drag = rolling_drag if speed_mps > 0 else 0.0
    if accel_mps2 >= 0:
        return min(1.0, (accel_mps2 + drag) / max_accel), 0.0
    decel = -accel_mps2 - drag
    if decel <= 0:
        return min(1.0, max(0.0, (accel_mps2 + drag) / max_accel)), 0.0
    return 0.0, min(1.0, decel / max_brake)
