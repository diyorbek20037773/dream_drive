"""Maneuver classification: a deterministic *description* of what the simulated vehicle is doing.

This is not a controller and never feeds back into decisions. It runs once per tick after the command is final
(after the safety supervisor override and the actuator clamp) and reads only that applied command, the
controller's target speed and ego telemetry (allowed runtime input). Evaluator truth is never used.

Units: time s (simulated), speed m/s, acceleration m/s^2, yaw rate rad/s (+ = counter-clockwise = left),
steer normalized [-1, 1] (+1 = full left, docs/IMPLEMENTATION_SPEC.md section 4), throttle/brake [0, 1].

Signals per tick at simulated time t:
- v: ego telemetry speed.
- a: measured longitudinal acceleration (v(t) - v(t0)) / (t - t0) where t0 is the oldest telemetry sample inside the
  last ``WINDOW_S`` (0.5 s). While that window spans less than ``MIN_SPAN_S`` (first ticks of a run) the LiteRoad pedal
  model of the applied command is used instead: a = 2.5 throttle - 6.0 brake - 0.3 (rolling drag while moving).
- r: measured yaw rate wrap(yaw(t) - yaw(t0)) / (t - t0); with a too-short window the kinematic bicycle estimate
  r = v tan(0.5 steer) / 2.6 is used. ``signal_source`` reports which one was used.

Longitudinal label (first match wins):
1. ``holding``: no command was applied this tick (operator pause: physics frozen; or the initial frame before the first
   command).
2. ``stopped``: v <= ``STOPPED_MPS`` (0.10) and throttle == 0 (standstill, brake hold).
3. ``braking_to_stop``: brake > 0 and the target speed is 0 (stop state) or the safety supervisor overrode the command.
4. ``decelerating``: brake > 0, or a <= -``DECEL_ENTER_MPS2`` (0.25); once decelerating it stays while a < -0.10
   (``DECEL_EXIT_MPS2``) - hysteresis.
5. ``accelerating``: throttle > 0 and a >= ``ACCEL_ENTER_MPS2`` (0.25); stays while throttle > 0 and a > 0.10.
6. ``cruising``: otherwise.

Lateral label (only while v >= ``MOVING_MPS`` = 0.3, otherwise ``straight``):
- enter ``turning_left`` when r >= ``TURN_ENTER_RADPS`` (0.08) and steer >= ``STEER_AGREE`` (0.02);
  enter ``turning_right`` when r <= -0.08 and steer <= -0.02;
- leave a turn when |r| <= ``TURN_EXIT_RADPS`` (0.04) (hysteresis band 0.04..0.08 rad/s).

Primary display code (precedence): HOLD > STOP > BRAKE > TURN_LEFT/TURN_RIGHT > SLOW_DOWN > ACCELERATE > CRUISE.
HOLD, STOP and BRAKE are shown immediately; any other change must persist for ``CONFIRM_S`` (0.15 s of simulated time)
before the primary label switches, so short PID ripples do not flicker the display.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

# Codes (mirrored by the API literals ManeuverPrimary / ManeuverLongitudinal / ManeuverLateral; a test checks that).
PRIMARY_CODES: tuple[str, ...] = (
    "STOP",
    "BRAKE",
    "SLOW_DOWN",
    "ACCELERATE",
    "CRUISE",
    "TURN_LEFT",
    "TURN_RIGHT",
    "HOLD",
)
LONGITUDINAL_CODES: tuple[str, ...] = (
    "accelerating",
    "cruising",
    "decelerating",
    "braking_to_stop",
    "stopped",
    "holding",
)
LATERAL_CODES: tuple[str, ...] = ("straight", "turning_left", "turning_right")
SIGNAL_SOURCES: tuple[str, ...] = ("ego_telemetry_window", "command_model", "none")

WINDOW_S = 0.5
MIN_SPAN_S = 0.1
STOPPED_MPS = 0.10
MOVING_MPS = 0.3
ACCEL_ENTER_MPS2, ACCEL_EXIT_MPS2 = 0.25, 0.10
DECEL_ENTER_MPS2, DECEL_EXIT_MPS2 = 0.25, 0.10
TURN_ENTER_RADPS, TURN_EXIT_RADPS = 0.08, 0.04
STEER_AGREE = 0.02
CONFIRM_S = 0.15
# LiteRoad longitudinal / lateral model constants (docs/IMPLEMENTATION_SPEC.md section 4), used only for the
# command-model fallback while the telemetry window is too short.
MAX_ACCEL_MPS2 = 2.5
MAX_BRAKE_DECEL_MPS2 = 6.0
ROLLING_DRAG_MPS2 = 0.3
MAX_STEER_RAD = 0.5
WHEELBASE_M = 2.6

IMMEDIATE_PRIMARY = frozenset({"HOLD", "STOP", "BRAKE"})


@dataclass(frozen=True)
class AppliedCommand:
    """The command actually applied this tick (after safety override and clamp)."""

    steer: float
    throttle: float
    brake: float
    safety_override: bool = False


@dataclass(frozen=True)
class Maneuver:
    primary: str
    longitudinal: str
    lateral: str
    since_sim_time_s: float
    input_frame_id: int
    speed_mps: float
    accel_mps2: float | None
    yaw_rate_radps: float | None
    signal_source: str

    def to_wire(self) -> dict[str, Any]:
        """Run.live.maneuver shape."""

        def r(x: float | None) -> float | None:
            return round(x, 6) if x is not None and math.isfinite(x) else None

        return {
            "primary": self.primary,
            "longitudinal": self.longitudinal,
            "lateral": self.lateral,
            "since_sim_time_s": round(max(0.0, self.since_sim_time_s), 6),
            "input_frame_id": int(self.input_frame_id),
            "speed_mps": round(max(0.0, self.speed_mps), 6),
            "accel_mps2": r(self.accel_mps2),
            "yaw_rate_radps": r(self.yaw_rate_radps),
            "signal_source": self.signal_source,
        }


def primary_for(longitudinal: str, lateral: str) -> str:
    """Display precedence: hold/stop/brake first, then turning, then decel/accel/cruise."""
    if longitudinal == "holding":
        return "HOLD"
    if longitudinal == "stopped":
        return "STOP"
    if longitudinal == "braking_to_stop":
        return "BRAKE"
    if lateral == "turning_left":
        return "TURN_LEFT"
    if lateral == "turning_right":
        return "TURN_RIGHT"
    if longitudinal == "decelerating":
        return "SLOW_DOWN"
    if longitudinal == "accelerating":
        return "ACCELERATE"
    return "CRUISE"


class ManeuverClassifier:
    """Per-run classifier state (telemetry window, hysteresis labels, pending primary change)."""

    def __init__(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.dt = dt
        self._samples: deque[tuple[float, float, float]] = deque()  # (sim_time_s, speed_mps, yaw_rad)
        self._long = "holding"
        self._lat = "straight"
        self._primary: str | None = None
        self._since = 0.0
        self._pending: str | None = None
        self._pending_since = 0.0

    @property
    def primary(self) -> str | None:
        return self._primary

    def _add_sample(self, t: float, speed: float, yaw: float) -> None:
        if self._samples and t <= self._samples[-1][0] + 1e-9:
            # same simulated instant (paused / repeated frame): keep the newest telemetry for that instant
            self._samples[-1] = (self._samples[-1][0], speed, yaw)
        else:
            self._samples.append((t, speed, yaw))
        while len(self._samples) > 1 and t - self._samples[0][0] > WINDOW_S + 1e-9:
            self._samples.popleft()

    def _measured(self, t: float, speed: float, yaw: float) -> tuple[float, float] | None:
        if not self._samples:
            return None
        t0, v0, yaw0 = self._samples[0]
        span = t - t0
        if span < MIN_SPAN_S - 1e-9:
            return None
        return (speed - v0) / span, math.remainder(yaw - yaw0, math.tau) / span

    def update(
        self,
        sim_time_s: float,
        speed_mps: float,
        yaw_rad: float,
        command: AppliedCommand | None,
        target_speed_mps: float | None,
        input_frame_id: int,
    ) -> Maneuver:
        t = float(sim_time_s)
        v = max(0.0, float(speed_mps)) if math.isfinite(speed_mps) else 0.0
        yaw = float(yaw_rad) if math.isfinite(yaw_rad) else 0.0
        self._add_sample(t, v, yaw)

        accel: float | None
        yaw_rate: float | None
        if command is None:
            accel, yaw_rate, signal = None, None, "none"
            longitudinal, lateral = "holding", "straight"
        else:
            measured = self._measured(t, v, yaw)
            if measured is not None:
                accel, yaw_rate = measured
                signal = "ego_telemetry_window"
            else:
                drag = ROLLING_DRAG_MPS2 if v > 0 else 0.0
                accel = command.throttle * MAX_ACCEL_MPS2 - command.brake * MAX_BRAKE_DECEL_MPS2 - drag
                yaw_rate = v * math.tan(command.steer * MAX_STEER_RAD) / WHEELBASE_M
                signal = "command_model"
            longitudinal = self._classify_longitudinal(v, accel, command, target_speed_mps)
            lateral = self._classify_lateral(v, yaw_rate, command.steer)
        self._long, self._lat = longitudinal, lateral

        candidate = primary_for(longitudinal, lateral)
        if self._primary is None or candidate in IMMEDIATE_PRIMARY:
            if candidate != self._primary:
                self._primary, self._since = candidate, t
            self._pending = None
        elif candidate == self._primary:
            self._pending = None
        else:
            if self._pending != candidate:
                self._pending, self._pending_since = candidate, t
            if t - self._pending_since + self.dt >= CONFIRM_S - 1e-9:
                self._primary, self._since = candidate, self._pending_since
                self._pending = None
        assert self._primary is not None
        return Maneuver(
            primary=self._primary,
            longitudinal=longitudinal,
            lateral=lateral,
            since_sim_time_s=self._since,
            input_frame_id=int(input_frame_id),
            speed_mps=v,
            accel_mps2=accel,
            yaw_rate_radps=yaw_rate,
            signal_source=signal,
        )

    def _classify_longitudinal(self, v: float, a: float, cmd: AppliedCommand, target_speed_mps: float | None) -> str:
        if v <= STOPPED_MPS and cmd.throttle <= 0.0:
            return "stopped"
        if cmd.brake > 0.0 and (cmd.safety_override or (target_speed_mps is not None and target_speed_mps <= 0.0)):
            return "braking_to_stop"
        prev = self._long
        if cmd.brake > 0.0 or a <= -DECEL_ENTER_MPS2 or (prev == "decelerating" and a < -DECEL_EXIT_MPS2):
            return "decelerating"
        if cmd.throttle > 0.0 and (a >= ACCEL_ENTER_MPS2 or (prev == "accelerating" and a > ACCEL_EXIT_MPS2)):
            return "accelerating"
        return "cruising"

    def _classify_lateral(self, v: float, r: float, steer: float) -> str:
        if v < MOVING_MPS:
            return "straight"
        prev = self._lat
        if r >= TURN_ENTER_RADPS and steer >= STEER_AGREE:
            return "turning_left"
        if r <= -TURN_ENTER_RADPS and steer <= -STEER_AGREE:
            return "turning_right"
        if prev == "turning_left" and r > TURN_EXIT_RADPS:
            return "turning_left"
        if prev == "turning_right" and r < -TURN_EXIT_RADPS:
            return "turning_right"
        return "straight"
