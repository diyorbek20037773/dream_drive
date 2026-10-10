"""Final actuator command shaping: NaN/inf rejection, bounds clamp, exclusive throttle/brake, reason codes."""

from __future__ import annotations

import math
from collections.abc import Iterable

from app.schemas.reason_codes import ReasonCode
from app.simulators.base import ControlCommand

STEER_BOUNDS = (-1.0, 1.0)
PEDAL_BOUNDS = (0.0, 1.0)
_EPS = 1e-9


def _dedupe(codes: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    for c in codes:
        if c not in out:
            out.append(c)
    return tuple(out)


def finalize_command(
    steer: float,
    throttle: float,
    brake: float,
    input_frame_id: int,
    reason_codes: Iterable[str],
    safety_override: bool = False,
) -> ControlCommand:
    """Return a command that is always inside the actuator bounds.

    - any non-finite input -> steer 0, throttle 0, full brake, ``CONTROL_NAN_REJECTED``, safety override
    - values outside bounds are clamped -> ``CONTROL_CLAMPED``
    - throttle and brake together -> throttle dropped (brake wins) -> ``CONTROL_CLAMPED``
    """
    codes = list(reason_codes)
    try:
        steer, throttle, brake = float(steer), float(throttle), float(brake)
        finite = math.isfinite(steer) and math.isfinite(throttle) and math.isfinite(brake)
    except (TypeError, ValueError):
        finite = False
    if not finite:
        codes.append(ReasonCode.CONTROL_NAN_REJECTED.value)
        return ControlCommand(0.0, 0.0, 1.0, int(input_frame_id), True, _dedupe(codes))
    s = min(max(steer, STEER_BOUNDS[0]), STEER_BOUNDS[1])
    t = min(max(throttle, PEDAL_BOUNDS[0]), PEDAL_BOUNDS[1])
    b = min(max(brake, PEDAL_BOUNDS[0]), PEDAL_BOUNDS[1])
    clamped = abs(s - steer) > _EPS or abs(t - throttle) > _EPS or abs(b - brake) > _EPS
    if t > 0 and b > 0:
        t = 0.0
        clamped = True
    if clamped:
        codes.append(ReasonCode.CONTROL_CLAMPED.value)
    if not codes:
        codes.append(ReasonCode.ROUTE_FOLLOWING.value)
    return ControlCommand(s, t, b, int(input_frame_id), bool(safety_override), _dedupe(codes))


def brake_override(command: ControlCommand, reason_codes: Iterable[str]) -> ControlCommand:
    """Safety supervisor override: throttle 0, full brake, steering kept only when finite and in bounds."""
    steer = command.steer if math.isfinite(command.steer) else 0.0
    steer = min(max(steer, STEER_BOUNDS[0]), STEER_BOUNDS[1])
    codes = _dedupe([*reason_codes, *command.reason_codes])
    return ControlCommand(steer, 0.0, 1.0, command.input_frame_id, True, codes)


def command_to_wire(command: ControlCommand) -> dict[str, object]:
    """Run.live.command / world-state ``command`` shape."""
    return {
        "steer": round(float(command.steer), 6),
        "throttle": round(float(command.throttle), 6),
        "brake": round(float(command.brake), 6),
        "input_frame_id": int(command.input_frame_id),
        "source": "dreamdrive_controller",
        "safety_override": bool(command.safety_override),
        "reason_codes": list(command.reason_codes),
    }
