"""Decision-basis records: the inputs a decision component actually evaluated on one tick.

The safety supervisor, the FSM and the controller each append ``BasisItem`` records *while* they evaluate their own
inputs (same variables, same thresholds), so the UI shows exactly what drove the decision; nothing is recomputed
for display. The session tags every item with the input frame id of the command it belongs to.

Field semantics
- ``value``: the evaluated input (number, string, bool or null when the input was unavailable).
- ``unit``: SI unit of ``value``/``threshold`` (``s``, ``m``, ``m/s``, ``1/m``) or None for unitless values.
- ``source``: where the value comes from (pixel perception mode, ego telemetry, static route graph, operator,
  safety supervisor watchdog measurements, controller internals). Evaluator truth is never a source.
- ``threshold`` / ``comparator``: the trigger condition is ``value <comparator> threshold`` (e.g. ``0.52 >= 0.35``).
  For a hysteresis hold the comparator/threshold shown is the exit condition that is currently being tested.
- ``triggered``: true when this input is currently driving ``effect`` (for a gate: the gate condition holds).
- ``effect``: what the input drives when triggered (``slow``, ``controlled_stop``, ``safety_stop``,
  ``steer_correction``, ``speed_limit``, ``pause``, ``arrive``, ``fail``); None for informational inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

BASIS_SOURCES: tuple[str, ...] = (
    "synthetic_rgb_cv",
    "learned_model",
    "ego_telemetry",
    "route_graph",
    "operator",
    "safety_supervisor",
    "controller",
)
BASIS_EFFECTS: tuple[str, ...] = (
    "slow",
    "controlled_stop",
    "safety_stop",
    "steer_correction",
    "speed_limit",
    "pause",
    "arrive",
    "fail",
)
BASIS_COMPARATORS: tuple[str, ...] = (">=", ">", "<=", "<", "==", "!=")

BasisValue = bool | int | float | str | None


def _clean(value: Any) -> BasisValue:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return int(value)
    f = float(value)
    return round(f, 6) if math.isfinite(f) else None


@dataclass(frozen=True)
class BasisItem:
    key: str
    value: BasisValue
    source: str
    unit: str | None = None
    threshold: float | None = None
    comparator: str | None = None
    triggered: bool = False
    effect: str | None = None

    def __post_init__(self) -> None:
        if self.source not in BASIS_SOURCES:
            raise ValueError(f"unknown basis source {self.source!r}")
        if self.effect is not None and self.effect not in BASIS_EFFECTS:
            raise ValueError(f"unknown basis effect {self.effect!r}")
        if self.comparator is not None and self.comparator not in BASIS_COMPARATORS:
            raise ValueError(f"unknown comparator {self.comparator!r}")

    def to_wire(self, input_frame_id: int) -> dict[str, Any]:
        """Run.live.decision_basis item (JSON-serialisable; non-finite numbers become null)."""
        threshold = _clean(self.threshold)
        return {
            "key": self.key,
            "value": _clean(self.value),
            "unit": self.unit,
            "source": self.source,
            "threshold": threshold if isinstance(threshold, int | float) and not isinstance(threshold, bool) else None,
            "comparator": self.comparator,
            "triggered": bool(self.triggered),
            "effect": self.effect,
            "input_frame_id": int(input_frame_id),
        }
