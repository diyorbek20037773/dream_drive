"""Deterministic driving-decision FSM (docs/IMPLEMENTATION_SPEC.md section 4).

Precedence (highest first): operator stop / failure -> safety_stop -> paused -> arrived -> controlled_stop / slow ->
follow_route.

Obstacle hysteresis on ``image_obstacle_cue`` (pixel perception only):
- controlled_stop: enter at cue >= 0.70; leave only after cue <= 0.45 has held for 1.0 s of simulated time;
- slow: enter at cue >= 0.35; leave only after cue <= 0.25 has held for 1.0 s.
Lane quality: ``uncertain``/``unavailable`` lane -> slow (LANE_UNCERTAIN_SLOW); back to normal after the lane has
been ``valid`` for 0.5 s. Safety stops and arrival latch until the run ends. Inputs never include simulator truth.

Decision basis: every ``step`` records, in ``last_basis``, the inputs it actually evaluated (in evaluation order, up to
the precedence level that decided) with the threshold/comparator that was tested - see ``app.control.basis``.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.control.basis import BasisItem
from app.schemas.reason_codes import ReasonCode

STOP_ENTER, STOP_EXIT = 0.70, 0.45
SLOW_ENTER, SLOW_EXIT = 0.35, 0.25
OBSTACLE_EXIT_DWELL_S = 1.0
LANE_RECOVER_DWELL_S = 0.5

R = ReasonCode


@dataclass(frozen=True)
class FsmInputs:
    sim_time_s: float
    operator_stop: bool = False
    failure_code: str | None = None
    safety_codes: tuple[str, ...] = ()
    paused: bool = False
    arrived: bool = False
    obstacle_cue: float | None = None
    lane_status: str = "valid"
    # Producer of obstacle_cue / lane_status (basis provenance only; never changes the decision).
    perception_source: str = "synthetic_rgb_cv"


@dataclass(frozen=True)
class FsmDecision:
    state: str
    reason_codes: tuple[str, ...]


class DrivingFsm:
    """``dt`` is the decision period: a condition first seen at time t0 has been held for (t - t0 + dt) seconds,
    so e.g. 20 consecutive 0.05 s samples satisfy a 1.0 s dwell."""

    def __init__(self, dt: float = 0.05) -> None:
        self.dt = dt
        self.obstacle_level = 0  # 0 none, 1 slow, 2 stop
        self._exit_since: float | None = None
        self.lane_slow = False
        self._lane_ok_since: float | None = None
        self.safety_latched: tuple[str, ...] = ()
        self.arrived_latched = False
        self.state = "observe"
        self.last_basis: tuple[BasisItem, ...] = ()
        self._basis: list[BasisItem] = []

    def _update_obstacle(self, cue: float | None, t: float, source: str) -> bool:
        """Returns True when the obstacle level dropped to 0 on this update (OBSTACLE_CLEARED)."""
        if cue is None:
            # no cue this tick: the hysteresis level is kept unchanged
            effect = "controlled_stop" if self.obstacle_level == 2 else "slow"
            self._basis.append(
                BasisItem("image_obstacle_cue", None, source, triggered=self.obstacle_level > 0, effect=effect)
            )
            return False
        prev = self.obstacle_level
        if cue >= STOP_ENTER:
            self.obstacle_level = 2
        elif cue >= SLOW_ENTER and self.obstacle_level < 1:
            self.obstacle_level = 1
        tested = self.obstacle_level
        exit_thr = STOP_EXIT if self.obstacle_level == 2 else SLOW_EXIT
        dwell: float | None = None
        if self.obstacle_level > 0 and cue <= exit_thr:
            if self._exit_since is None:
                self._exit_since = t
            dwell = t - self._exit_since + self.dt
            if dwell >= OBSTACLE_EXIT_DWELL_S - 1e-9:
                # stop -> slow while the cue is still above the slow exit threshold, otherwise straight to none
                self.obstacle_level = 1 if self.obstacle_level == 2 and cue > SLOW_EXIT else 0
                self._exit_since = None
        else:
            self._exit_since = None
        self._obstacle_basis(cue, prev, tested, dwell, source)
        return prev > 0 and self.obstacle_level == 0

    def _obstacle_basis(self, cue: float, prev: int, tested: int, dwell: float | None, source: str) -> None:
        """Basis for the obstacle cue: the enter threshold when a level was (or could be) entered, otherwise the exit
        threshold of the level being held; plus the exit dwell timer while it runs."""
        if tested == 2 and (prev < 2 or cue >= STOP_ENTER):
            item = BasisItem("image_obstacle_cue", cue, source, None, STOP_ENTER, ">=", True, "controlled_stop")
        elif tested == 2:
            item = BasisItem(
                "image_obstacle_cue", cue, source, None, STOP_EXIT, ">", cue > STOP_EXIT, "controlled_stop"
            )
        elif tested == 1 and (prev < 1 or cue >= SLOW_ENTER):
            item = BasisItem("image_obstacle_cue", cue, source, None, SLOW_ENTER, ">=", True, "slow")
        elif tested == 1:
            item = BasisItem("image_obstacle_cue", cue, source, None, SLOW_EXIT, ">", cue > SLOW_EXIT, "slow")
        else:
            item = BasisItem("image_obstacle_cue", cue, source, None, SLOW_ENTER, ">=", False, "slow")
        self._basis.append(item)
        if dwell is not None:
            effect = "controlled_stop" if tested == 2 else "slow"
            held = self.obstacle_level == tested  # the level is held until the dwell has elapsed
            self._basis.append(
                BasisItem("obstacle_exit_dwell_s", dwell, source, "s", OBSTACLE_EXIT_DWELL_S, "<", held, effect)
            )

    def _update_lane(self, lane_status: str, t: float, source: str) -> None:
        dwell: float | None = None
        if lane_status != "valid":
            self.lane_slow = True
            self._lane_ok_since = None
        elif self.lane_slow:
            if self._lane_ok_since is None:
                self._lane_ok_since = t
            dwell = t - self._lane_ok_since + self.dt
            if dwell >= LANE_RECOVER_DWELL_S - 1e-9:
                self.lane_slow = False
                self._lane_ok_since = None
        self._basis.append(
            BasisItem("lane_status", lane_status, source, triggered=lane_status != "valid", effect="slow")
        )
        if dwell is not None:
            self._basis.append(
                BasisItem("lane_valid_dwell_s", dwell, source, "s", LANE_RECOVER_DWELL_S, "<", self.lane_slow, "slow")
            )

    def step(self, inp: FsmInputs) -> FsmDecision:
        self._basis = []
        b = self._basis
        failed = bool(inp.failure_code)
        b.append(BasisItem("failure_code", inp.failure_code, "safety_supervisor", triggered=failed, effect="fail"))
        if inp.failure_code:
            return self._set("failed", (inp.failure_code,))
        b.append(
            BasisItem("operator_stop", inp.operator_stop, "operator", triggered=inp.operator_stop, effect="safety_stop")
        )
        if inp.operator_stop:
            return self._set("safety_stop", (R.OPERATOR_STOP.value,))
        if inp.safety_codes and not self.safety_latched:
            self.safety_latched = tuple(dict.fromkeys(inp.safety_codes))
        elif inp.safety_codes:
            self.safety_latched = tuple(dict.fromkeys((*self.safety_latched, *inp.safety_codes)))
        latched = ",".join(self.safety_latched) or None
        b.append(BasisItem("safety_codes", latched, "safety_supervisor", triggered=bool(latched), effect="safety_stop"))
        if self.safety_latched:
            return self._set("safety_stop", self.safety_latched)
        b.append(BasisItem("operator_pause", inp.paused, "operator", triggered=inp.paused, effect="pause"))
        if inp.paused:
            return self._set("paused", (R.OPERATOR_PAUSE.value,))
        arrived = inp.arrived or self.arrived_latched
        b.append(BasisItem("arrival_condition", arrived, "route_graph", triggered=arrived, effect="arrive"))
        if inp.arrived or self.arrived_latched:
            self.arrived_latched = True
            return self._set("arrived", (R.ARRIVED_AT_DESTINATION.value,))
        cleared = self._update_obstacle(inp.obstacle_cue, inp.sim_time_s, inp.perception_source)
        self._update_lane(inp.lane_status, inp.sim_time_s, inp.perception_source)
        lane_codes = (R.LANE_UNCERTAIN_SLOW.value,) if self.lane_slow else ()
        if self.obstacle_level == 2:
            return self._set("controlled_stop", (R.OBSTACLE_AHEAD_STOP.value, *lane_codes))
        if self.obstacle_level == 1:
            return self._set("slow", (R.OBSTACLE_AHEAD_SLOW.value, *lane_codes))
        cleared_codes = (R.OBSTACLE_CLEARED.value,) if cleared else ()
        if self.lane_slow:
            return self._set("slow", (*lane_codes, *cleared_codes))
        return self._set("follow_route", (R.ROUTE_FOLLOWING.value, *cleared_codes))

    def _set(self, state: str, codes: tuple[str, ...]) -> FsmDecision:
        self.state = state
        self.last_basis = tuple(self._basis)
        return FsmDecision(state, codes)
