# LiteRoad simulation evidence (bounded synthetic results)

**Scope label.** Everything below was measured in LiteRoad, DreamDrive's own CPU synthetic simulator, with the
classical pixel perception `synthetic_rgb_cv`. These are results about a flat, clear-daylight synthetic town with
painted lanes and orange/white striped boxes. They are **not** evidence about real roads, real vehicles or the
YOLOPv2 model (except the separately labelled learned-model validation section). No physical actuator is ever driven.

## Measurement context
- Date: 2026-10-10, ~12:00-12:25 UTC. Source: uncommitted working tree on top of commit
  `f85c397d5e8435aed29cafd1f4b5e54658d62202` (the code measured is the LiteRoad component as delivered to the lead;
  the suite artifact therefore records `code_commit: null` instead of a commit that does not contain this code).
- Host: Windows 11 Pro (10.0.26300), Intel Core i5-13420H (12 logical CPUs), 15.7 GB RAM, Python 3.11.9,
  numpy 2.4.6, opencv-python-headless 5.0.0, PyAV 18.1.0 (backend `.venv`). CPU only.
- Concurrent load: other engineers' Docker builds and test suites were running on the same machine during all runs
  (an earlier sample read ~45 % total CPU busy before these runs started). Wall-clock timings are therefore
  conservative and vary between runs; simulated-time results (ticks, outcomes, deviations, gaps) are deterministic
  per seed and do not depend on load.
- Commands: `python <scratch>/evidence.py ../artifacts/evaluations` (calls
  `app.evaluation.scenario_suite.run_scenario_suite([1, 2, 3], ...)` plus render/perception micro-benchmarks); the
  Evaluation artifact is `artifacts/evaluations/26f38e26ca9348a0adebad38b14746ab.json` (git-ignored).

## Conventions that the numbers rely on
- World frame ENU metres, x east, y north, yaw CCW from +x. Ego origin = rear-axle centre; footprint 4.5 x 1.8 m
  (0.9 m rear overhang, front bumper 3.6 m ahead of the origin). Kinematic bicycle, wheelbase 2.6 m, max steer
  0.5 rad, steer +1 = full left, speed 0..5 m/s, accel 2.5 m/s^2, brake 6 m/s^2, drag 0.3 m/s^2, dt 0.05 s.
- Camera 640x360, HFOV 90 deg (fx = fy = 320), 1.4 m high, 1.0 m ahead of the origin, pitched 8 deg down.
- Roads 7 m wide, right-hand traffic, ego lane centre 1.75 m right of the centreline; junction boxes (degree >= 3)
  are 18 x 18 m unmarked asphalt; route corners are filleted with an 8 m centreline radius (right-turn lane radius
  6.25 m, above the 4.76 m minimum turning radius).
- Controller: cruise 4.8 m/s, slow 2.5 m/s, lateral-acceleration budget 0.6 m/s^2 for the curve speed limit,
  Pure Pursuit lookahead clamp(2 + 0.6 v, 3, 8) m, pixel lane blend weight 0.5 (valid) / 0.25 (uncertain).
- Obstacle cue: 0 at >= 24 m, 1 at <= 8 m (flat-ground IPM distance from the ego origin to the box contact line);
  FSM stop enter 0.70 (12.8 m) / exit 0.45 with 1.0 s dwell, slow enter 0.35 (18.4 m) / exit 0.25.
- Lateral deviation = evaluator distance of the rear axle from the truth lane-centre path of the route, sampled after
  every physics step. Obstacle gap = distance between the ego footprint and the obstacle footprint.

## Scenario suite, seeds 1-3 (15 runs, headless, no pacing, no JPEG encoding)

| scenario | seed | route | length m | outcome | ticks | sim s | wall s | collisions | dev mean m | dev max m | final gap m | first override (sim s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| normal | 1 | A->C | 130.0 | arrived | 746 | 37.25 | 9.47 | 0 | 0.034 | 0.231 | - | - |
| normal | 2 | A->C | 130.0 | arrived | 725 | 36.20 | 8.89 | 0 | 0.027 | 0.107 | - | - |
| normal | 3 | A->C | 130.0 | arrived | 723 | 36.10 | 9.13 | 0 | 0.034 | 0.202 | - | - |
| bend | 1 | D->H->I | 130.0 | arrived | 724 | 36.15 | 3.45 | 0 | 0.036 | 0.196 | - | - |
| bend | 2 | D->H->I | 130.0 | arrived | 725 | 36.20 | 6.72 | 0 | 0.029 | 0.130 | - | - |
| bend | 3 | D->H->I | 130.0 | arrived | 706 | 35.25 | 6.75 | 0 | 0.042 | 0.278 | - | - |
| obstacle_stop | 1 | A->C | 130.0 | stopped_before_obstacle | 198 | 9.90 | 2.89 | 0 | 0.023 | 0.026 | 8.152 | - |
| obstacle_stop | 2 | A->C | 130.0 | stopped_before_obstacle | 205 | 10.25 | 3.39 | 0 | 0.054 | 0.185 | 8.099 | - |
| obstacle_stop | 3 | A->C | 130.0 | stopped_before_obstacle | 198 | 9.90 | 3.61 | 0 | 0.055 | 0.275 | 8.188 | - |
| camera_blackout | 1 | A->C | 130.0 | stopped_camera_invalid | 195 | 9.75 | 1.23 | 0 | 0.050 | 0.193 | - | 4.00 (CAMERA_BLACK) |
| camera_blackout | 2 | A->C | 130.0 | stopped_camera_invalid | 195 | 9.75 | 1.30 | 0 | 0.030 | 0.065 | - | 4.00 (CAMERA_BLACK) |
| camera_blackout | 3 | A->C | 130.0 | stopped_camera_invalid | 192 | 9.60 | 1.38 | 0 | 0.037 | 0.105 | - | 4.00 (CAMERA_BLACK) |
| stale_frame | 1 | A->C | 130.0 | stopped_frame_stale | 195 | 9.75 | 1.20 | 0 | 0.039 | 0.106 | - | 3.95* (FRAME_STALE) |
| stale_frame | 2 | A->C | 130.0 | stopped_frame_stale | 192 | 9.60 | 1.20 | 0 | 0.059 | 0.243 | - | 3.95* (FRAME_STALE) |
| stale_frame | 3 | A->C | 130.0 | stopped_frame_stale | 191 | 9.55 | 1.34 | 0 | 0.027 | 0.044 | - | 3.95* (FRAME_STALE) |

\* The frozen camera keeps returning the last good frame (id 79, t = 3.95 s); the session detects the repeated
frame id on the first tick at simulated t = 4.00 s and brakes in that same tick. The camera blackout is likewise
caught on the first black frame (t = 4.00 s), before any further physics step with throttle. Seeds vary the initial
pose (+-0.3 m, +-0.04 rad) and, for `obstacle_stop`, the obstacle position (25 +- 3 m along, +-0.3 m across);
deviation maxima are dominated by the initial lateral offset.

Aggregates written to the Evaluation artifact (all `measured`):

| metric | value | samples |
|---|---|---|
| expected_outcome_rate (all 15 runs) | 1.0 | 15 |
| normal / bend arrival rate | 1.0 / 1.0 | 3 / 3 |
| obstacle_controlled_stop_rate, obstacle_false_arrival_count | 1.0, 0 | 3 |
| camera_blackout / stale_frame safety-stop rate | 1.0 / 1.0 | 3 / 3 |
| collisions_total | 0 | 15 |
| lateral_deviation mean / p95 / max (normal + bend) | 0.0334 / 0.1097 / 0.2779 m | 4343 steps |
| obstacle_stop_gap mean / min | 8.146 / 8.098 m | 3 |
| sim_time_total / wall_time_total / realtime_factor_headless | 305.2 s / 61.95 s / 4.93 | 15 |
| tick_wall_ms p50 / p95 (headless tick incl. evaluator) | 10.36 / 18.07 ms | 6110 ticks |
| real_road_driving_accuracy | null (`unsupported`) | 0 |

## Timing per tick on this CPU
- World raster build (once per process, 3000 x 2450 px at 0.1 m/px + 2 mip levels): 0.46 s.
- Render 640x360 (ground via 3 perspective warps + ~15 projected boxes): p50 2.32 ms, p95 3.42 ms (300 frames).
- `synthetic_rgb_cv` perception: p50 3.20 ms, p95 4.39 ms (300 frames, same session, no other work in-loop).
- Full closed-loop tick (frame, perception, FSM, control, step, evaluator; JPEG skipped in the suite): p50 10.4 ms,
  p95 18.1 ms under the concurrent load described above; individual runs ranged from p50 4.7 ms (bend seed 1) to
  p50 16-18 ms. Live runs add JPEG encoding (~1-2 ms) and are paced by the worker to realtime_factor <= 1.0, so the
  20 Hz simulation has a > 2x CPU margin on this machine.

## Additional routes (normal scenario, seed 1, current code)
All reachable start/destination pairs were run once with seed 1; see the all-pairs table below. Representative:
A->D (left turn) 823 ticks, D->C 822, E->I (right turn at D + S-bend) 1142, I->E 1179, B->G (curved edge) 892,
G->I (longest route, 353.9 m) 2014 ticks, F->A 1587, C->A 705 - all `arrived`, 0 collisions, 0 clamped commands,
mean deviation 0.032-0.046 m, max 0.231-0.234 m. The longest route uses 84 % of the 2400-tick limit.

All-pairs check (same code, seed 1, scenario `normal`): every one of the 72 ordered pairs of the 9 connected nodes
(A-I) was run to a terminal state. Result: 72/72 `arrived`, 0 collisions, 0 clamped or NaN-rejected commands;
ticks 301 (H->I) to 2140 (I->G, 89 % of the 2400-tick limit); per-run mean lateral deviation 0.032-0.053 m
(average 0.039 m), largest single sample 0.267 m. Routes to or from `Q` are rejected (`destination_unreachable`).

## Behavioural tests that ran (host `backend/.venv`, all passing)
Final combined run: `cd backend && .venv/Scripts/python -m pytest ../tests/scenarios ../tests/unit/backend/test_sim_*.py
../tests/unit/backend/test_{navigation,control,fsm,safety,synthetic_cv}_*.py -q` -> 107 passed, 1 skipped (43.6 s).

- `tests/scenarios/` (32 passed, 1 skipped without a live service): normal arrival seeds 1-3 within 1000 ticks with
  zero collisions; bend through the curved edges with curve speed limit engaged and both steering directions;
  obstacle controlled stop seeds 1-3 (gap >= 2 m, never `arrived`); camera blackout -> `safety_stop` with full brake
  on the first black frame and on every later tick; stale frame -> `FRAME_STALE` stop; unreachable/unknown/same
  destination rejected; learned mode gated; pause/resume freezes pose, frame id and sim time and the resumed frame is
  valid; operator stop (also while paused) reaches standstill in < 40 ticks and is idempotent; injected obstacle
  stop and 5-injection limit; tick limit, wall-clock watchdog (no further stepping) and out-of-bounds failures;
  in-loop NaN control rejection; **pixel ablation** (occluded / black RGB with identical ego and route changes
  perception, decision and command; a laterally shifted camera with identical telemetry changes steering);
  **hidden-truth isolation** (an evaluator-only obstacle with identical RGB yields identical commands, decisions and
  perception every tick until the evaluator reports the collision); deterministic repeat (seed 3 twice -> identical
  command sequence; seed 4 differs); world-state / Run.live / run-report contract validation; scenario suite
  artifact + registry; simulator catalogue; CLI (H.264 video, report only with an explicit commit).
- `tests/unit/backend/test_sim_*`, `test_navigation_*`, `test_control_*`, `test_fsm_*`, `test_safety_*`,
  `test_synthetic_cv_*` (75 passed): ENU/yaw/steer sign conventions and units, renderer projection/occlusion/time
  budget (<= 15 ms), A* vs Floyd-Warshall on every pair, lane-offset geometry, PID anti-windup and clamp, command
  clamp/NaN/exclusive pedals, FSM precedence/hysteresis/dwell/latching, safety checks, pixel perception validity,
  lane/heading signs, obstacle IPM distance (+-0.8 m), corridor test, determinism, learned-mode mapping (fake
  client, labelled as such) and gating.

## Learned-model perception on LiteRoad renders (separate, real model)
Measured with `tests/scenarios/test_learned_sim_validation.py` against a real `dreamdrive-inference:dev` container
(CPU, FP32, YOLOPv2 checkpoint SHA256 `f2a8c8374203ae3e67ff9c184e931f763957de92a993b23269e4e721627f1f8c`,
class map `{"3": "vehicle"}`), started locally on a private port with the verified weights mounted read-only.
Outcome file: `artifacts/learned_sim_validation/learned_sim_validation_20261010T121211Z.json` (git-ignored).

| check | result | measurement |
|---|---|---|
| lane (5 straight renders, ego offset -0.5..+0.5 m, tolerance 0.4 m) | **fail** | 3/5 within tolerance; errors 0.11, 0.43, 0.01, 0.14 m; one frame without any model lane line |
| obstacle (closed loop `obstacle_stop`, seed 1) | **fail** | the orange/white box was not detected (only class `vehicle` exists); evaluator collision at tick 98 (`failed_collision`) |
| missing camera (closed loop `camera_blackout`, seed 1) | pass | black frame never sent to the model; `safety_stop` + full brake at t = 4.00 s |

Model inference round trip p50 382 ms per frame on this CPU. Conclusion: `LEARNED_SIM_MODE_ENABLED` must stay
`false`; the mode is visibly unavailable in the simulator catalogue with that reason. This is the expected domain-shift
result, reported rather than hidden.

## Known limitations
- One fixed route per suite scenario; the operating domain is the synthetic town only (flat ground, no weather,
  no traffic, no pedestrians, no traffic rules or junction right-of-way).
- Lane markings are absent inside junction boxes by design, so perception reports `unavailable` there and the FSM
  drives those segments in `slow` (LANE_UNCERTAIN_SLOW); about 40-50 % of ticks on junction-heavy routes are `slow`.
- Obstacle distance is a flat-ground IPM estimate from the static calibration; it is not metric depth and no TTC is
  computed (`ttc_seconds` null, metric `unsupported`).
- The obstacle corridor uses the perceived lane curve when available and a straight corridor otherwise.
- Localization is perfect simulator telemetry (allowed runtime input); the pixel lane estimate is blended into the
  steering target but cannot correct a localization error because none is simulated.
- CARLA is reported unavailable (no recorded smoke test; 6 GB VRAM < 8 GB recommendation).
