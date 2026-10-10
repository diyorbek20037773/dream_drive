# Simulation and control implementation

## LiteRoad default
Implement a CPU, deterministic, fixed-step environment with a small reachable road graph and planar ego dynamics. A renderer creates perspective RGB frames using the ego pose, camera transform and road/scene geometry; a separate top-down route map is only for UI/navigation. Use a simple tested projection and low-speed bounded operating domain. It need not look like CARLA, but it must respond to controls and create new observations.

The renderer necessarily knows scene objects. That scene object list is private to the simulator/evaluator. `SyntheticRgbPerception` receives only image pixels plus static camera calibration. It extracts lane/drivable regions and visually marked obstacle boxes using an explicit classical pixel algorithm. Its source is `synthetic_rgb_cv`, not YOLOPv2 or real-road ML. Do not project hidden obstacle coordinates directly into world state. If a lane/obstacle cannot be seen, runtime must lose confidence/slow/stop as configured.

For the optional learned mode, send the same rendered RGB to the learned inference service. Do not enable it until controlled lane/obstacle/missing-camera tests pass. Failure to generalize is a reportable result. Synthetic CV makes the feedback demo feasible; it does not prove the learned camera model works in simulation or public roads.

## Interfaces and separation
`SimulatorAdapter.reset(seed, scenario)`, `get_rgb_frame()`, `get_ego_telemetry()`, `get_route_graph()`, `apply_control(command)`, `step(dt)`, `close()`.
Evaluation truth is exposed through a separate evaluator-only object/interface, not through the runtime adapter passed to decision code. Ground truth may measure collisions, route progress and lane deviation. Ego x/y/yaw/speed and static road topology are allowed localization/navigation inputs and must be tagged telemetry.

## Loop
Fixed simulated dt default 0.05 s: acquire frame at tick N → validate frame/source/time → perception → world state → FSM/route target → controller → safety clamp/override → apply command → step to tick N+1 → evaluator-only measurements. Initial reset captures a valid initial image; never control against an uninitialized frame. A* selects route graph waypoints; Pure Pursuit follows bounded curvature and PID controls speed. Pixel lane offset/quality and obstacle region must influence the target/control/safety, not merely decorate the feed.

Navigation localization may follow the known route, but a pixel ablation must alter behavior. Commands contain reason codes and input frame identity. Steering [-1,1], throttle/brake [0,1], no throttle+brake together. Declare units, coordinate handedness, yaw sign, camera convention, maximum speed and timestep. Synthetic ODD default clear daylight, marked lanes and visually distinctive obstacles; no unimplemented junction/law claims.

FSM precedence: operator stop/failure → emergency/safety stop → paused → arrived → controlled stop/slow → follow route. Define transition conditions, hysteresis and dwell times. A short stop scenario may require stopping before an obstacle rather than successfully navigating around it. This is a valid safety test, not route completion.

Safety checks: missing/black/corrupt frame, wall acquisition age, invalid confidence status, invalid controls, out-of-bound ego state, RPC timeout and operator stop. Wall time and simulation time are distinct. Synchronous pause does not age a simulated frame, but a stalled acquisition/inference still needs a wall watchdog. Default 5 s live-simulation inference timeout (offline video inference uses a separate 30 s default); stop acknowledgement is immediate while stopping completion is bounded by the actual worker/inference cancellation path. Measure its limit. A watchdog must stop stepping and apply brake on failure; don't report an unrealistically small stop latency from a mock.

## Mandatory scenarios
Normal route arrival; lane bend; visible obstacle controlled stop; black/occluded camera safety state; stale/missing frame; unreachable destination rejected; pause/resume; operator stop; control clamp/NaN; deterministic repeat at a fixed seed. At least three seeds for normal and obstacle scenarios. Record actual completion/failure, collisions, deviations, commands, pixel sources and wall-time factor. Change hidden evaluator actor state with identical RGB to verify runtime commands are unchanged. Change/occlude RGB with identical ego/route to verify commands or capability/safety state react.

## CARLA optional
Candidate 0.9.16 UE4 with matched client wheel/server. The official guide recommends a 2070-equivalent GPU and 8 GB VRAM or more; current laptop has 6 GB. Timebox download feasibility/startup/RGB/control smoke and keep LiteRoad working if CARLA cannot fit. Do not download a huge UE5 package by default. Versioned server image/driver/Linux-container assumptions must be documented; Docker Desktop/Windows Vulkan behavior may differ from native Linux.

Use synchronous fixed timestep with one ticking client, frame ID alignment, bounded sensor queue and proper cleanup/restoration of world settings. Disable ego `set_autopilot`; traffic actors may use Traffic Manager only as labeled scene generation. Simulator depth/actor boxes are evaluation-only. Use offscreen rendering, not no-rendering mode for an RGB camera. Public web must never expose CARLA RPC ports. Successful manual setup is not portable-image proof.
