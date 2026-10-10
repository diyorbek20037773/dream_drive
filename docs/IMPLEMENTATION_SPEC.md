# DreamDrive implementation specification (engineering contract)

This file pins the concrete module boundaries, names and wire formats that every component must share.
It refines docs/ARCHITECTURE.md, docs/API.md, docs/SIMULATION.md and docs/ML_PIPELINE.md. When a detail
here conflicts with those documents, those documents win on *policy*; this file wins on *names/shapes*.

## 0. Runtime and tooling (resolved 2026-10-10)
- Python 3.11 for backend and inference, dependencies locked with uv 0.12.9 (`backend/uv.lock`, `inference/uv.lock`).
  Do not hand-edit locks. Add a dependency only with `uv add` inside that project and re-lock.
- Node 22 LTS + npm lockfile for `frontend/`.
- Base images: `python:3.11-slim-bookworm`, `node:22-bookworm-slim` (build), `nginxinc/nginx-unprivileged:1.27-alpine` (web),
  `caddy:2.10-alpine` (gateway), `redis:7.4-alpine`. Digests are recorded at release time, never typed by hand.
- All Dockerfiles build with the **repository root as context** (`docker build -f backend/Dockerfile .`) so they can copy
  `config/`, `contracts/` and `tests/`. Root `.dockerignore` excludes `.git`, `node_modules`, `models/`, `artifacts/`,
  `DreamDrive_Claude_Kit/`, `**/.venv`, `frontend/dist`.
- Two Python packages are both named `app` (`backend/app`, `inference/app`). They never share an interpreter. Backend tests
  run in the backend `test` image, inference tests in the inference `test` image.
- Image filesystem layout mirrors the repository under `/app`: `/app/backend` (WORKDIR of backend images, package `app`),
  `/app/inference` (WORKDIR of inference images), `/app/config`, `/app/contracts`, `/app/tests` (test targets only).
  Code finds repo files relative to its package (`Path(__file__).resolve().parents[N]`), never via absolute dev paths.
- Shared cross-component call signatures:
  - `app.perception.client.InferenceClient(base_url: str, timeout_s: float)` with `.model_info() -> ModelInfo`,
    `.ready() -> bool`, `.infer(image_bytes: bytes, frame_id: int, content_type="image/jpeg", conf_threshold=0.1,
    include_masks=True) -> InferenceResult`; raises `InferenceUnavailable`, `InferenceBusy`, `InferenceTimeout`,
    `InferenceError` (all subclasses of `InferenceClientError`). Models from `app.schemas.inference`.
  - `app.video.pipeline.run_video_analysis(job_id, input_path, out_dir, client, settings, progress_cb, cancel_check)
    -> dict` (summary as in `JobResults.summary`); raises `app.video.errors.VideoError(code, message)` and
    `app.video.errors.JobCancelled`.
  - `app.evaluation.scenario_suite.run_scenario_suite(seeds: list[int], out_dir: Path, code_commit: str,
    progress_cb=None) -> dict` (an `Evaluation` dict, see §3.4) and `app.evaluation.registry.list_evaluations(root: Path)
    -> list[dict]`, `load_evaluation(root, eval_id) -> dict | None`.

## 1. Services
| Service | Image | Port (internal) | Command |
|---|---|---|---|
| gateway | caddy | 8080 (published `127.0.0.1:${DD_HTTP_PORT:-8080}`) | Caddyfile in `deploy/caddy/Caddyfile` |
| web | dreamdrive-web | 8080 | nginx static, SPA fallback |
| api | dreamdrive-backend | 8000 | `uvicorn app.main:app --host 0.0.0.0 --port 8000` |
| worker | dreamdrive-backend | 9101 (metrics) | `python -m app.jobs.worker_main` |
| redis | redis | 6379 | AOF on, volume `redis-data` |
| inference | dreamdrive-inference | 8001 | `uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 1` |
| model-init | dreamdrive-inference | – | `python -m app.model_init` (one-shot; download+verify into volume `model-cache`) |

Volumes: `app-data` (/data: SQLite, artifacts, tmp, evaluations), `model-cache` (/models), `redis-data`, `caddy-data`, `caddy-config`.
Gateway: `/api/*` and `/health/*` → api:8000 (SSE: `flush_interval -1`), everything else → web:8080, `/metrics` → 404,
request body max 110 MB.

## 2. Inference service (`inference/app`)
Files: `main.py` (FastAPI), `settings.py`, `manifest.py` (load `config/models.resolved.yaml`, verify file SHA256+size),
`model.py` (TorchScript load, device/precision, one global lock), `preprocess.py` (letterbox + metadata),
`postprocess.py` (decode heads, NMS, mask inversion), `vendored/yolopv2_utils.py` (copied upstream helpers, MIT notice,
upstream commit 9388c6aa55f3bf5a878679a1338bb090cff6a256, list of modifications), `model_init.py` (download with timeout,
`.part` file, SHA256 verify, atomic rename; exit 0 if already verified), `metrics.py`.

Env: `MODEL_ROOT=/models`, `MODEL_MANIFEST_PATH=/app/config/models.resolved.yaml`, `INFERENCE_DEVICE=cpu|cuda`,
`INFERENCE_PRECISION=fp32|fp16` (fp16 only on cuda), `INFERENCE_MAX_WAITERS=4`, `INFERENCE_MAX_IMAGE_BYTES=8388608`.

Startup: verify manifest file exists, size and SHA256 match, then `torch.jit.load`, warm-up one forward pass. Any failure →
`ready=false` with reason `model_missing | hash_mismatch | load_failed | device_unavailable`; never mock inference.

HTTP (internal only, never routed by the gateway):
- `GET /health/live` → `{"status":"ok"}`
- `GET /health/ready` → 200 `{"ready":true,...}` or 503 `{"ready":false,"reason":...}`
- `GET /v1/model` → `ModelInfo` (below), always 200 (ready flag inside)
- `POST /v1/infer?frame_id=<int>&conf_threshold=0.1&iou_threshold=0.45&include_masks=true`
  body = raw JPEG or PNG bytes (`Content-Type: image/jpeg|image/png`), max 8 MiB, max 4096x4096 decoded.
  Errors: 400 `image_invalid`, 413 `image_too_large`, 503 `model_not_ready`, 503 `inference_busy` (waiters exceeded).
  One forward pass at a time (lock). Returns `InferenceResult`.
- `GET /metrics` Prometheus.

```jsonc
// ModelInfo
{"model_id":"yolopv2","ready":true,"reason":null,"sha256":"f2a8…","size_bytes":156380200,
 "repository_revision":"9388c6a…","class_map":{"3":"vehicle"},"device":"cpu","precision":"fp32",
 "torch_version":"2.14.1+cpu","input_policy":"letterbox640_stride32"}
// InferenceResult
{"schema_version":"1.0","frame_id":12,
 "model":{"model_id":"yolopv2","sha256":"f2a8…","device":"cpu","precision":"fp32"},
 "image":{"width":1280,"height":720},
 "geometry":{"canvas_width":640,"canvas_height":384,"scale":0.5,"pad_left":0,"pad_top":12,"pad_right":0,"pad_bottom":12},
 "detections":[{"class_id":3,"class_name":"vehicle","score":0.91,"xyxy_pixels":[x1,y1,x2,y2]}],
 "drivable_mask_png":"<base64 PNG, uint8 {0,255}, original HxW>",
 "lane_mask_png":"<base64 PNG, uint8 {0,255}, original HxW>",
 "mask_stats":{"drivable_fraction":0.22,"lane_fraction":0.03},
 "timing_ms":{"decode":3.1,"preprocess":2.0,"inference":410.2,"postprocess":20.5,"total":436.0}}
```
Geometry: letterbox longest side to 640, `auto=True` stride-32 minimal padding, pad colour 114, RGB, /255.
Boxes: subtract pad, divide by scale, clip to image. Masks: crop canvas by pad, resize to original (W,H) with bilinear
on probabilities, then drivable = argmax(ch)==1, lane = prob>=0.5. Class ids not in class_map → `unverified_class_<id>`.
Tests must prove: 1280x720 inversion equals upstream `driving_area_mask/lane_line_mask`; boxes on a padded synthetic
image map back to the correct original pixels for at least landscape, portrait and 4:3 inputs.

### 2.1 Second detector: RF-DETR Nano (owner request 2026-10-10: person, traffic light, stop sign)
YOLOPv2's checkpoint detects only `vehicle` (class 3). The owner asked for people, traffic lights and signs, which is
the documented trigger for the conditional extra detector (docs/ML_PIPELINE.md). Decision recorded in docs/DECISIONS.md.
- Model: RF-DETR-N (rfdetr 1.11.2, git tag `7433d5919d0f7e72cc6cb78663e0a731afd499e0`, Apache 2.0 per README model
  table), checkpoint `https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth`, 366,287,238 bytes,
  publisher MD5 `fb6504cce7fbdc783f7a46991f07639f` (matched), SHA256
  `d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d`. COCO-pretrained, 384x384 input.
- Same inference service, loaded once beside YOLOPv2; both models run on every `/v1/infer` frame; one lock.
- Enabled RF-DETR classes and DreamDrive class ids (`100 + COCO category id`): person 101, bicycle 102, motorcycle 104,
  bus 106, truck 108, traffic_light 110, stop_sign 113. COCO `car` is not enabled (YOLOPv2 `vehicle` covers cars,
  in-domain BDD100K). A bus/truck box with IoU ≥ 0.5 to a YOLOPv2 vehicle replaces that vehicle box (class refinement);
  the response records `refined_from_model`.
- Wire additions (additive, backward compatible): `InferenceResult.models: list[ModelIdentity]` (every model that ran),
  `InferenceResult.ensemble_id` (`"yolopv2+rfdetr_nano"`), `InferenceResult.ensemble_sha256` = sha256 of the UTF-8 text
  `"yolopv2:<sha256>\nrfdetr_nano:<sha256>\n"`; `RawDetection.model_id` (`"yolopv2"` | `"rfdetr_nano"`).
  `ModelInfo` gains `models: list[...]` with each model's identity, class map and readiness.
- WorldState `perception.model_id` = ensemble id and `perception.model_sha256` = ensemble sha256 when both ran;
  reports list each checkpoint SHA256 in `details`/limitations. If RF-DETR is not ready the service reports
  `ready=false` (both models are required for the release configuration; no silent single-model fallback).
- Traffic-light colour: a classical HSV classifier on each `traffic_light` crop → `red|yellow|green|unknown`, stored as
  `light_state` in timeline extras and reason codes `TRAFFIC_LIGHT_RED_HEURISTIC` etc. It is a heuristic, never labelled
  as model output.
- New reason codes: `PERSON_IN_CORRIDOR_NEAR, PERSON_IN_CORRIDOR, TWO_WHEELER_IN_CORRIDOR, TRAFFIC_LIGHT_VISIBLE,
  TRAFFIC_LIGHT_RED_HEURISTIC, TRAFFIC_LIGHT_YELLOW_HEURISTIC, TRAFFIC_LIGHT_GREEN_HEURISTIC, STOP_SIGN_VISIBLE`.
- Accuracy: COCO val2017 street-scene subset (annotations CC BY 4.0; per-image Flickr licences recorded), evaluated
  through the production `/v1/infer` path; per-class AP50 and AP50:95 with image/instance counts → Evaluation
  `kind=video_ml_accuracy`. YOLOPv2 `vehicle` is scored against COCO car+bus+truck as a cross-dataset check. Lane and
  drivable-area accuracy stay `not_evaluated` (no openly licensed labels; BDD100K requires owner registration).

## 3. Backend package (`backend/app`)
```
app/config.py            Settings (pydantic-settings); single source of env names (see .env.example)
app/main.py              create_app(); routers; error handlers; middleware (request id, JSON logs, metrics)
app/logging_setup.py     JSON logs (utc ts, level, component, request_id/job_id/run_id, code); redaction
app/metrics.py           prometheus metric objects (names in docs/OBSERVABILITY.md)
app/db.py                sqlite3 connection factory (WAL, foreign_keys, busy_timeout), migration runner
app/migrations/0001_init.sql
app/security.py          signed session cookie (itsdangerous), CSRF double-submit, rate limiting (redis)
app/schemas/             Pydantic models: world_state.py, report.py, events.py, inference.py, api.py, reason_codes.py
app/storage/repo.py      DB functions (sessions/jobs/runs/evaluations), ownership checks
app/storage/artifacts.py artifact dirs, atomic writes, allowlisted names, byte-range file responses
app/api/                 routers: health, session, capabilities, videos, jobs, simulators, runs, evaluations, events, admin
app/events/bus.py        Redis Streams per owner (`dd:events:{owner_id}`, MAXLEN~1000), publish(owner, type, resource_id, payload)
app/jobs/queue.py        RQ queues `video`, `sim`, `eval`; enqueue helpers
app/jobs/video_job.py    RQ entry `run_video_job(job_id)` → calls app.video.pipeline
app/jobs/sim_job.py      RQ entry `run_simulation_job(run_id)` → drives app.agents.runner.SimulationSession
app/jobs/eval_job.py     RQ entry `run_scenario_suite_job(eval_id)`
app/jobs/worker_main.py  starts SimpleWorker processes (video+eval, sim), metrics server :9101 (multiprocess dir),
                         startup reconciliation (running → interrupted), periodic retention cleanup
app/video/probe.py       ffprobe (argument arrays, timeout) → VideoProbe
app/video/decode.py      PyAV decode with real PTS seconds, rotation applied, sampling by PTS grid
app/video/encode.py      H.264 libx264 (verified present in the av 18.1.0 wheel) yuv420p +faststart, no audio
app/video/annotate.py    draw masks/boxes/track ids/HUD with source label
app/video/pipeline.py    run_video_analysis(...) orchestration (see §5)
app/perception/client.py InferenceClient (httpx) → schemas.inference.InferenceResult
app/perception/tracking.py ByteTrack (vendored from FoundationVision/ByteTrack, MIT, pinned commit, per-instance id counter)
app/perception/world_state.py  build WorldState from inference + decisions
app/perception/video_decisions.py analytical reason codes for uploaded video
app/perception/synthetic_rgb_cv.py SyntheticRgbPerception (pixels + static calibration only)
app/simulators/base.py   SimulatorAdapter + EvaluatorTruth protocols
app/simulators/lite/     graph.py, world.py, renderer.py, dynamics.py, scenarios.py, adapter.py, evaluator.py
app/simulators/carla_adapter.py  optional; reports unavailable unless CARLA_HOST configured and smoke-tested
app/navigation/          astar.py, route.py (lane-centre path from graph)
app/control/             pure_pursuit.py, pid.py, command.py (clamp/NaN/exclusive throttle-brake)
app/agents/              fsm.py, safety.py, runner.py (SimulationSession)
app/evaluation/          sim_metrics.py, scenario_suite.py, registry.py (list evaluation artifacts)
app/reporting/           video_report.py (report.json + report.html), sim_report.py
app/export_openapi.py    `python -m app.export_openapi > frontend/src/api/openapi.json`
```

### 3.1 Settings (env names; defaults)
`APP_ENV=development`, `APP_SECRET` (required in production; dev generates an ephemeral one with a warning),
`ADMIN_PASSWORD` (empty → operator login disabled), `PUBLIC_ORIGIN`, `DEFAULT_LOCALE=uz`,
`MAX_UPLOAD_BYTES=104857600`, `MAX_VIDEO_SECONDS=120`, `MAX_VIDEO_WIDTH=1920`, `MAX_VIDEO_HEIGHT=1080`,
`MAX_QUEUED_PER_SESSION=2`, `MAX_QUEUED_TOTAL=10`, `MAX_REQUESTS_PER_MINUTE=10`, `MAX_SIMULATION_WALL_SECONDS=120`,
`ARTIFACT_RETENTION_HOURS=24`, `ANALYSIS_FPS=5`, `INFERENCE_CONF_THRESHOLD=0.1`, `INFERENCE_BASE_URL=http://inference:8001`,
`INFERENCE_TIMEOUT_SECONDS=30`, `SIM_INFERENCE_TIMEOUT_SECONDS=5`, `SIMULATOR=lite`, `SIM_PERCEPTION_SOURCE=synthetic_rgb_cv`,
`SIM_FIXED_DT_SECONDS=0.05`, `REDIS_URL`, `SQLITE_PATH=/data/dreamdrive.db`, `DATA_ROOT=/data`, `ARTIFACT_ROOT=/data/artifacts`,
`LOG_LEVEL=INFO`, `GIT_COMMIT` (40-hex when built from git), `BUILD_VERSION`, `BUILD_DIRTY=0`, `WORKER_METRICS_PORT=9101`,
`ENABLE_MLFLOW=false`, `MLFLOW_TRACKING_URI`, `LEARNED_SIM_MODE_ENABLED=false`, `CARLA_HOST=` (empty → unavailable).

### 3.2 Database (SQLite WAL, `app/migrations/0001_init.sql`)
```sql
sessions(id TEXT PK, role TEXT CHECK(role IN ('anonymous','operator')), csrf_token TEXT, created_utc, last_seen_utc)
jobs(id TEXT PK, owner_id TEXT NOT NULL, kind TEXT DEFAULT 'video_analysis', state TEXT, stage TEXT, progress REAL,
     frames_done INT, frames_total INT, error_code TEXT, error_message TEXT, input_sha256 TEXT, input_bytes INT,
     input_ext TEXT, probe_json TEXT, model_json TEXT, settings_json TEXT, summary_json TEXT,
     cancel_requested INT DEFAULT 0, created_utc, started_utc, finished_utc, updated_utc)
runs(id TEXT PK, owner_id TEXT NOT NULL, simulator TEXT, perception_source TEXT, scenario TEXT, seed INT,
     start_node TEXT, destination_node TEXT, route_json TEXT, state TEXT, outcome TEXT, error_code TEXT, error_message TEXT,
     summary_json TEXT, created_utc, started_utc, finished_utc, updated_utc)
evaluations(id TEXT PK, owner_id TEXT, kind TEXT, state TEXT, created_utc, finished_utc, summary_json TEXT, error_code TEXT)
schema_migrations(version INT PK, applied_utc)
```
IDs are `uuid4().hex`. Ownership: every job/run/evaluation query filters by `owner_id`; a foreign id returns 404
(`not_found`), never 403, so existence is not leaked. Operator role may list all for admin endpoints only.

### 3.3 Sessions, CSRF, quotas
- `POST /api/v1/session` (body optional `{"admin_password": "..."}`) creates/refreshes the session, sets cookie
  `dd_session` (signed id, HttpOnly, SameSite=Lax, Secure when `APP_ENV=production`, max-age 24 h) and cookie `dd_csrf`
  (readable by JS). Response `SessionInfo {session_id_suffix, role, csrf_token, expires_utc, limits{...}}`.
- Every POST/DELETE except `POST /api/v1/session` requires header `X-CSRF-Token` equal to the session's token → else 403 `csrf_failed`.
- Missing/invalid session on any `/api/v1/*` route except `/session` and `/capabilities` → 401 `session_required`.
- Rate limit `MAX_REQUESTS_PER_MINUTE` for expensive requests (upload, run start, eval start) per session and per client IP
  (Redis fixed window) → 429 `rate_limited`. Queue admission: per-session queued/running video jobs ≥ 2 → 429
  `session_queue_full`; global ≥ 10 → 429 `queue_full`; active simulation exists → 429 `simulation_busy`.
- Error envelope everywhere: `{"code","message","request_id","details"}`; no paths/stack traces.

### 3.4 REST shapes (Pydantic in `app/schemas/api.py`; OpenAPI is generated from these)
- `Job {id, kind, state, stage, progress(0..1), frames_done, frames_total, error_code, error_message, created_utc, started_utc,
  finished_utc, input{sha256, bytes, ext}, video{duration_s, width, height, fps_avg, rotation, codec}|null, model{model_id, sha256,
  device, precision}|null, analysis_fps}`; `JobList {items, next_cursor}`.
- `JobResults {job_id, state, summary{analyzed_frames, decoded_frames, coverage_ratio, duration_s, effective_analyzed_fps,
  timing_ms{decode,inference_p50,inference_p95,postprocess,encode,total_wall}, decision_counts{state:count},
  reason_counts{code:count}, tracks[{track_id, class_name, first_t, last_t, frames, max_score}],
  class_counts{name:count}, audio:"removed"}, artifacts[{name, url, content_type, bytes}], timeline_url, limitations[]}`.
  Artifact names allowlist: `input`, `annotated.mp4`, `timeline.json`, `report.json`, `report.html`, `poster.jpg`.
- `timeline.json`: `{schema_version, job_id, video{...}, model{...}, analysis_fps, frames:[WorldState + {"pts_s", "timing_ms",
  "lane_points"?: [[x,y],...]}]}` (every analyzed frame; detections in original pixels).
- `Simulator {id, name, available, reason, perception_modes[{id, available, reason}], scenarios[{id, available, reason}],
  road_graph{nodes[{id, x, y, label_uz, label_en, reachable_from_default}], edges[{id, from, to, polyline[[x,y]...]}],
  units:"m", frame:"ENU x-east y-north yaw CCW"}, default_start}`.
- `RunCreate {simulator:"lite", perception_source:"synthetic_rgb_cv"|"learned_model", start_node, destination_node,
  scenario, seed:int(0..1_000_000)}`.
- `Run {id, state, outcome, simulator, perception_source, scenario, seed, start_node, destination_node,
  route{node_ids, polyline[[x,y]...], length_m}, created_utc, started_utc, finished_utc, error_code, error_message,
  live{frame_id, sim_time_s, wall_time_s, realtime_factor, ego{source:"ego_telemetry", x_m, y_m, yaw_rad, speed_mps},
       decision{state, reason_codes, action_kind}, command{steer, throttle, brake, input_frame_id, source, safety_override,
       reason_codes}|null, perception{source, quality_status, lane_status, image_lane_offset_normalized, image_obstacle_cue,
       obstacle_boxes[[x1,y1,x2,y2]], lane_points[[x,y]], frame_width, frame_height}, route_progress(0..1),
       acquisition_age_s}|null}`.
- `RunControl {action:"pause"|"resume"|"stop"}` → returns `Run`. Idempotent: stop after terminal returns current state.
- `ObstacleInject {distance_ahead_m: 8..40, lateral_offset_m: -1.5..1.5}`; max 5 per run; 409 if run not running/paused.
- `GET /runs/{id}/frame` → `image/jpeg`, headers `X-Frame-Id`, `X-Sim-Time`, `X-Perception-Source`, `Cache-Control: no-store`;
  204 when no frame yet.
- `GET /runs/{id}/report` → run-report JSON (contract) + extra `details`; 409 `report_unavailable` while running.
- `Evaluation {id, kind:"synthetic_scenario_suite"|"model_benchmark"|"video_ml_accuracy", state, created_utc, finished_utc,
  source, sample_count, metrics{name:{value,unit,sample_count,status,note}}, limitations[], code_commit, model_sha256}`;
  `POST /api/v1/evaluations` `{kind:"synthetic_scenario_suite", seeds:[1,2,3]}` enqueues the suite.
- `Capabilities {build{commit, version, dirty}, video_analysis{available, reason, model{...}|null, device, precision,
  supported_classes[], unsupported_examples["person","traffic_light","traffic_sign"]}, simulation{simulators[...]},
  queue{video_queued, video_running, capacity_total, per_session, sim_active}, limits{...}, monitoring{metrics:true}}`.
- `GET /api/v1/events` SSE: `event: <type>`, `id: <redis stream id>`, `data: <EventEnvelope JSON>` per
  contracts/event.schema.json; `Last-Event-ID` resumes; ping comment every 15 s.
  Payloads: `job_state{job_id,state,error_code}`, `job_progress{job_id,progress,stage,frames_done,frames_total}`,
  `run_state{run_id,state,outcome}`, `frame_ready{run_id,frame_id,sim_time_s}` (≤5 Hz), `decision{run_id,frame_id,state,
  reason_codes,command}`, `capability_changed{...}`, `error{code,message}`.
- Admin (operator role): `POST /api/v1/admin/cleanup` runs retention cleanup now.

### 3.5 Reason codes (`app/schemas/reason_codes.py`, enum `ReasonCode`, mirrored by frontend i18n keys)
Video: `LANE_VALID, LANE_UNCERTAIN, LANE_UNAVAILABLE, DRIVABLE_AREA_LOW, VEHICLE_IN_CORRIDOR_NEAR, VEHICLE_IN_CORRIDOR,
NO_VEHICLE_IN_CORRIDOR, FRAME_DARK, FRAME_BLURRY, INFERENCE_FAILED`.
Simulation: `INITIALIZING, ROUTE_FOLLOWING, CURVE_SPEED_LIMIT, OBSTACLE_AHEAD_SLOW, OBSTACLE_AHEAD_STOP, OBSTACLE_CLEARED,
LANE_UNCERTAIN_SLOW, CAMERA_BLACK, CAMERA_INVALID, FRAME_STALE, PERCEPTION_TIMEOUT, OPERATOR_PAUSE, OPERATOR_STOP,
ARRIVED_AT_DESTINATION, CONTROL_CLAMPED, CONTROL_NAN_REJECTED, EGO_OUT_OF_BOUNDS, WALL_TIME_LIMIT, TICK_LIMIT,
COLLISION_DETECTED_BY_EVALUATOR`.
Explanation text is rendered from these codes by the frontend templates; no free text generation.

## 4. Simulation conventions (LiteRoad)
- World frame ENU: x east (m), y north (m), yaw rad CCW from +x, right-handed, z up. Steering command +1 = full left
  (positive yaw rate). Max steer angle 0.5 rad, wheelbase 2.6 m, kinematic bicycle, dt 0.05 s, speed ∈ [0, 5] m/s,
  max accel 2.5 m/s², max brake decel 6 m/s², rolling drag 0.3 m/s².
- Roads: two-way, width 7 m (2 × 3.5 m lanes), right-hand traffic: ego follows the lane centre 1.75 m right of the
  centreline. Markings: white solid edge lines at ±3.5 m, yellow dashed centre line; markings omitted inside junction
  boxes (nodes with degree ≥ 3). Grass outside the road. Obstacles: orange/white striped boxes 1.0 × 1.0 × 1.2 m.
- Camera: pinhole, 640×360, horizontal FOV 90°, mounted 1.4 m above ground at the ego origin + 1.0 m forward, pitch 8°
  down, no roll. `CameraCalibration` (intrinsics, height, pitch) is static and may be given to perception.
- Road graph: ~10 nodes, curved edges (polylines), one node not connected to the rest (destination unreachable → 422
  `destination_unreachable`). Graph is static topology (allowed runtime input).
- Adapter (`app/simulators/base.py`): `reset(seed, scenario) -> None`, `get_rgb_frame() -> RgbFrame(frame_id,
  sim_time_s, image(np.uint8 HxWx3 RGB), acquired_wall_monotonic)`, `get_ego_telemetry() -> EgoTelemetry`,
  `get_route_graph() -> RoadGraph`, `get_camera_calibration()`, `apply_control(Command)`, `step(dt)`, `close()`.
  Evaluator-only: `adapter.evaluator() -> EvaluatorTruth` (obstacle truth, collisions, lateral deviation, route progress).
  The `SimulationSession` passes **only** the runtime adapter view (no evaluator, no scene object list) to
  perception/FSM/controller. Tests enforce this (hidden-truth isolation, pixel ablation).
- Scenarios: `normal`, `bend` (route through the curved edges), `obstacle_stop` (obstacle on route ~25 m ahead ± seed jitter),
  `camera_blackout` (camera returns black frames from t=4 s), `stale_frame` (camera freezes frame id from t=4 s),
  `pause_resume` (test-only helper), all deterministic for a fixed seed.
- Loop per tick: frame → validate (age, id monotonic, black/invalid) → perception → world state → FSM → pure pursuit
  (route) + pixel lane correction → PID speed → safety supervisor clamp/override → apply → step(dt) → evaluator measures.
- FSM precedence: operator stop/failure → safety_stop → paused → arrived → controlled_stop/slow → follow_route.
  Hysteresis: obstacle stop enter cue ≥ 0.70 / exit ≤ 0.45 with 1.0 s dwell; slow enter ≥ 0.35 / exit ≤ 0.25.
- Terminal: arrived (≤ 2.0 m from destination lane point and speed < 0.5 m/s); stopped (operator stop or safety/controlled
  stop held 5 s sim time with stop reason → outcome e.g. `stopped_before_obstacle`, `stopped_camera_invalid`);
  failed (collision, out of bounds, internal error, wall limit 120 s → `WALL_TIME_LIMIT`, tick limit 2400).
- Pacing: the worker sleeps so simulated time does not run faster than wall time (realtime_factor ≤ 1.0, measured).
- Redis keys (TTL 60 s): `dd:run:{id}:frame` (JPEG bytes), `dd:run:{id}:live` (Run.live JSON), `dd:run:{id}:control`
  (pending action), `dd:run:{id}:inject` (list of ObstacleInject JSON).

### 4.1 `app/agents/runner.py` (owned by the simulation component, called by `app/jobs/sim_job.py`)
```python
class RouteUnreachable(ValueError): ...
class UnsupportedMode(ValueError): ...           # e.g. learned_model while not validated, unknown scenario

@dataclass
class SessionConfig:
    run_id: str
    start_node: str
    destination_node: str
    scenario: str = "normal"
    seed: int = 42
    perception_source: str = "synthetic_rgb_cv"   # or "learned_model" (needs inference_client + validation flag)
    dt: float = 0.05
    max_ticks: int = 2400

@dataclass
class TickResult:
    frame_id: int
    sim_time_s: float
    jpeg: bytes                  # the exact RGB frame perception consumed, JPEG-encoded for the UI
    world_state: WorldState      # mode="simulation", action_kind="simulator_control"
    live: dict                   # exactly the Run.live shape of §3.4 (JSON-serialisable)
    run_state: str               # initializing|running|paused|stopping|stopped|arrived|failed
    outcome: str | None          # e.g. arrived, stopped_operator, stopped_before_obstacle, stopped_camera_invalid,
                                 # failed_collision, failed_out_of_bounds, failed_wall_time, failed_tick_limit
    terminal: bool

class SimulationSession:
    @staticmethod
    def road_graph() -> RoadGraph: ...                       # static LiteRoad topology for /simulators
    @staticmethod
    def scenarios() -> list[dict]: ...                       # [{id, available, reason}]
    @staticmethod
    def plan_route(start: str, dest: str) -> dict: ...       # {node_ids, polyline, length_m}; raises RouteUnreachable
    def __init__(self, cfg: SessionConfig, inference_client=None): ...   # validates; raises RouteUnreachable/UnsupportedMode
    def start(self) -> TickResult: ...                       # reset(seed, scenario), capture initial frame, no stepping
    def tick(self) -> TickResult: ...                        # one closed-loop step; when paused: no physics step
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def request_stop(self) -> None: ...                      # operator stop → brake to standstill → stopped
    def inject_obstacle(self, distance_ahead_m: float, lateral_offset_m: float) -> bool: ...
    def note_wall_time(self, wall_elapsed_s: float) -> None: ...   # watchdog: wall limit → failed_wall_time
    def report(self, code_commit: str, wall_time_s: float) -> dict: ...  # RunReport dump + "details" (commands summary,
                                                             # per-tick decision counts, collisions, deviations, rtf)
```

## 5. Video pipeline (`app/video/pipeline.py`)
`run_video_analysis(job_id, input_path, out_dir, client, settings, progress_cb(frames_done, frames_total, stage),
cancel_check() -> bool) -> VideoAnalysisSummary`.
1. probe (ffprobe JSON, 15 s timeout): container mov/mp4 family, ≥1 video stream, duration ≤ MAX_VIDEO_SECONDS,
   decoded display size ≤ 1920×1080 (after rotation), codec decodable. Errors → `video_invalid`, `video_too_long`,
   `video_too_large_dimensions`, `video_no_stream`, `video_decode_failed`.
2. decode with PyAV: real PTS seconds (`frame.time`), rotation from stream side data applied, decode watchdog.
   Sample frames by PTS grid at `ANALYSIS_FPS` (first frame with pts ≥ k/fps), never by `index/fps`.
3. per sample: JPEG q95 → `InferenceClient.infer` → tracker.update(dets, pts) (ByteTrack high 0.5 / low 0.1 / match 0.8,
   low-score boxes kept for the second association) → video decisions → WorldState(mode=video_analysis, ego=null,
   command=null, action_kind=analytical_only, perception.source=learned_model with model id/sha).
4. annotate each analyzed frame (drivable mask green α0.35, lane mask cyan, boxes+track ids+score, HUD: source
   "YOLOPv2 · learned_model · CPU", PTS, decision state) and encode CFR at ANALYSIS_FPS → `annotated.mp4`
   (H.264 yuv420p, +faststart, no audio; report says `audio: removed`). Each output frame's original PTS is in timeline.json.
5. write `timeline.json`, `report.json` (run-report contract, kind video_analysis), `report.html`, `poster.jpg` into a temp
   dir, then atomic rename into `artifacts/jobs/{job_id}/`. Cancellation is checked between frames → state cancelled,
   partial outputs removed.
Timing captured: decode, inference (client-side round trip and server-reported), postprocess, encode, total wall;
effective analyzed FPS = analyzed frames / total wall.

## 6. Frontend (`frontend/`)
React 18 + TypeScript + Vite + Tailwind CSS, react-router. `src/api/openapi.json` generated by the backend,
`src/api/schema.d.ts` generated by `openapi-typescript`; `src/api/client.ts` uses relative `/api/v1` URLs,
`credentials: "same-origin"`, adds `X-CSRF-Token` from the session response, parses the error envelope.
i18n: `src/i18n/uz.ts` (default, Uzbek Latin with ‘ U+2018 apostrophes) and `src/i18n/en.ts`; reason-code templates.
Font: `@fontsource/inter` (served locally). Pages: Umumiy, Video tahlili, Simulyatsiya, Baholash, Tizim.
`data-testid` on interactive elements used by Playwright (`frontend/e2e/*.spec.ts`).

## 6.1 Tests layout
`tests/unit/backend/` (backend test image), `tests/unit/inference/` (inference test image), `tests/integration/` (backend
test image against a running compose stack: real redis + inference), `tests/model_smoke/` (inference image + real weights),
`tests/scenarios/` (LiteRoad closed-loop scenarios, backend image), `tests/security/` (ownership, CSRF, traversal, upload
abuse), `frontend/e2e/` (Playwright against the running gateway). Fixtures: `tests/fixtures/` small licensed files only,
each listed with source/license in `docs/DATA_SOURCES.md`. Each test directory has its own `conftest.py`; there is no
root-level conftest that imports either `app` package. Integration/e2e/model-smoke tests must fail, not skip, when their
required service or weights are absent while running under `make qa` (`DD_REQUIRE_LIVE=1`).

## 7. Make targets (all must exit non-zero on failure)
`doctor, lint, typecheck, test, integration, model-init, model-smoke, e2e, scenarios, security, qa, up, down, logs,
build, smoke-published, deploy-check, backup, backup-test, restore-test, rollback, gen-types, eval-scenarios`.
Python gates run inside the backend/inference `test` images (no host Python needed). `GIT_COMMIT` is exported from
`git rev-parse HEAD` and passed as build arg. QA evidence goes to `artifacts/qa/<commit>/`.
