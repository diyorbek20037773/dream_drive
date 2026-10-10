/**
 * LAYOUT FIXTURES — hand-written, obviously synthetic API responses used ONLY by the dev-only layout spec to render
 * populated pages for responsive-layout inspection. They are not measurements, are never shipped in the bundle and
 * are never used as evidence. Hashes are a repeating placeholder pattern on purpose.
 */
const FAKE_SHA = '0123456789abcdef'.repeat(4);
const NOW = new Date('2026-10-10T09:00:00Z').getTime();
const iso = (offsetS: number) => new Date(NOW + offsetS * 1000).toISOString();

export const limits = {
  accepted_extensions: ['mp4', 'mov'],
  analysis_fps: 5,
  artifact_retention_hours: 24,
  max_active_simulations: 1,
  max_expensive_requests_per_minute: 10,
  max_obstacles_per_run: 5,
  max_queued_per_session: 2,
  max_queued_total: 10,
  max_simulation_wall_s: 120,
  max_upload_bytes: 104857600,
  max_video_duration_s: 120,
  max_video_height_px: 1080,
  max_video_width_px: 1920,
};

export const session = {
  csrf_token: 'layout-fixture-token',
  expires_utc: new Date(Date.now() + 86_400_000).toISOString(),
  limits,
  role: 'anonymous',
  session_id_suffix: 'fx01',
};

const nodes = [
  { id: 'n1', x: 0, y: 0, label_uz: 'Markaz', label_en: 'Centre' },
  { id: 'n2', x: 60, y: 0, label_uz: 'Bozor', label_en: 'Market' },
  { id: 'n3', x: 120, y: 0, label_uz: 'Ko‘prik', label_en: 'Bridge' },
  { id: 'n4', x: 120, y: 60, label_uz: 'Maktab', label_en: 'School' },
  { id: 'n5', x: 60, y: 60, label_uz: 'Park', label_en: 'Park' },
  { id: 'n6', x: 0, y: 60, label_uz: 'Kutubxona', label_en: 'Library' },
  { id: 'n7', x: -50, y: 30, label_uz: 'G‘arb', label_en: 'West' },
  { id: 'n8', x: 175, y: 30, label_uz: 'Sharq', label_en: 'East' },
  { id: 'n9', x: 60, y: 115, label_uz: 'Shimol', label_en: 'North' },
  { id: 'n10', x: 190, y: 120, label_uz: 'Orol', label_en: 'Island' },
].map((node) => ({ ...node, reachable_from_default: node.id !== 'n10' }));

const curve = (a: [number, number], b: [number, number], bend: number): number[][] => {
  const points: number[][] = [];
  for (let i = 0; i <= 8; i += 1) {
    const s = i / 8;
    const x = a[0] + (b[0] - a[0]) * s;
    const y = a[1] + (b[1] - a[1]) * s + Math.sin(Math.PI * s) * bend;
    points.push([x, y]);
  }
  return points;
};
const xy = (id: string): [number, number] => {
  const node = nodes.find((item) => item.id === id);
  return [node?.x ?? 0, node?.y ?? 0];
};
const edgeDefs: [string, string, number][] = [
  ['n1', 'n2', 0],
  ['n2', 'n3', 0],
  ['n3', 'n4', 8],
  ['n4', 'n5', 0],
  ['n5', 'n6', 0],
  ['n6', 'n1', -6],
  ['n2', 'n5', 0],
  ['n6', 'n7', 6],
  ['n7', 'n1', 6],
  ['n3', 'n8', 10],
  ['n4', 'n8', -10],
  ['n5', 'n9', 0],
];
const edges = edgeDefs.map(([from, to, bend], index) => ({ id: `e${index + 1}`, from, to, polyline: curve(xy(from), xy(to), bend) }));

export const simulator = {
  id: 'lite',
  name: 'LiteRoad',
  available: true,
  reason: null,
  default_start: 'n1',
  perception_modes: [
    { id: 'synthetic_rgb_cv', available: true, reason: null },
    { id: 'learned_model', available: false, reason: 'not_validated' },
  ],
  scenarios: ['normal', 'bend', 'obstacle_stop', 'camera_blackout', 'stale_frame', 'pause_resume'].map((id) => ({
    id,
    available: id !== 'pause_resume',
    reason: id === 'pause_resume' ? 'test-only scenario' : null,
  })),
  road_graph: { nodes, edges, lane_width_m: 3.5, units: 'm', frame: 'ENU x-east y-north yaw CCW' },
};

export const capabilities = {
  build: { commit: 'a'.repeat(40), version: 'layout-fixture', dirty: false },
  video_analysis: {
    available: true,
    reason: null,
    device: 'cpu',
    precision: 'fp32',
    model: {
      model_id: 'yolopv2',
      ready: true,
      reason: null,
      sha256: FAKE_SHA,
      size_bytes: 1,
      repository_revision: 'f'.repeat(40),
      class_map: { '0': 'vehicle' },
      device: 'cpu',
      precision: 'fp32',
      torch_version: 'fixture',
      input_policy: 'letterbox640_stride32',
      models: [
        { model_id: 'yolopv2', ready: true, sha256: FAKE_SHA, device: 'cpu', precision: 'fp32', class_map: { '3': 'vehicle' } },
        { model_id: 'rfdetr_nano', ready: true, sha256: FAKE_SHA, device: 'cpu', precision: 'fp32', class_map: { '101': 'person', '110': 'traffic_light' } },
      ],
    },
    supported_classes: ['vehicle'],
    unsupported_examples: ['person', 'traffic_light', 'traffic_sign'],
  },
  simulation: { simulators: [simulator, { ...simulator, id: 'carla', name: 'CARLA', available: false, reason: 'carla_not_configured', road_graph: null, default_start: null }] },
  queue: { video_queued: 1, video_running: 1, capacity_total: 10, per_session: 2, sim_active: 0, sim_capacity: 1, reason: null },
  limits,
  monitoring: { metrics: true },
};

const model = { model_id: 'yolopv2', sha256: FAKE_SHA, device: 'cpu', precision: 'fp32' };
const baseJob = {
  kind: 'video_analysis',
  analysis_fps: 5,
  input: { sha256: FAKE_SHA, bytes: 8_400_000, ext: 'mp4' },
  video: { duration_s: 12, width: 1280, height: 720, fps_avg: 30, rotation: 0, codec: 'h264' },
  model,
  error_code: null,
  error_message: null,
};
export const jobs = [
  { ...baseJob, id: 'fx0001aa'.padEnd(32, '0'), state: 'succeeded', stage: 'report', progress: 1, frames_done: 60, frames_total: 60, created_utc: iso(-600), started_utc: iso(-590), finished_utc: iso(-420) },
  { ...baseJob, id: 'fx0002bb'.padEnd(32, '0'), state: 'running', stage: 'inference', progress: 0.42, frames_done: 25, frames_total: 60, created_utc: iso(-120), started_utc: iso(-100), finished_utc: null },
  { ...baseJob, id: 'fx0003cc'.padEnd(32, '0'), state: 'failed', stage: 'probe', progress: 0, frames_done: 0, frames_total: null, created_utc: iso(-900), started_utc: iso(-899), finished_utc: iso(-898), error_code: 'video_decode_failed', error_message: 'decoder error (fixture)' },
];

const reasonsFor = (k: number): { state: string; reasons: string[] } => {
  if (k < 12) return { state: 'observe', reasons: ['LANE_VALID', 'NO_VEHICLE_IN_CORRIDOR'] };
  if (k < 25) return { state: 'slow', reasons: ['LANE_VALID', 'VEHICLE_IN_CORRIDOR'] };
  if (k < 31) return { state: 'controlled_stop', reasons: ['VEHICLE_IN_CORRIDOR_NEAR'] };
  if (k < 45) return { state: 'observe', reasons: ['LANE_UNCERTAIN', 'NO_VEHICLE_IN_CORRIDOR'] };
  return { state: 'observe', reasons: ['LANE_VALID', 'TRAFFIC_LIGHT_VISIBLE', 'TRAFFIC_LIGHT_RED_HEURISTIC'] };
};
export const timeline = {
  schema_version: '1.0',
  job_id: jobs[0]?.id,
  analysis_fps: 5,
  frames: Array.from({ length: 60 }, (_, k) => {
    const decision = reasonsFor(k);
    return {
      pts_s: k * 0.2,
      frame: { id: k, timestamp_seconds: k * 0.2, width: 1280, height: 720, quality_status: 'valid' },
      perception: {
        source: 'learned_model',
        model_id: 'yolopv2',
        model_sha256: FAKE_SHA,
        detections: [
          ...(k >= 12 && k < 31
            ? [{ track_id: 3, class_id: 0, class_name: 'vehicle', score: 0.8, xyxy_pixels: [560, 330, 760, 470] }]
            : []),
          ...(k >= 40
            ? [{ track_id: 9, class_id: 110, class_name: 'traffic_light', score: 0.6, xyxy_pixels: [900, 80, 930, 160], model_id: 'rfdetr_nano', light_state: 'red' }]
            : []),
        ],
        lane_status: decision.reasons.includes('LANE_UNCERTAIN') ? 'uncertain' : 'valid',
        image_lane_offset_normalized: 0.05,
        image_obstacle_cue: k >= 12 && k < 31 ? 0.6 : 0.1,
        uncertainty_reasons: [],
      },
      decision: { state: decision.state, reason_codes: decision.reasons, action_kind: 'analytical_only' },
      timing_ms: { inference: 400 },
    };
  }),
};

export const results = {
  job_id: jobs[0]?.id,
  state: 'succeeded',
  mode: 'video_analysis',
  model,
  perception_source: 'learned_model',
  summary: {
    analyzed_frames: 60,
    decoded_frames: 360,
    coverage_ratio: 1,
    duration_s: 12,
    effective_analyzed_fps: 0.35,
    timing_ms: { decode: 900, inference_p50: 410, inference_p95: 520, postprocess: 1200, encode: 800, total_wall: 170000 },
    decision_counts: { observe: 35, slow: 13, controlled_stop: 6, follow_route: 0 },
    reason_counts: { LANE_VALID: 40, NO_VEHICLE_IN_CORRIDOR: 29, VEHICLE_IN_CORRIDOR: 13, VEHICLE_IN_CORRIDOR_NEAR: 6, LANE_UNCERTAIN: 14 },
    tracks: [
      { track_id: 3, class_name: 'vehicle', first_t: 2.4, last_t: 6.0, frames: 19, max_score: 0.8 },
      { track_id: 7, class_name: 'vehicle', first_t: 9.0, last_t: 9.6, frames: 4, max_score: 0.4 },
    ],
    class_counts: { vehicle: 23 },
    audio: 'removed',
  },
  artifacts: ['annotated.mp4', 'input', 'report.json', 'report.html', 'timeline.json', 'poster.jpg'].map((name) => ({
    name,
    url: `/api/v1/jobs/${jobs[0]?.id}/artifacts/${name}`,
    content_type: name.endsWith('.json') ? 'application/json' : name.endsWith('.html') ? 'text/html' : name.endsWith('.jpg') ? 'image/jpeg' : 'video/mp4',
    bytes: 1_234_567,
  })),
  timeline_url: `/api/v1/jobs/${jobs[0]?.id}/artifacts/timeline.json`,
  limitations: ['LAYOUT FIXTURE: not a real analysis result.', 'Person, traffic light and sign classes are not supported by this checkpoint.'],
};

export const runId = 'fxrun001'.padEnd(32, '0');
/** LAYOUT FIXTURE decision-basis item (shape of Run.live.decision_basis; values are not measurements). */
function basis(
  key: string,
  value: number | string | boolean | null,
  source: string,
  unit: string | null,
  threshold: number | null,
  comparator: string | null,
  triggered: boolean,
  effect: string | null,
) {
  return { key, value, source, unit, threshold, comparator, triggered, effect, input_frame_id: 120 };
}
/** LAYOUT FIXTURE maneuver changes sent on the fixture event stream (not measurements). */
export const decisionEvents = [
  { frame_id: 2, sim_time_s: 0.1, state: 'follow_route', reason_codes: ['ROUTE_FOLLOWING'], primary: 'ACCELERATE', longitudinal: 'accelerating' },
  { frame_id: 70, sim_time_s: 3.5, state: 'follow_route', reason_codes: ['ROUTE_FOLLOWING'], primary: 'CRUISE', longitudinal: 'cruising' },
  { frame_id: 108, sim_time_s: 5.4, state: 'slow', reason_codes: ['OBSTACLE_AHEAD_SLOW'], primary: 'SLOW_DOWN', longitudinal: 'decelerating' },
];
const routePolyline = [...curve(xy('n1'), xy('n2'), 0), ...curve(xy('n2'), xy('n3'), 0).slice(1), ...curve(xy('n3'), xy('n4'), 8).slice(1)].map(
  ([x = 0, y = 0]) => [x, y - 1.75],
);
export const liveFrameId = 120;
export const run = {
  id: runId,
  state: 'running',
  outcome: null,
  simulator: 'lite',
  perception_source: 'synthetic_rgb_cv',
  scenario: 'obstacle_stop',
  seed: 42,
  start_node: 'n1',
  destination_node: 'n4',
  route: { node_ids: ['n1', 'n2', 'n3', 'n4'], polyline: routePolyline, length_m: 184.2 },
  created_utc: iso(-30),
  started_utc: iso(-29),
  finished_utc: null,
  error_code: null,
  error_message: null,
  live: {
    frame_id: liveFrameId,
    sim_time_s: 6.0,
    wall_time_s: 6.4,
    realtime_factor: 0.94,
    ego: { source: 'ego_telemetry', x_m: 78, y_m: -1.75, yaw_rad: 0, speed_mps: 2.1 },
    decision: { state: 'slow', reason_codes: ['OBSTACLE_AHEAD_SLOW', 'ROUTE_FOLLOWING'], action_kind: 'simulator_control' },
    command: { steer: 0.12, throttle: 0, brake: 0.35, input_frame_id: liveFrameId, source: 'dreamdrive_controller', safety_override: false, reason_codes: ['OBSTACLE_AHEAD_SLOW'] },
    perception: {
      source: 'synthetic_rgb_cv',
      quality_status: 'valid',
      lane_status: 'valid',
      image_lane_offset_normalized: -0.04,
      image_obstacle_cue: 0.52,
      obstacle_boxes: [[292, 196, 348, 248]],
      lane_points: [[150, 360], [215, 320], [262, 280], [298, 240], [320, 210]],
      frame_width: 640,
      frame_height: 360,
    },
    route_progress: 0.42,
    acquisition_age_s: 0.021,
    maneuver: {
      primary: 'SLOW_DOWN',
      longitudinal: 'decelerating',
      lateral: 'straight',
      since_sim_time_s: 5.4,
      input_frame_id: liveFrameId,
      speed_mps: 2.1,
      accel_mps2: -1.6,
      yaw_rate_radps: 0.004,
      signal_source: 'ego_telemetry_window',
    },
    decision_basis: [
      basis('frame_received', true, 'safety_supervisor', null, null, null, false, 'safety_stop'),
      basis('frame_wall_age_s', 0.021, 'safety_supervisor', 's', 1.0, '>', false, 'safety_stop'),
      basis('frame_quality', 'valid', 'safety_supervisor', null, null, null, false, 'safety_stop'),
      basis('operator_stop', false, 'operator', null, null, null, false, 'safety_stop'),
      basis('safety_codes', null, 'safety_supervisor', null, null, null, false, 'safety_stop'),
      basis('destination_distance_m', 96.4, 'route_graph', 'm', 2.0, '<=', false, 'arrive'),
      basis('image_obstacle_cue', 0.52, 'synthetic_rgb_cv', null, 0.35, '>=', true, 'slow'),
      basis('lane_status', 'valid', 'synthetic_rgb_cv', null, null, null, false, 'slow'),
      basis('pixel_lane_disagreement_m', 0.12, 'synthetic_rgb_cv', 'm', 1.0, '<=', true, 'steer_correction'),
      basis('pixel_blend_weight', 0.5, 'synthetic_rgb_cv', null, null, null, true, 'steer_correction'),
      basis('ego_speed_mps', 2.1, 'ego_telemetry', 'm/s', null, null, false, null),
      basis('curve_speed_limit_mps', null, 'route_graph', 'm/s', 2.5, '<', false, 'speed_limit'),
      basis('target_speed_mps', 2.5, 'controller', 'm/s', null, null, false, null),
      basis('safety_override', false, 'safety_supervisor', null, null, null, false, 'safety_stop'),
    ],
  },
};

/** A drawn road scene used as the "camera" image for layout only. */
export const frameSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 360" width="640" height="360">
<rect width="640" height="180" fill="#9cc3e6"/><rect y="180" width="640" height="180" fill="#5f8f4e"/>
<polygon points="40,360 600,360 350,190 290,190" fill="#55595e"/>
<polygon points="150,360 160,360 300,190 297,190" fill="#ffffff"/>
<polygon points="480,360 490,360 343,190 340,190" fill="#ffffff"/>
<rect x="296" y="198" width="48" height="48" fill="#f97316"/><rect x="296" y="214" width="48" height="8" fill="#ffffff"/>
<text x="12" y="350" font-size="12" fill="#ffffff" font-family="sans-serif">LAYOUT FIXTURE</text></svg>`;

export const evaluations = [
  {
    id: 'fxeval01'.padEnd(32, '0'),
    kind: 'synthetic_scenario_suite',
    state: 'succeeded',
    created_utc: iso(-3600),
    finished_utc: iso(-3300),
    source: 'literoad_synthetic_rgb_cv',
    sample_count: 15,
    seeds: [1, 2, 3],
    metrics: {
      completion_rate: { value: 0.8, unit: 'ratio', sample_count: 15, status: 'measured', note: 'fixture' },
      collisions: { value: 0, unit: 'count', sample_count: 15, status: 'measured', note: 'fixture' },
      mean_abs_lane_deviation_m: { value: 0.21, unit: 'm', sample_count: 15, status: 'measured', note: 'fixture' },
    },
    limitations: ['LAYOUT FIXTURE: synthetic environment only.'],
    code_commit: 'a'.repeat(40),
    model_sha256: null,
    error_code: null,
  },
  {
    id: 'fxeval02'.padEnd(32, '0'),
    kind: 'video_ml_accuracy',
    state: 'succeeded',
    created_utc: iso(-7200),
    finished_utc: iso(-7100),
    source: 'none',
    sample_count: 0,
    seeds: null,
    metrics: {
      lane_iou: { value: null, unit: 'ratio', sample_count: 0, status: 'not_evaluated', note: 'not evaluated: no labeled subset' },
      vehicle_map50: { value: null, unit: 'ratio', sample_count: 0, status: 'not_evaluated', note: 'not evaluated: no labeled subset' },
    },
    limitations: ['No labeled evaluation subset exists.'],
    code_commit: 'a'.repeat(40),
    model_sha256: FAKE_SHA,
    error_code: null,
  },
];

export const report = {
  schema_version: '1.0',
  run_id: runId,
  kind: 'simulation',
  status: 'stopped',
  perception_source: 'synthetic_rgb_cv',
  code_commit: 'a'.repeat(40),
  model_sha256: null,
  input_manifest_sha256: null,
  generated_utc: iso(0),
  metrics: {
    route_completion: { value: 0.46, unit: 'ratio', sample_count: 1, status: 'measured', note: 'fixture' },
    collisions: { value: 0, unit: 'count', sample_count: 1, status: 'measured', note: 'fixture' },
    stop_distance_to_obstacle_m: { value: 3.4, unit: 'm', sample_count: 1, status: 'measured', note: 'fixture' },
    lateral_deviation_p95_m: { value: null, unit: 'm', sample_count: 0, status: 'not_evaluated', note: 'fixture: not measured' },
  },
  limitations: ['LAYOUT FIXTURE: not a real run report.'],
  evaluation_truth_used: true,
  runtime_used_dynamic_ground_truth: false,
  details: {
    ticks: 260,
    realtime_factor_mean: 0.95,
    decision_counts: { follow_route: 180, slow: 50, controlled_stop: 30 },
    maneuver_counts: { HOLD: 3, ACCELERATE: 35, SLOW_DOWN: 37, BRAKE: 24, STOP: 161 },
    maneuver_timeline: [
      { sim_time_s: 0, since_sim_time_s: 0, frame_id: 0, primary: 'HOLD', longitudinal: 'holding', lateral: 'straight', speed_mps: 0, reason_codes: ['INITIALIZING'] },
      { sim_time_s: 0.1, since_sim_time_s: 0, frame_id: 2, primary: 'ACCELERATE', longitudinal: 'accelerating', lateral: 'straight', speed_mps: 0.23, reason_codes: ['ROUTE_FOLLOWING'] },
      { sim_time_s: 1.85, since_sim_time_s: 1.75, frame_id: 37, primary: 'SLOW_DOWN', longitudinal: 'decelerating', lateral: 'straight', speed_mps: 3.64, reason_codes: ['OBSTACLE_AHEAD_SLOW'] },
      { sim_time_s: 3.7, since_sim_time_s: 3.7, frame_id: 74, primary: 'BRAKE', longitudinal: 'braking_to_stop', lateral: 'straight', speed_mps: 2.48, reason_codes: ['OBSTACLE_AHEAD_STOP'] },
      { sim_time_s: 4.9, since_sim_time_s: 4.9, frame_id: 98, primary: 'STOP', longitudinal: 'stopped', lateral: 'straight', speed_mps: 0.07, reason_codes: ['OBSTACLE_AHEAD_STOP'] },
    ],
    maneuver_timeline_truncated: false,
  },
};

export const health = { status: 'ok', checked_utc: iso(0), checks: { database: { ok: true, reason: null }, queue: { ok: true, reason: null } } };
