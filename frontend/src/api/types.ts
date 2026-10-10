/**
 * Convenient aliases over the generated OpenAPI types (src/api/schema.d.ts, produced by `npm run gen:types` from the
 * backend's exported OpenAPI). Never hand-edit schema.d.ts; extend aliases here.
 */
import type { components } from './schema';

type Schemas = components['schemas'];

export type Capabilities = Schemas['Capabilities'];
export type BuildInfo = Schemas['BuildInfo'];
export type Limits = Schemas['Limits'];
export type QueueStatus = Schemas['QueueStatus'];
export type VideoAnalysisCapability = Schemas['VideoAnalysisCapability'];
export type ModelInfo = Schemas['ModelInfo'];
/** Per-model entry of `ModelInfo.models` (section 2.1 ensemble); read defensively until it is in the OpenAPI. */
export interface ModelEntry {
  model_id: string;
  ready?: boolean;
  reason?: string | null;
  sha256?: string | null;
  device?: string | null;
  precision?: string | null;
  class_map?: Record<string, string>;
}
export type ModelIdentity = Schemas['ModelIdentity'];
export type SessionInfo = Schemas['SessionInfo'];
export type SessionRole = Schemas['SessionRole'];
export type HealthStatus = Schemas['HealthStatus'];

export type Job = Schemas['Job'];
export type JobList = Schemas['JobList'];
export type JobState = Schemas['JobState'];
export type JobResults = Schemas['JobResults'];
export type JobSummary = Schemas['JobSummary'];
export type JobArtifact = Schemas['JobArtifact'];
export type ArtifactName = Schemas['ArtifactName'];
export type TrackSummary = Schemas['TrackSummary'];
export type DeleteResult = Schemas['DeleteResult'];

export type Simulator = Schemas['Simulator'];
export type SimulatorList = Schemas['SimulatorList'];
export type SimulatorId = Schemas['SimulatorId'];
export type SimPerceptionSource = Schemas['SimPerceptionSource'];
export type ScenarioId = Schemas['ScenarioId'];
export type ScenarioInfo = Schemas['ScenarioInfo'];
export type PerceptionModeInfo = Schemas['PerceptionModeInfo'];
export type RoadGraph = Schemas['RoadGraphInfo'];
export type RoadNode = Schemas['RoadNode'];
export type RoadEdge = Schemas['RoadEdge'];
export type Run = Schemas['Run'];
export type RunList = Schemas['RunList'];
export type RunState = Schemas['RunState'];
export type RunLive = Schemas['RunLive'];
export type RunCreate = Schemas['RunCreate'];
export type RunAction = Schemas['RunAction'];
export type RunReport = Schemas['RunReportResponse'];
export type ObstacleInject = Schemas['ObstacleInject'];
export type ObstacleInjectResult = Schemas['ObstacleInjectResult'];
export type LiveCommand = Schemas['LiveCommand'];
export type LiveManeuver = Schemas['LiveManeuver'];
export type ManeuverPrimary = Schemas['ManeuverPrimary'];
export type ManeuverLongitudinal = Schemas['ManeuverLongitudinal'];
export type ManeuverLateral = Schemas['ManeuverLateral'];
export type ManeuverSignalSource = Schemas['ManeuverSignalSource'];
export type DecisionBasisItem = Schemas['DecisionBasisItem'];
export type DecisionBasisSource = Schemas['DecisionBasisSource'];
export type DecisionBasisEffect = Schemas['DecisionBasisEffect'];
export type DecisionBasisComparator = Schemas['DecisionBasisComparator'];
export type DecisionManeuver = Schemas['DecisionManeuver'];
export type LivePerception = Schemas['LivePerception'];

export type Evaluation = Schemas['Evaluation'];
export type EvaluationList = Schemas['EvaluationList'];
export type EvaluationKind = Schemas['EvaluationKind'];
export type EvaluationState = Schemas['EvaluationState'];
export type Metric = Schemas['Metric'];

export type ReasonCode = Schemas['ReasonCode'];
/**
 * Reason codes added with the RF-DETR Nano detector (docs/IMPLEMENTATION_SPEC.md section 2.1) that are not yet in the
 * generated OpenAPI enum. Once the backend exports them they simply overlap with ReasonCode.
 */
export type ExtraReasonCode =
  | 'PERSON_IN_CORRIDOR_NEAR'
  | 'PERSON_IN_CORRIDOR'
  | 'TWO_WHEELER_IN_CORRIDOR'
  | 'TRAFFIC_LIGHT_VISIBLE'
  | 'TRAFFIC_LIGHT_RED_HEURISTIC'
  | 'TRAFFIC_LIGHT_YELLOW_HEURISTIC'
  | 'TRAFFIC_LIGHT_GREEN_HEURISTIC'
  | 'STOP_SIGN_VISIBLE';
export type AnyReasonCode = ReasonCode | ExtraReasonCode;
export type DecisionState = Schemas['DecisionState'];
export type PerceptionSource = Schemas['PerceptionSource'];
export type QualityStatus = Schemas['QualityStatus'];
export type LaneStatus = Schemas['LaneStatus'];

export type ErrorCode = Schemas['ErrorCode'];
export type ErrorEnvelope = Schemas['ErrorEnvelope'];
/** Error codes produced by the browser client itself (no server envelope was received). */
export type ClientErrorCode = 'network_error' | 'service_offline' | 'bad_response' | 'aborted';
export type AnyErrorCode = ErrorCode | ClientErrorCode;

export type EventEnvelope = Schemas['EventEnvelope'];
export type EventType = EventEnvelope['type'];
export interface EventPayloads {
  job_state: Schemas['JobStatePayload'];
  job_progress: Schemas['JobProgressPayload'];
  run_state: Schemas['RunStatePayload'];
  frame_ready: Schemas['FrameReadyPayload'];
  decision: Schemas['DecisionPayload'];
  capability_changed: Schemas['CapabilityChangedPayload'];
  error: Schemas['ErrorPayload'];
}

export const TERMINAL_JOB_STATES: readonly JobState[] = ['cancelled', 'succeeded', 'failed', 'interrupted'];
export const ACTIVE_RUN_STATES: readonly RunState[] = ['initializing', 'running', 'paused', 'stopping'];
export const TERMINAL_RUN_STATES: readonly RunState[] = ['stopped', 'arrived', 'failed'];

export function isTerminalJob(state: JobState): boolean {
  return TERMINAL_JOB_STATES.includes(state);
}
export function isActiveRun(state: RunState): boolean {
  return ACTIVE_RUN_STATES.includes(state);
}

/**
 * `timeline.json` artifact (docs/IMPLEMENTATION_SPEC.md section 3.4): one WorldState per analyzed frame
 * (contracts/world-state.schema.json) plus `pts_s`, `timing_ms` and optional `lane_points`. It is downloaded as an
 * artifact, so it is not part of the OpenAPI document; the fields read by the UI are declared here.
 */
export interface TimelineDetection {
  track_id: number | null;
  class_id: number;
  class_name: string;
  score: number;
  xyxy_pixels: [number, number, number, number];
  /** Which checkpoint produced the box (ensemble, section 2.1); optional. */
  model_id?: string;
  /** Traffic-light colour from a classical HSV heuristic (never model output); optional. */
  light_state?: 'red' | 'yellow' | 'green' | 'unknown';
}
export interface TimelineFrame {
  pts_s: number;
  frame: {
    id: number;
    timestamp_seconds: number;
    width: number;
    height: number;
    quality_status: QualityStatus;
  };
  perception: {
    source: PerceptionSource;
    model_id: string | null;
    model_sha256: string | null;
    detections: TimelineDetection[];
    lane_status: LaneStatus;
    image_lane_offset_normalized: number | null;
    image_obstacle_cue: number | null;
    uncertainty_reasons: string[];
  };
  decision: { state: DecisionState; reason_codes: string[]; action_kind: string };
  timing_ms?: Record<string, number | null>;
  lane_points?: number[][];
}
export interface TimelineFile {
  schema_version: string;
  job_id: string;
  analysis_fps: number;
  video?: { width?: number; height?: number; duration_s?: number } | null;
  model?: Partial<ModelIdentity> | null;
  frames: TimelineFrame[];
}
