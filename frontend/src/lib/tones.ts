import type { DecisionState, EvaluationState, JobState, RunState } from '../api/types';

export type Tone = 'neutral' | 'info' | 'success' | 'warning' | 'danger' | 'cyan';

export const JOB_TONE: Record<JobState, Tone> = {
  queued: 'neutral',
  validating: 'info',
  running: 'info',
  cancelling: 'warning',
  cancelled: 'neutral',
  succeeded: 'success',
  failed: 'danger',
  interrupted: 'warning',
};
export const RUN_TONE: Record<RunState, Tone> = {
  initializing: 'info',
  running: 'info',
  paused: 'warning',
  stopping: 'warning',
  stopped: 'neutral',
  arrived: 'success',
  failed: 'danger',
};
export const DECISION_TONE: Record<DecisionState, Tone> = {
  observe: 'neutral',
  follow_route: 'info',
  slow: 'warning',
  controlled_stop: 'warning',
  safety_stop: 'danger',
  paused: 'neutral',
  arrived: 'success',
  failed: 'danger',
};
export const EVALUATION_TONE: Record<EvaluationState, Tone> = {
  queued: 'neutral',
  running: 'info',
  succeeded: 'success',
  failed: 'danger',
  interrupted: 'warning',
};
/** Solid colours for timeline segments (same semantic mapping as the pills). */
export const DECISION_FILL: Record<DecisionState, string> = {
  observe: '#94a3b8',
  follow_route: '#2563eb',
  slow: '#d97706',
  controlled_stop: '#b45309',
  safety_stop: '#b91c1c',
  paused: '#64748b',
  arrived: '#15803d',
  failed: '#7f1d1d',
};
