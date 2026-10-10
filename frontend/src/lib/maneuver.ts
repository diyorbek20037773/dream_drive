/**
 * Display mapping for maneuvers (Run.live.maneuver, app.control.maneuver on the backend) and for the analysis-only
 * recommendation shown for uploaded videos. Labels live in i18n (`t.maneuver`); this module holds icons and tones.
 */
import type { LucideIcon } from 'lucide-react';
import { ArrowDown, ArrowUp, ArrowUpLeft, ArrowUpRight, Equal, OctagonX, Pause } from 'lucide-react';

import type { DecisionBasisComparator, DecisionBasisSource, DecisionState, ManeuverPrimary } from '../api/types';
import type { Tone } from './tones';

export const MANEUVER_ICON: Record<ManeuverPrimary, LucideIcon> = {
  STOP: OctagonX,
  BRAKE: ArrowDown,
  SLOW_DOWN: ArrowDown,
  ACCELERATE: ArrowUp,
  CRUISE: Equal,
  TURN_LEFT: ArrowUpLeft,
  TURN_RIGHT: ArrowUpRight,
  HOLD: Pause,
};

/** Severity colouring: stop danger, braking/slowing warning, accelerate accent, turns cyan, cruise/hold neutral. */
export const MANEUVER_TONE: Record<ManeuverPrimary, Tone> = {
  STOP: 'danger',
  BRAKE: 'danger',
  SLOW_DOWN: 'warning',
  ACCELERATE: 'info',
  CRUISE: 'neutral',
  TURN_LEFT: 'cyan',
  TURN_RIGHT: 'cyan',
  HOLD: 'neutral',
};

/** Large-indicator box classes per tone (border, background, foreground). */
export const TONE_BOX: Record<Tone, string> = {
  neutral: 'border-line bg-canvas text-ink',
  info: 'border-accent/30 bg-accent/8 text-accent-strong',
  success: 'border-success/30 bg-success/8 text-success',
  warning: 'border-warning/30 bg-warning/8 text-warning',
  danger: 'border-danger/30 bg-danger/8 text-danger',
  cyan: 'border-cyan/30 bg-cyan/8 text-cyan-ink',
};

export const COMPARATOR_SYMBOL: Record<DecisionBasisComparator, string> = {
  '>=': '≥',
  '>': '>',
  '<=': '≤',
  '<': '<',
  '==': '=',
  '!=': '≠',
};

export const BASIS_SOURCE_TONE: Record<DecisionBasisSource, Tone> = {
  learned_model: 'info',
  synthetic_rgb_cv: 'cyan',
  ego_telemetry: 'neutral',
  route_graph: 'neutral',
  operator: 'warning',
  safety_supervisor: 'danger',
  controller: 'neutral',
};

/** Uploaded video: what the analytical decision would recommend. No vehicle is controlled. */
export type VideoRecommendation = 'CONTINUE' | 'SLOW_DOWN' | 'STOP';

export function recommendationFor(state: DecisionState): VideoRecommendation | null {
  switch (state) {
    case 'observe':
    case 'follow_route':
      return 'CONTINUE';
    case 'slow':
      return 'SLOW_DOWN';
    case 'controlled_stop':
    case 'safety_stop':
      return 'STOP';
    default:
      return null;
  }
}

export const RECOMMENDATION_ICON: Record<VideoRecommendation, LucideIcon> = {
  CONTINUE: ArrowUp,
  SLOW_DOWN: ArrowDown,
  STOP: OctagonX,
};
export const RECOMMENDATION_TONE: Record<VideoRecommendation, Tone> = {
  CONTINUE: 'success',
  SLOW_DOWN: 'warning',
  STOP: 'danger',
};

/** One maneuver change (from SSE decision events, live polling or the run report timeline). */
export interface ManeuverChange {
  primary: ManeuverPrimary;
  longitudinal: string;
  lateral: string;
  since_sim_time_s: number;
  frame_id: number;
  reason_codes: string[];
}

/**
 * Appends a change when the primary maneuver differs from the last kept entry and is not older than it (SSE events
 * and REST refreshes may arrive out of order). Keeps the newest `max` entries.
 */
export function appendChange(items: readonly ManeuverChange[], next: ManeuverChange, max = 30): readonly ManeuverChange[] {
  const last = items[items.length - 1];
  if (last && (last.primary === next.primary || next.since_sim_time_s + 1e-9 < last.since_sim_time_s)) return items;
  return [...items, next].slice(-max);
}
