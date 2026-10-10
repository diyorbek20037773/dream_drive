/**
 * Mapping between video playback time and analyzed timeline frames.
 *
 * - Original video: frame k is shown at its real presentation timestamp `frames[k].pts_s` (timeline.json PTS).
 * - Annotated MP4: encoded at constant `analysis_fps`, output frame k corresponds to timeline frame k, so it occupies
 *   [k / fps, (k + 1) / fps). We seek to the middle of that interval to avoid landing on a boundary.
 */
import type { DecisionState, TimelineFile, TimelineFrame } from '../api/types';

export type VideoView = 'annotated' | 'original';

export interface TimelineEvent {
  /** Index of the event in the event list. */
  id: number;
  startIndex: number;
  endIndex: number;
  startPts: number;
  endPts: number;
  state: DecisionState;
  reasons: string[];
}

/** Last index whose pts is <= t (frames are sorted by pts); 0 when t precedes the first frame. */
export function indexAtPts(frames: readonly TimelineFrame[], t: number): number {
  let lo = 0;
  let hi = frames.length - 1;
  let found = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const frame = frames[mid];
    if (frame && frame.pts_s <= t + 1e-6) {
      found = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return found;
}

export function indexForTime(timeline: TimelineFile, t: number, view: VideoView): number {
  const count = timeline.frames.length;
  if (count === 0) return 0;
  if (view === 'annotated' && timeline.analysis_fps > 0) {
    return Math.min(count - 1, Math.max(0, Math.floor(t * timeline.analysis_fps + 1e-6)));
  }
  return indexAtPts(timeline.frames, t);
}

export function timeForIndex(timeline: TimelineFile, index: number, view: VideoView): number {
  const frame = timeline.frames[index];
  if (!frame) return 0;
  if (view === 'annotated' && timeline.analysis_fps > 0) return (index + 0.5) / timeline.analysis_fps;
  // A millisecond past the PTS so the decoder shows this frame rather than the previous one.
  return frame.pts_s + 0.001;
}

/** Groups consecutive frames with the same decision state and reason set into events. */
export function buildEvents(frames: readonly TimelineFrame[]): TimelineEvent[] {
  const events: TimelineEvent[] = [];
  let current: TimelineEvent | null = null;
  let currentKey = '';
  frames.forEach((frame, index) => {
    const reasons = [...frame.decision.reason_codes].sort();
    const key = `${frame.decision.state}|${reasons.join(',')}`;
    if (current && key === currentKey) {
      current.endIndex = index;
      current.endPts = frame.pts_s;
      return;
    }
    current = {
      id: events.length,
      startIndex: index,
      endIndex: index,
      startPts: frame.pts_s,
      endPts: frame.pts_s,
      state: frame.decision.state,
      reasons: frame.decision.reason_codes,
    };
    currentKey = key;
    events.push(current);
  });
  return events;
}

export function eventForIndex(events: readonly TimelineEvent[], index: number): number {
  let lo = 0;
  let hi = events.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const event = events[mid];
    if (!event) break;
    if (index < event.startIndex) hi = mid - 1;
    else if (index > event.endIndex) lo = mid + 1;
    else return mid;
  }
  return -1;
}

/** First analyzed frame at or after `t` seconds (used to jump to a track's first appearance). */
export function firstIndexAtOrAfter(frames: readonly TimelineFrame[], t: number): number {
  const index = indexAtPts(frames, t);
  const frame = frames[index];
  if (frame && frame.pts_s + 1e-6 < t && index + 1 < frames.length) return index + 1;
  return index;
}
