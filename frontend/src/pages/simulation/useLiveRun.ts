import { useCallback, useEffect, useRef, useState } from 'react';

import type { ApiError } from '../../api/client';
import { api, toApiError } from '../../api/client';
import type { Run, RunLive } from '../../api/types';

const LIVE_HISTORY = 40;

interface LiveState {
  runId: string | null;
  run: Run | null;
  lives: RunLive[];
  error: ApiError | null;
}

/**
 * The selected run, refreshed on demand (frame_ready / run_state events, polling). Concurrent refreshes are coalesced
 * into at most one in-flight request plus one follow-up. Recent `live` snapshots are kept by frame id so the camera
 * overlay can be matched to the exact image it was computed from.
 */
export function useLiveRun(runId: string | null) {
  const [state, setState] = useState<LiveState>({ runId: null, run: null, lives: [], error: null });
  const inFlight = useRef(false);
  const again = useRef(false);
  const currentId = useRef(runId);

  const accept = useCallback((run: Run) => {
    if (run.id !== currentId.current) return;
    setState((previous) => {
      const base = previous.runId === run.id ? previous.lives : [];
      // A newer snapshot of the same frame (e.g. paused: no new frame, decision/maneuver change) replaces the old one.
      const lives = run.live ? [...base.filter((item) => item.frame_id !== run.live?.frame_id), run.live].slice(-LIVE_HISTORY) : base;
      return { runId: run.id, run, lives, error: null };
    });
  }, []);

  const refresh = useCallback(() => {
    const fetchOnce = () => {
      const id = currentId.current;
      if (!id) return;
      inFlight.current = true;
      api
        .getRun(id)
        .then(accept, (error: unknown) => {
          if (id !== currentId.current) return;
          setState((previous) => ({ ...previous, runId: id, error: toApiError(error) }));
        })
        .finally(() => {
          inFlight.current = false;
          if (again.current) {
            again.current = false;
            fetchOnce();
          }
        });
    };
    if (inFlight.current) {
      again.current = true;
      return;
    }
    fetchOnce();
  }, [accept]);

  useEffect(() => {
    currentId.current = runId;
    again.current = false;
    if (runId) refresh();
  }, [runId, refresh]);

  const same = state.runId === runId;
  return {
    run: same ? state.run : null,
    lives: same ? state.lives : [],
    error: same ? state.error : null,
    refresh,
    /** Applies a Run returned by a mutation (control/create) immediately. */
    accept,
  };
}
