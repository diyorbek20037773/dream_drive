import { useCallback, useEffect, useRef, useState } from 'react';

import type { ApiError } from '../api/client';
import { toApiError } from '../api/client';

export interface Resource<T> {
  data: T | undefined;
  error: ApiError | null;
  /** True while a request for the current key is in flight (also during a background reload). */
  loading: boolean;
  /** True only before the first response for the current key (no data or error yet). */
  initialLoading: boolean;
  reload: () => void;
  /** Replaces the cached data locally (e.g. after an SSE event or a mutation response). */
  setData: (updater: T | ((previous: T | undefined) => T | undefined)) => void;
}

interface Snapshot<T> {
  key: string | null;
  token: number;
  data: T | undefined;
  error: ApiError | null;
}

/**
 * Fetches `fetcher` whenever `key` changes (null disables the fetch) or `reload()` is called. Aborts stale requests.
 * Loading state is derived from which request completed last, so no state is set synchronously inside the effect.
 */
export function useResource<T>(key: string | null, fetcher: (signal: AbortSignal) => Promise<T>): Resource<T> {
  const fetcherRef = useRef(fetcher);
  useEffect(() => {
    fetcherRef.current = fetcher;
  });
  const [token, setToken] = useState(0);
  const [snapshot, setSnapshot] = useState<Snapshot<T>>({ key: null, token: -1, data: undefined, error: null });

  useEffect(() => {
    if (key === null) return;
    const controller = new AbortController();
    fetcherRef.current(controller.signal).then(
      (data) => {
        if (!controller.signal.aborted) setSnapshot({ key, token, data, error: null });
      },
      (error: unknown) => {
        if (controller.signal.aborted) return;
        // Keep the last good data for the same key; the error is shown next to it.
        setSnapshot((previous) => ({
          key,
          token,
          data: previous.key === key ? previous.data : undefined,
          error: toApiError(error),
        }));
      },
    );
    return () => controller.abort();
  }, [key, token]);

  const reload = useCallback(() => setToken((value) => value + 1), []);
  const setData = useCallback((updater: T | ((previous: T | undefined) => T | undefined)) => {
    setSnapshot((previous) => ({
      ...previous,
      data:
        typeof updater === 'function'
          ? (updater as (value: T | undefined) => T | undefined)(previous.data)
          : updater,
    }));
  }, []);

  const sameKey = snapshot.key === key;
  const settled = sameKey && snapshot.token === token;
  return {
    data: sameKey ? snapshot.data : undefined,
    error: sameKey ? snapshot.error : null,
    loading: key !== null && !settled,
    initialLoading: key !== null && !sameKey,
    reload,
    setData,
  };
}
