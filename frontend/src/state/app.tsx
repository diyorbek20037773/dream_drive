/**
 * Application-wide state: the cookie session (created at startup), the capability document and the shared event
 * stream lifecycle. Pages read limits/capabilities from here instead of re-fetching them.
 */
import type { ReactNode } from 'react';
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';

import type { ApiError } from '../api/client';
import { api, ensureSession, getSession, onSessionChange, toApiError } from '../api/client';
import type { Capabilities, Limits, SessionInfo } from '../api/types';
import { eventHub, useEvents, useResync } from '../hooks/useEvents';
import { useInterval } from '../hooks/useTimers';

const CAPABILITIES_REFRESH_MS = 30_000;
const SESSION_RETRY_MS = [3_000, 6_000, 12_000, 30_000];

interface AppState {
  session: SessionInfo | null;
  sessionError: ApiError | null;
  /** Epoch milliseconds of the next automatic session retry (null when none is scheduled). */
  sessionRetryAt: number | null;
  retrySession: () => void;
  capabilities: Capabilities | null;
  capabilitiesError: ApiError | null;
  /** True until the first capabilities response (success or error). */
  capabilitiesLoading: boolean;
  refreshCapabilities: () => void;
  limits: Limits | null;
}

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<SessionInfo | null>(getSession);
  const [sessionError, setSessionError] = useState<ApiError | null>(null);
  const [sessionRetryAt, setSessionRetryAt] = useState<number | null>(null);
  const [sessionAttempt, setSessionAttempt] = useState(0);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [capabilitiesError, setCapabilitiesError] = useState<ApiError | null>(null);
  const failures = useRef(0);
  const capsController = useRef<AbortController | null>(null);

  useEffect(() => onSessionChange(setSession), []);

  // Background refresh: state only changes when the response arrives (no loading flicker).
  const refreshCapabilities = useCallback(() => {
    capsController.current?.abort();
    const controller = new AbortController();
    capsController.current = controller;
    api.capabilities(controller.signal).then(
      (caps) => {
        if (controller.signal.aborted) return;
        setCapabilities(caps);
        setCapabilitiesError(null);
      },
      (error: unknown) => {
        if (!controller.signal.aborted) setCapabilitiesError(toApiError(error));
      },
    );
  }, []);

  useEffect(() => {
    refreshCapabilities();
    return () => capsController.current?.abort();
  }, [refreshCapabilities]);

  // Session bootstrap with bounded backoff; the event stream starts once a session exists.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    ensureSession().then(
      () => {
        if (cancelled) return;
        failures.current = 0;
        setSessionError(null);
        setSessionRetryAt(null);
        eventHub.start();
      },
      (error: unknown) => {
        if (cancelled) return;
        const delay = SESSION_RETRY_MS[Math.min(failures.current, SESSION_RETRY_MS.length - 1)] ?? 30_000;
        failures.current += 1;
        setSessionError(toApiError(error));
        setSessionRetryAt(Date.now() + delay);
        timer = setTimeout(() => {
          setSessionAttempt((value) => value + 1);
          refreshCapabilities();
        }, delay);
      },
    );
    return () => {
      cancelled = true;
      if (timer !== null) clearTimeout(timer);
    };
  }, [sessionAttempt, refreshCapabilities]);

  useEffect(() => () => eventHub.stop(), []);

  useInterval(refreshCapabilities, CAPABILITIES_REFRESH_MS);
  useResync(refreshCapabilities);
  useEvents({ capability_changed: refreshCapabilities });

  const retrySession = useCallback(() => {
    failures.current = 0;
    setSessionRetryAt(null);
    setSessionAttempt((value) => value + 1);
    refreshCapabilities();
  }, [refreshCapabilities]);

  const capabilitiesLoading = capabilities === null && capabilitiesError === null;
  const value = useMemo<AppState>(
    () => ({
      session,
      sessionError,
      sessionRetryAt,
      retrySession,
      capabilities,
      capabilitiesError,
      capabilitiesLoading,
      refreshCapabilities,
      limits: capabilities?.limits ?? session?.limits ?? null,
    }),
    [session, sessionError, sessionRetryAt, retrySession, capabilities, capabilitiesError, capabilitiesLoading, refreshCapabilities],
  );
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

// eslint-disable-next-line react-refresh/only-export-components
export function useApp(): AppState {
  const value = useContext(AppContext);
  if (!value) throw new Error('useApp must be used inside AppProvider');
  return value;
}
