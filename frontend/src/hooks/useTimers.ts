import { useEffect, useRef, useState } from 'react';

/** Calls `callback` every `delayMs`; pass null to pause. The latest callback is always used. */
export function useInterval(callback: () => void, delayMs: number | null): void {
  const ref = useRef(callback);
  useEffect(() => {
    ref.current = callback;
  });
  useEffect(() => {
    if (delayMs === null) return;
    const handle = setInterval(() => ref.current(), delayMs);
    return () => clearInterval(handle);
  }, [delayMs]);
}

/** Wall-clock milliseconds, refreshed every `intervalMs` while `active` (for elapsed-time displays). */
export function useNow(intervalMs = 1000, active = true): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const handle = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(handle);
  }, [intervalMs, active]);
  return now;
}

/** True when the user asked the OS to reduce motion. */
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(() => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false);
  useEffect(() => {
    const query = window.matchMedia?.('(prefers-reduced-motion: reduce)');
    if (!query) return;
    const onChange = () => setReduced(query.matches);
    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  }, []);
  return reduced;
}
