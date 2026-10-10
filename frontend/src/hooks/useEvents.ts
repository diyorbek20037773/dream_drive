/**
 * One shared Server-Sent Events connection to `/api/v1/events` (owner-filtered by the session cookie).
 *
 * - Typed handlers per event type (payloads from the generated OpenAPI types).
 * - When the browser closes the stream (HTTP error, gateway 502, expired session) the hub refreshes the session and
 *   reconnects with exponential backoff + jitter, resuming after the last stream id (`?last_event_id=`).
 * - Every successful re-open after a disconnect emits a "resync" signal: pages re-fetch REST state because missed
 *   events (and camera frames) are not replayed.
 */
import { useEffect, useRef, useSyncExternalStore } from 'react';

import { API_BASE, ensureSession } from '../api/client';
import type { EventEnvelope, EventPayloads, EventType } from '../api/types';

export type EventStreamStatus = 'idle' | 'connecting' | 'online' | 'reconnecting' | 'offline';

export type EventHandlers = {
  [K in EventType]?: (payload: EventPayloads[K], envelope: EventEnvelope) => void;
};

const EVENT_TYPES: readonly EventType[] = [
  'job_state',
  'job_progress',
  'run_state',
  'frame_ready',
  'decision',
  'capability_changed',
  'error',
];
const STREAM_ID = /^\d{1,20}-\d{1,20}$/;
const BASE_DELAY_MS = 1000;
const MAX_DELAY_MS = 30_000;

function isEnvelope(value: unknown): value is EventEnvelope {
  if (typeof value !== 'object' || value === null) return false;
  const candidate = value as Partial<EventEnvelope>;
  return (
    typeof candidate.type === 'string' &&
    EVENT_TYPES.includes(candidate.type) &&
    typeof candidate.payload === 'object' &&
    candidate.payload !== null
  );
}

class EventHub {
  private source: EventSource | null = null;
  private status: EventStreamStatus = 'idle';
  private lastEventId: string | null = null;
  private failures = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private wanted = false;
  private everOpened = false;
  private readonly handlers = new Set<{ current: EventHandlers }>();
  private readonly statusListeners = new Set<() => void>();
  private readonly resyncListeners = new Set<() => void>();

  start(): void {
    if (this.wanted) return;
    this.wanted = true;
    this.connect();
  }

  stop(): void {
    this.wanted = false;
    this.clearTimer();
    this.source?.close();
    this.source = null;
    this.setStatus('idle');
  }

  /** Forces an immediate reconnect attempt (e.g. the user pressed "retry"). */
  reconnectNow(): void {
    if (!this.wanted) return;
    this.source?.close();
    this.source = null;
    this.failures = 0;
    this.connect();
  }

  getStatus = (): EventStreamStatus => this.status;

  subscribeStatus = (listener: () => void): (() => void) => {
    this.statusListeners.add(listener);
    return () => this.statusListeners.delete(listener);
  };

  subscribeResync(listener: () => void): () => void {
    this.resyncListeners.add(listener);
    return () => this.resyncListeners.delete(listener);
  }

  subscribe(ref: { current: EventHandlers }): () => void {
    this.handlers.add(ref);
    return () => this.handlers.delete(ref);
  }

  private setStatus(status: EventStreamStatus): void {
    if (this.status === status) return;
    this.status = status;
    for (const listener of this.statusListeners) listener();
  }

  private clearTimer(): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
  }

  private connect(): void {
    this.clearTimer();
    if (!this.wanted) return;
    this.setStatus(this.everOpened || this.failures > 0 ? 'reconnecting' : 'connecting');
    const url = this.lastEventId
      ? `${API_BASE}/events?last_event_id=${encodeURIComponent(this.lastEventId)}`
      : `${API_BASE}/events`;
    const source = new EventSource(url);
    this.source = source;

    source.onopen = () => {
      const isReconnect = this.everOpened || this.failures > 0;
      this.everOpened = true;
      this.failures = 0;
      this.setStatus('online');
      if (isReconnect) for (const listener of this.resyncListeners) listener();
    };
    source.onerror = () => {
      if (this.source !== source) return;
      if (source.readyState === EventSource.CLOSED) {
        // The browser gave up (HTTP error status, wrong content type, 401): we own the retry.
        this.source = null;
        this.scheduleReconnect();
      } else {
        // Network blip: the browser retries by itself and sends Last-Event-ID automatically.
        this.setStatus('reconnecting');
      }
    };
    for (const type of EVENT_TYPES) {
      source.addEventListener(type, (event) => this.dispatch(event as MessageEvent<string>));
    }
    // Messages without an `event:` field arrive as "message"; the envelope still carries its type.
    source.onmessage = (event: MessageEvent<string>) => this.dispatch(event);
  }

  private scheduleReconnect(): void {
    if (!this.wanted) return;
    this.failures += 1;
    this.setStatus(this.failures >= 3 ? 'offline' : 'reconnecting');
    const exp = Math.min(MAX_DELAY_MS, BASE_DELAY_MS * 2 ** Math.min(this.failures - 1, 6));
    const delay = exp / 2 + Math.random() * (exp / 2);
    this.timer = setTimeout(() => {
      // A closed stream is frequently an expired/missing session (401): refresh it, then reconnect.
      ensureSession(true)
        .then(() => this.connect())
        .catch(() => this.scheduleReconnect());
    }, delay);
  }

  private dispatch(event: MessageEvent<string>): void {
    let envelope: unknown;
    try {
      envelope = JSON.parse(event.data) as unknown;
    } catch {
      return;
    }
    if (!isEnvelope(envelope)) return;
    if (event.lastEventId && STREAM_ID.test(event.lastEventId)) this.lastEventId = event.lastEventId;
    for (const ref of this.handlers) {
      const handler = ref.current[envelope.type] as
        | ((payload: EventPayloads[EventType], env: EventEnvelope) => void)
        | undefined;
      if (!handler) continue;
      try {
        handler(envelope.payload, envelope);
      } catch (error) {
        console.error('event handler failed', error);
      }
    }
  }
}

export const eventHub = new EventHub();

/** Current connection status of the shared event stream. */
export function useEventStatus(): EventStreamStatus {
  return useSyncExternalStore(eventHub.subscribeStatus, eventHub.getStatus, eventHub.getStatus);
}

/** Subscribes typed handlers to the shared stream; the latest handlers are used without resubscribing. */
export function useEvents(handlers: EventHandlers): void {
  const ref = useRef<EventHandlers>(handlers);
  useEffect(() => {
    ref.current = handlers;
  });
  useEffect(() => eventHub.subscribe(ref), []);
}

/** Runs `callback` after the stream re-opens following a disconnect (REST state must be re-fetched). */
export function useResync(callback: () => void): void {
  const ref = useRef(callback);
  useEffect(() => {
    ref.current = callback;
  });
  useEffect(() => eventHub.subscribeResync(() => ref.current()), []);
}
