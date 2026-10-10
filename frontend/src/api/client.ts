/**
 * Typed same-origin API client.
 *
 * - Only relative URLs (`/api/v1`, `/health`): the gateway routes them; no API host is compiled into the bundle.
 * - `credentials: 'same-origin'`: the HttpOnly `dd_session` cookie authenticates every call.
 * - The CSRF token from `POST /api/v1/session` is kept in memory only (never localStorage) and sent as
 *   `X-CSRF-Token` on every POST/DELETE.
 * - Non-2xx responses are parsed from the `ErrorEnvelope` into a typed `ApiError`; a `401 session_required` is retried
 *   once after re-creating the session.
 */
import type {
  AnyErrorCode,
  Capabilities,
  DeleteResult,
  ErrorEnvelope,
  Evaluation,
  EvaluationList,
  HealthStatus,
  Job,
  JobList,
  JobResults,
  ObstacleInject,
  ObstacleInjectResult,
  Run,
  RunAction,
  RunCreate,
  RunList,
  RunReport,
  SessionInfo,
  SimulatorList,
  TimelineFile,
} from './types';

export const API_BASE = '/api/v1';

export class ApiError extends Error {
  readonly status: number;
  readonly code: AnyErrorCode;
  readonly requestId: string | null;
  readonly details: Record<string, unknown>;
  readonly retryAfterS: number | null;

  constructor(init: {
    status: number;
    code: AnyErrorCode;
    message: string;
    requestId?: string | null;
    details?: Record<string, unknown>;
    retryAfterS?: number | null;
  }) {
    super(init.message);
    this.name = 'ApiError';
    this.status = init.status;
    this.code = init.code;
    this.requestId = init.requestId ?? null;
    this.details = init.details ?? {};
    this.retryAfterS = init.retryAfterS ?? null;
  }

  /** The service could not be reached at all (as opposed to a feature reporting itself unavailable). */
  get isOffline(): boolean {
    return this.code === 'network_error' || this.code === 'service_offline';
  }
}

export function isApiError(value: unknown): value is ApiError {
  return value instanceof ApiError;
}

/** Converts anything thrown into an ApiError so UI code handles one error type. */
export function toApiError(value: unknown): ApiError {
  if (value instanceof ApiError) return value;
  if (value instanceof DOMException && value.name === 'AbortError') {
    return new ApiError({ status: 0, code: 'aborted', message: 'Request aborted' });
  }
  const message = value instanceof Error ? value.message : String(value);
  return new ApiError({ status: 0, code: 'network_error', message });
}

// ---------------------------------------------------------------------------------------------------------------
// Session state (memory only)

let session: SessionInfo | null = null;
let sessionInFlight: Promise<SessionInfo> | null = null;
const sessionListeners = new Set<(info: SessionInfo | null) => void>();

export function getSession(): SessionInfo | null {
  return session;
}

export function onSessionChange(listener: (info: SessionInfo | null) => void): () => void {
  sessionListeners.add(listener);
  return () => sessionListeners.delete(listener);
}

function setSession(info: SessionInfo | null): void {
  session = info;
  for (const listener of sessionListeners) listener(info);
}

function sessionStillValid(info: SessionInfo): boolean {
  const expires = Date.parse(info.expires_utc);
  // Refresh one minute early so a request never races the cookie expiry.
  return Number.isFinite(expires) && expires - Date.now() > 60_000;
}

/** Creates (or refreshes) the cookie session. Concurrent callers share one request. */
export function ensureSession(force = false): Promise<SessionInfo> {
  if (sessionInFlight) return sessionInFlight;
  if (!force && session && sessionStillValid(session)) return Promise.resolve(session);
  sessionInFlight = rawRequest<SessionInfo>(`${API_BASE}/session`, { method: 'POST', body: {} })
    .then((info) => {
      setSession(info);
      return info;
    })
    .finally(() => {
      sessionInFlight = null;
    });
  return sessionInFlight;
}

// ---------------------------------------------------------------------------------------------------------------
// Low-level request helpers

type Method = 'GET' | 'POST' | 'DELETE';

interface RequestOptions {
  method?: Method;
  body?: unknown;
  signal?: AbortSignal;
  query?: Record<string, string | number | null | undefined>;
  /** Defaults to true: establish the session before the call and retry once on `401 session_required`. */
  session?: boolean;
  /** Non-2xx statuses whose JSON body is a valid result (e.g. 503 from /health/ready). */
  acceptStatuses?: number[];
}

function withQuery(url: string, query: RequestOptions['query']): string {
  if (!query) return url;
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== null && value !== undefined && value !== '') params.set(key, String(value));
  }
  const text = params.toString();
  return text ? `${url}?${text}` : url;
}

function retryAfterSeconds(response: Response): number | null {
  const header = response.headers.get('Retry-After');
  if (!header) return null;
  const seconds = Number(header);
  if (Number.isFinite(seconds)) return Math.max(0, Math.round(seconds));
  const date = Date.parse(header);
  return Number.isFinite(date) ? Math.max(0, Math.round((date - Date.now()) / 1000)) : null;
}

function isJson(response: Response): boolean {
  return (response.headers.get('Content-Type') ?? '').toLowerCase().includes('json');
}

function isEnvelope(value: unknown): value is ErrorEnvelope {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as { code?: unknown }).code === 'string' &&
    typeof (value as { message?: unknown }).message === 'string'
  );
}

function isHealthStatus(value: unknown): value is HealthStatus {
  if (typeof value !== 'object' || value === null) return false;
  const candidate = value as Partial<HealthStatus>;
  return (
    (candidate.status === 'ok' || candidate.status === 'unavailable') &&
    typeof candidate.checks === 'object' &&
    candidate.checks !== null
  );
}

/** Builds an ApiError from a non-2xx response, preferring the server's ErrorEnvelope. */
export async function errorFromResponse(response: Response): Promise<ApiError> {
  const requestId = response.headers.get('X-Request-ID');
  const retryAfterS = retryAfterSeconds(response);
  if (isJson(response)) {
    try {
      const body: unknown = await response.json();
      if (isEnvelope(body)) {
        return new ApiError({
          status: response.status,
          code: body.code,
          message: body.message,
          requestId: body.request_id || requestId,
          details: body.details ?? {},
          retryAfterS,
        });
      }
    } catch {
      // fall through to the generic mapping below
    }
  }
  return errorFromStatus(response.status, requestId, retryAfterS);
}

function errorFromStatus(status: number, requestId: string | null, retryAfterS: number | null): ApiError {
  // A gateway without a reachable API answers 502/503/504 without our JSON envelope.
  if (status === 502 || status === 503 || status === 504) {
    return new ApiError({ status, code: 'service_offline', message: `HTTP ${status}`, requestId, retryAfterS });
  }
  return new ApiError({ status, code: 'bad_response', message: `HTTP ${status}`, requestId, retryAfterS });
}

async function send(url: string, options: RequestOptions): Promise<Response> {
  const method = options.method ?? 'GET';
  const headers: Record<string, string> = { Accept: 'application/json' };
  const csrf = session?.csrf_token;
  if (method !== 'GET' && csrf) headers['X-CSRF-Token'] = csrf;
  let body: string | undefined;
  if (options.body !== undefined) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(options.body);
  }
  try {
    return await fetch(withQuery(url, options.query), {
      method,
      headers,
      body,
      credentials: 'same-origin',
      cache: 'no-store',
      signal: options.signal,
    });
  } catch (error) {
    throw toApiError(error);
  }
}

async function parseSuccess<T>(response: Response): Promise<T> {
  if (response.status === 204) return undefined as T;
  if (!isJson(response)) {
    throw new ApiError({
      status: response.status,
      code: 'bad_response',
      message: `Expected JSON, got ${response.headers.get('Content-Type') ?? 'no content type'}`,
      requestId: response.headers.get('X-Request-ID'),
    });
  }
  try {
    return (await response.json()) as T;
  } catch (error) {
    throw new ApiError({ status: response.status, code: 'bad_response', message: String(error) });
  }
}

/** One request without session handling. */
async function rawRequest<T>(url: string, options: RequestOptions = {}): Promise<T> {
  const response = await send(url, options);
  if (response.ok || options.acceptStatuses?.includes(response.status)) return parseSuccess<T>(response);
  throw await errorFromResponse(response);
}

/** Session-aware request: ensures the session first and retries once on `401 session_required`. */
export async function request<T>(url: string, options: RequestOptions = {}): Promise<T> {
  if (options.session === false) return rawRequest<T>(url, options);
  await ensureSession();
  try {
    return await rawRequest<T>(url, options);
  } catch (error) {
    if (isApiError(error) && error.status === 401 && error.code === 'session_required') {
      await ensureSession(true);
      return rawRequest<T>(url, options);
    }
    throw error;
  }
}

/** Accepts only same-origin relative URLs returned by the API (defence against absolute/protocol-relative links). */
export function safeRelativeUrl(url: string | null | undefined): string | null {
  if (!url || !url.startsWith('/') || url.startsWith('//')) return null;
  return url;
}

const id = (value: string) => encodeURIComponent(value);

// ---------------------------------------------------------------------------------------------------------------
// Endpoints

export const api = {
  createSession: (adminPassword?: string) =>
    rawRequest<SessionInfo>(`${API_BASE}/session`, {
      method: 'POST',
      body: adminPassword ? { admin_password: adminPassword } : {},
    }).then((info) => {
      setSession(info);
      return info;
    }),
  capabilities: (signal?: AbortSignal) =>
    request<Capabilities>(`${API_BASE}/capabilities`, { session: false, signal }),
  healthReady: async (signal?: AbortSignal) => {
    // 503 carries the same HealthStatus shape from the API; anything else (e.g. a gateway/web error envelope) is an error.
    const body = await request<unknown>('/health/ready', { session: false, signal, acceptStatuses: [503] });
    if (isHealthStatus(body)) return body;
    if (isEnvelope(body)) {
      throw new ApiError({ status: 503, code: body.code, message: body.message, requestId: body.request_id || null });
    }
    throw new ApiError({ status: 0, code: 'bad_response', message: 'Unexpected /health/ready body' });
  },

  listJobs: (limit = 20, cursor?: string | null, signal?: AbortSignal) =>
    request<JobList>(`${API_BASE}/jobs`, { query: { limit, cursor }, signal }),
  getJob: (jobId: string, signal?: AbortSignal) => request<Job>(`${API_BASE}/jobs/${id(jobId)}`, { signal }),
  cancelJob: (jobId: string) => request<Job>(`${API_BASE}/jobs/${id(jobId)}/cancel`, { method: 'POST' }),
  deleteJob: (jobId: string) => request<DeleteResult>(`${API_BASE}/jobs/${id(jobId)}`, { method: 'DELETE' }),
  getJobResults: (jobId: string, signal?: AbortSignal) =>
    request<JobResults>(`${API_BASE}/jobs/${id(jobId)}/results`, { signal }),
  getTimeline: async (url: string, signal?: AbortSignal) => {
    const safe = safeRelativeUrl(url);
    if (!safe) throw new ApiError({ status: 0, code: 'bad_response', message: 'Timeline URL is not same-origin' });
    return request<TimelineFile>(safe, { signal });
  },

  listSimulators: (signal?: AbortSignal) => request<SimulatorList>(`${API_BASE}/simulators`, { signal }),
  listRuns: (limit = 20, signal?: AbortSignal) => request<RunList>(`${API_BASE}/runs`, { query: { limit }, signal }),
  getRun: (runId: string, signal?: AbortSignal) => request<Run>(`${API_BASE}/runs/${id(runId)}`, { signal }),
  createRun: (body: RunCreate) => request<Run>(`${API_BASE}/runs`, { method: 'POST', body }),
  controlRun: (runId: string, action: RunAction) =>
    request<Run>(`${API_BASE}/runs/${id(runId)}/control`, { method: 'POST', body: { action } }),
  injectObstacle: (runId: string, body: ObstacleInject) =>
    request<ObstacleInjectResult>(`${API_BASE}/runs/${id(runId)}/obstacles`, { method: 'POST', body }),
  getRunReport: (runId: string, signal?: AbortSignal) =>
    request<RunReport>(`${API_BASE}/runs/${id(runId)}/report`, { signal }),

  listEvaluations: (signal?: AbortSignal) => request<EvaluationList>(`${API_BASE}/evaluations`, { signal }),
  getEvaluation: (evalId: string, signal?: AbortSignal) =>
    request<Evaluation>(`${API_BASE}/evaluations/${id(evalId)}`, { signal }),
  createEvaluation: (seeds: number[]) =>
    request<Evaluation>(`${API_BASE}/evaluations`, {
      method: 'POST',
      body: { kind: 'synthetic_scenario_suite', seeds },
    }),
};

// ---------------------------------------------------------------------------------------------------------------
// Camera frame (binary) for simulation runs

export interface RunFrame {
  blob: Blob;
  frameId: number | null;
  simTimeS: number | null;
  perceptionSource: string | null;
}

function numberHeader(response: Response, name: string): number | null {
  const raw = response.headers.get(name);
  if (raw === null || raw.trim() === '') return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

/**
 * Latest actual camera JPEG. `frameHint` only busts caches (the endpoint is `no-store` anyway); the authoritative
 * frame identity is the `X-Frame-Id` header of the returned image. Returns null on 204 (no frame captured yet).
 */
export async function fetchRunFrame(runId: string, frameHint: number, signal?: AbortSignal): Promise<RunFrame | null> {
  await ensureSession();
  const url = `${API_BASE}/runs/${id(runId)}/frame?f=${frameHint}`;
  const doFetch = async () => {
    try {
      return await fetch(url, { credentials: 'same-origin', cache: 'no-store', signal });
    } catch (error) {
      throw toApiError(error);
    }
  };
  let response = await doFetch();
  if (response.status === 401) {
    const error = await errorFromResponse(response);
    if (error.code !== 'session_required') throw error;
    await ensureSession(true);
    response = await doFetch();
  }
  if (response.status === 204) return null;
  if (!response.ok) throw await errorFromResponse(response);
  return {
    blob: await response.blob(),
    frameId: numberHeader(response, 'X-Frame-Id'),
    simTimeS: numberHeader(response, 'X-Sim-Time'),
    perceptionSource: response.headers.get('X-Perception-Source'),
  };
}

// ---------------------------------------------------------------------------------------------------------------
// Video upload with real upload progress (XMLHttpRequest exposes upload progress; fetch does not)

export interface UploadHandle {
  promise: Promise<Job>;
  abort: () => void;
}

export function uploadVideo(file: File, onProgress: (loaded: number, total: number) => void): UploadHandle {
  let xhr: XMLHttpRequest | null = null;
  let aborted = false;

  const attempt = (): Promise<Job> =>
    new Promise<Job>((resolve, reject) => {
      const request = new XMLHttpRequest();
      xhr = request;
      request.open('POST', `${API_BASE}/videos`);
      request.setRequestHeader('Accept', 'application/json');
      const csrf = session?.csrf_token;
      if (csrf) request.setRequestHeader('X-CSRF-Token', csrf);
      request.upload.onprogress = (event) => {
        onProgress(event.loaded, event.lengthComputable ? event.total : file.size);
      };
      request.onerror = () =>
        reject(new ApiError({ status: 0, code: 'network_error', message: 'Upload connection failed' }));
      request.onabort = () => reject(new ApiError({ status: 0, code: 'aborted', message: 'Upload cancelled' }));
      request.onload = () => {
        const contentType = request.getResponseHeader('Content-Type') ?? '';
        const requestId = request.getResponseHeader('X-Request-ID');
        let body: unknown = null;
        if (contentType.includes('json')) {
          try {
            body = JSON.parse(request.responseText) as unknown;
          } catch {
            body = null;
          }
        }
        if (request.status >= 200 && request.status < 300 && body) {
          resolve(body as Job);
          return;
        }
        const retryHeader = request.getResponseHeader('Retry-After');
        const retryAfterS = retryHeader && Number.isFinite(Number(retryHeader)) ? Number(retryHeader) : null;
        if (isEnvelope(body)) {
          reject(
            new ApiError({
              status: request.status,
              code: body.code,
              message: body.message,
              requestId: body.request_id || requestId,
              details: body.details ?? {},
              retryAfterS,
            }),
          );
        } else {
          reject(errorFromStatus(request.status, requestId, retryAfterS));
        }
      };
      const form = new FormData();
      form.append('file', file, file.name);
      request.send(form);
    });

  const promise = (async () => {
    await ensureSession();
    if (aborted) throw new ApiError({ status: 0, code: 'aborted', message: 'Upload cancelled' });
    try {
      return await attempt();
    } catch (error) {
      if (!aborted && isApiError(error) && error.status === 401 && error.code === 'session_required') {
        await ensureSession(true);
        return attempt();
      }
      throw error;
    }
  })();

  return {
    promise,
    abort: () => {
      aborted = true;
      xhr?.abort();
    },
  };
}
