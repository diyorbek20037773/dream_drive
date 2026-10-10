-- DreamDrive schema v1 (docs/IMPLEMENTATION_SPEC.md section 3.2).
-- Timestamps are ISO-8601 UTC strings with a fixed "+00:00" offset and microseconds, so they sort lexically.
-- owner_id is the session id of the creator; every owner-facing query filters by it.

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    role TEXT NOT NULL CHECK (role IN ('anonymous', 'operator')),
    csrf_token TEXT NOT NULL,
    created_utc TEXT NOT NULL,
    last_seen_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_last_seen ON sessions (last_seen_utc);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'video_analysis',
    state TEXT NOT NULL CHECK (
        state IN ('queued', 'validating', 'running', 'cancelling', 'cancelled', 'succeeded', 'failed', 'interrupted')
    ),
    stage TEXT,
    progress REAL NOT NULL DEFAULT 0,
    frames_done INTEGER NOT NULL DEFAULT 0,
    frames_total INTEGER,
    error_code TEXT,
    error_message TEXT,
    input_sha256 TEXT NOT NULL,
    input_bytes INTEGER NOT NULL,
    input_ext TEXT NOT NULL CHECK (input_ext IN ('mp4', 'mov')),
    probe_json TEXT,
    model_json TEXT,
    settings_json TEXT,
    summary_json TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    -- Set when the owner deleted an in-progress job: hidden from the API, removed by the worker once it stops.
    delete_requested INTEGER NOT NULL DEFAULT 0,
    created_utc TEXT NOT NULL,
    started_utc TEXT,
    finished_utc TEXT,
    updated_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_owner_created ON jobs (owner_id, created_utc DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs (state);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    simulator TEXT NOT NULL,
    perception_source TEXT NOT NULL,
    scenario TEXT NOT NULL,
    seed INTEGER NOT NULL,
    start_node TEXT NOT NULL,
    destination_node TEXT NOT NULL,
    route_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('initializing', 'running', 'paused', 'stopping', 'stopped', 'arrived', 'failed')
    ),
    outcome TEXT,
    error_code TEXT,
    error_message TEXT,
    summary_json TEXT,
    obstacles_injected INTEGER NOT NULL DEFAULT 0,
    created_utc TEXT NOT NULL,
    started_utc TEXT,
    finished_utc TEXT,
    updated_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_owner_created ON runs (owner_id, created_utc DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_runs_state ON runs (state);

CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY,
    owner_id TEXT,
    kind TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('queued', 'running', 'succeeded', 'failed', 'interrupted')),
    created_utc TEXT NOT NULL,
    finished_utc TEXT,
    summary_json TEXT,
    error_code TEXT,
    -- Requested seeds and the id of the evaluation artifact the suite produced (registry id).
    seeds_json TEXT,
    artifact_id TEXT,
    updated_utc TEXT
);
CREATE INDEX IF NOT EXISTS idx_evaluations_owner_created ON evaluations (owner_id, created_utc DESC);
CREATE INDEX IF NOT EXISTS idx_evaluations_state ON evaluations (state);

-- Owner-scoped record of deleted jobs so a repeated DELETE by the same owner stays idempotent (200) while unknown
-- or foreign ids still answer 404. Purged by retention cleanup.
CREATE TABLE IF NOT EXISTS job_tombstones (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    deleted_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_job_tombstones_deleted ON job_tombstones (deleted_utc);
