"""Prometheus metric objects (names and bounded labels from docs/OBSERVABILITY.md).

Labels are bounded enumerations only (route templates, methods, status classes, queue names, model ids, reason
codes); job/run ids, URLs, file names, timestamps and exception strings are never labels.

The API process uses the default registry. The worker sets ``PROMETHEUS_MULTIPROC_DIR`` before importing this module
(see app/jobs/worker_main.py), so values written by the RQ child processes are aggregated by the supervisor's
``MultiProcessCollector``. Gauges therefore declare a multiprocess mode. Queue depth/age, heartbeat age and artifact
bytes are computed at scrape time by the worker's collector (app/jobs/monitoring.py) from Redis and the data volume.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "dreamdrive_http_requests_total",
    "HTTP requests handled by the API.",
    ["route", "method", "status_class"],
)
HTTP_DURATION = Histogram(
    "dreamdrive_http_request_duration_seconds",
    "API request handling time (excludes long-lived SSE streams).",
    ["route", "method"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
INFERENCE_DURATION = Histogram(
    "dreamdrive_inference_duration_seconds",
    "Model forward-pass time reported by the inference service per analyzed frame (excludes HTTP, queue, encode).",
    ["model", "device", "outcome"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
)
FRAMES_PROCESSED = Counter(
    "dreamdrive_frames_processed_total",
    "Frames processed by the worker, by mode, perception source and outcome.",
    ["mode", "source", "outcome"],
)
JOBS_TOTAL = Counter(
    "dreamdrive_jobs_total",
    "Jobs that reached a terminal status, by kind (video_analysis, simulation, evaluation).",
    ["kind", "status"],
)
MODEL_READY = Gauge(
    "dreamdrive_model_ready",
    "1 when the inference service reports the verified model ready, as last observed by this process.",
    ["model", "device"],
    multiprocess_mode="livemostrecent",
)
FRAME_ACQUISITION_AGE = Gauge(
    "dreamdrive_frame_acquisition_age_seconds",
    "Wall-clock age of the latest simulator camera frame when perception consumed it (active run).",
    ["source"],
    multiprocess_mode="livemostrecent",
)
SAFETY_OVERRIDES = Counter(
    "dreamdrive_safety_overrides_total",
    "Simulator ticks where the safety supervisor overrode the controller command.",
    ["reason", "perception_source"],
)
SIM_REALTIME_FACTOR = Gauge(
    "dreamdrive_simulation_realtime_factor",
    "Measured simulated seconds per wall second of the active run (<= 1 by pacing).",
    ["simulator", "source"],
    multiprocess_mode="livemostrecent",
)

# Families produced at scrape time by app/jobs/monitoring.py (documented here so all names live in one place).
QUEUE_DEPTH_NAME = "dreamdrive_queue_depth"
QUEUE_OLDEST_AGE_NAME = "dreamdrive_queue_oldest_age_seconds"
WORKER_HEARTBEAT_AGE_NAME = "dreamdrive_worker_heartbeat_age_seconds"
ARTIFACT_BYTES_NAME = "dreamdrive_artifact_bytes"
DATA_VOLUME_FREE_NAME = "dreamdrive_data_volume_free_bytes"


def status_class(status: int) -> str:
    return f"{status // 100}xx" if 100 <= status < 600 else "other"
