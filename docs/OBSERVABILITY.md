# Monitoring and incident behavior

Implement structured JSON logs with UTC time, level, component, request/job/run ID, frame ID when needed, reason/error code, timing and source. IDs belong in logs/traces, not unbounded metric labels. Never log credentials, cookies, raw uploaded images, personal filenames or image content. Redact query/header secrets.

Prometheus `/metrics` is internal; one process per inference service simplifies collection. Worker exposes its own monitored metrics endpoint or uses a documented supported aggregation pattern. Don't pretend an API-only metric measures worker inference. Health/readiness and actual queue timestamps are separate signals.

| Metric | Type and bounded labels |
|---|---|
| dreamdrive_http_requests_total | Counter: route template, method, status class |
| dreamdrive_http_request_duration_seconds | Histogram: route template, method |
| dreamdrive_inference_duration_seconds | Histogram: model, device, outcome |
| dreamdrive_frames_processed_total | Counter: mode/source, outcome |
| dreamdrive_jobs_total | Counter: kind, terminal status |
| dreamdrive_queue_depth | Gauge: queue |
| dreamdrive_queue_oldest_age_seconds | Gauge: queue |
| dreamdrive_model_ready | Gauge: model/device |
| dreamdrive_frame_acquisition_age_seconds | Gauge: source; limited active run slots |
| dreamdrive_safety_overrides_total | Counter: reason, perception source |
| dreamdrive_simulation_realtime_factor | Gauge: simulator/source |
| dreamdrive_worker_heartbeat_age_seconds | Gauge: worker slot |
| dreamdrive_artifact_bytes | Gauge: artifact category |

No job/run ID, raw URL, exception string, timestamp or filename as labels. FPS is derived from actual analyzed-frame count/time; histogram inference latency excludes queue/encoding and reports that distinction. GPU VRAM/driver metrics are optional and unavailable on CPU, not zero by default. Host/node exporter may need documented mounting privileges; do not add privileged containers without necessity.

Provision Grafana dashboards from code: service health, queue, end-to-end stage timing, real inference latency/throughput, synthetic scenario state, disk and resource indicators. Ops profile must have real scrape targets and persistence. Default admin credentials are generated privately. Alert rules: service unavailable >2 min, worker heartbeat stale, oldest job above configured SLA, elevated failed jobs, disk low, model not ready. Validate an actual firing transition with promtool/test fixture or controlled outage. Alert destinations are left unconfigured until supplied; do not send messages to third parties by default.

Optional OpenTelemetry correlation is P2 if multi-service diagnosis needs it. JSON logs + metrics are mandatory. Adding Loki/Tempo/cAdvisor everywhere is not required for this single-node demo. Monitor failure cannot stop the control loop. Provide a runbook for model-not-ready, GPU OOM, corrupt video, queue backlog, disk full, worker crash, secret compromise and bad release, with symptoms, diagnostics, safe recovery and evidence to retain.
