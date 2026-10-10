# API and state contract

Base `/api/v1`; return UTC ISO timestamps and explicit seconds/units. Client types come from generated OpenAPI after implementation. Cookie-authenticated sessions own resources. JSON error envelope: `code`, `message`, `request_id`, `details` with no internal path/stack trace.

| Endpoint | Behavior |
|---|---|
| GET /health/live | API process liveness |
| GET /health/ready | API DB/queue readiness; model capabilities separate |
| GET /api/v1/capabilities | Model/device/classes and simulator modes with unavailable reasons |
| POST /api/v1/session | Establish bounded anonymous session or configured login |
| POST /api/v1/videos | Multipart file, validated; 202 with job ID, no synchronous inference |
| GET /api/v1/jobs | Owner's jobs and bounded pagination |
| GET /api/v1/jobs/{id} | State/progress/error and immutable model identity |
| POST /api/v1/jobs/{id}/cancel | Idempotent cancellation request |
| GET /api/v1/jobs/{id}/results | Structured result summary and protected artifact links |
| GET /api/v1/jobs/{id}/artifacts/{name} | Allowed artifacts only, range support for MP4 |
| DELETE /api/v1/jobs/{id} | Idempotent delete with in-progress coordination |
| GET /api/v1/simulators | Supported adapters, modes and road graphs |
| POST /api/v1/runs | Start/destination/scenario/seed/perception mode; 202 accepted |
| GET /api/v1/runs/{id} | State, allowed ego telemetry, decision, command, run metadata |
| POST /api/v1/runs/{id}/control | action=pause/resume/stop; idempotent transitions |
| POST /api/v1/runs/{id}/obstacles | Synthetic scenario injection only, bounded validation |
| GET /api/v1/runs/{id}/frame | Latest actual JPEG, frame/time identity headers, no cache |
| GET /api/v1/runs/{id}/report | Measured report; running state reports unavailable |
| GET /api/v1/evaluations | Real evaluation artifacts, sources and sample counts |
| GET /api/v1/events | Owner-filtered SSE, event ID and reconnect/resync |
| GET /metrics | Internal Prometheus route; public gateway denies it |

Job states: queued, validating, running, cancelling, cancelled, succeeded, failed, interrupted. Run states: initializing, running, paused, stopping, stopped, arrived, failed. Stop after arrived returns that terminal state and does not start another run. Unsupported mode/scenario returns 422 with reason. Busy/quota returns 429; absent ready model returns 503.

SSE message follows `contracts/event.schema.json`. Store sufficient state to recover UI after reconnect; missed historical camera frames do not need replay. Artifact links remain owner-checked; SSE filters cannot leak another owner's IDs. Internal inference endpoint accepts a bounded image/frame identity, returns validated detection/mask metadata, and is never routed publicly.

Security: reject oversized, corrupt, decompression-heavy or duration-limit input; use `ffprobe`/`ffmpeg` as argument arrays, timeouts, no shell interpolation. Do not accept arbitrary remote media URLs in MVP. Safe generated names, private temp dirs, path allowlists, decode watchdog, cleanup on exception and output faststart are required. Do not trust Content-Type or extension alone.
