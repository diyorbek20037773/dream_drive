# DreamDrive product requirements

The product is a road-video perception and simulation evaluation platform. It accepts new video input, produces actual model results, explains deterministic decision rules and lets a user run a bounded virtual route with camera feedback. A small persistent label says “Simulyatsion prototip” / “Simulation prototype”.

## Video analysis acceptance
Upload supported MP4/MOV, validate contents, create durable job, display queue/progress, cancel work, show original and annotated media, browse timestamped decisions/tracks, download annotated MP4 and JSON/HTML report, delete input/results. Model output and report must be generated from the uploaded bytes. Confidence is detector confidence; do not label it an overall safety probability. Unsupported class/metric is visibly unavailable.

## Simulation acceptance
Select simulator and perception source, choose reachable start/destination on the virtual road graph, start/pause/resume/stop, see actual camera/route/ego state/decision and control signals, optionally inject a permitted visual obstacle, then get measured completion, collisions, lane deviation and timing. Stop and state changes must reach the worker. A successful run is measured, never pre-scripted. Failure and unsupported combinations remain visible.

## Evaluation and system
List versioned evaluation runs with real sample counts, labels, metrics and limitations. Separate actual ML evaluation from synthetic scenario validation. System page shows model readiness, selected device, commit/build, supported class list, simulator capability and queue capacity. Monitoring uses measured values. Empty results show an explanation, not demo percentages.

## Persistence and user boundaries
Use anonymous signed session cookies for a public demo or a small configured login. Per-session ownership guards every job/run/artifact; job IDs are not authorization. An operator role protects destructive/admin actions. CSRF defense applies to cookie-authenticated mutations. HttpOnly/Secure/SameSite cookies in HTTPS production. Do not put reusable API tokens in URLs or localStorage. Demo limits apply even to anonymous sessions.

Defaults: 100 MB per upload, 120 seconds, 1920x1080 decoded maximum, two queued uploads per session, ten queued jobs globally, ten expensive requests per minute per session/IP, 120 seconds maximum simulation wall time, one active simulation per instance, one ML inference request at a time. Admission denies queue overload clearly. Production retention is 24 hours; scheduled cleanup removes bytes and DB references. These are configurable engineering defaults, not measured server capacity.

## Explicit exclusions
Real vehicle integration, public-road Level 4 claims, unattended-driver promises, metric depth from uncalibrated RGB, LLM steering and fake real-time dashboards. Road junction traffic-law behavior is unsupported unless implemented and tested. Do not render an unsupported control as if it works.
