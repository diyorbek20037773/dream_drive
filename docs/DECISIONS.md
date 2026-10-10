# Engineering decisions

| Area | Decision | Reason and validation |
|---|---|---|
| Scope | Real video analysis plus simulator-only closed loop | Follows supplied brief; keeps recorded video distinct from interactivity |
| Developer | Windows/WSL2, RTX 4050 6 GB | Owner answer; audit RAM/disk/driver rather than guessing |
| Baseline ML | YOLOPv2 official checkpoint candidate | Detection, lane and drivable outputs; exact classes/output verified first |
| Tracking | ByteTrack adapter, one state per run | Temporal consistency; timestamp-aware frame sampling |
| Simulation | LiteRoad CPU default | Small synchronous RGB feedback environment works without CARLA GPU |
| Lite perception | synthetic_rgb_cv from rendered pixels | Honest bounded synthetic CV; no simulator object coordinates as perception |
| Learned simulation | YOLOPv2 mode gated by scenario results | Domain shift must be measured, not hidden |
| CARLA | 0.9.16 candidate optional, client/server match | 8 GB+ recommendation exceeds current 6 GB; smoke test then record support |
| Backend | FastAPI, Pydantic, SQLite WAL, Redis/RQ, separate inference service | Small single-node durable product; avoid model per API process |
| UI | React, TypeScript, Vite, Tailwind, accessible components | Static web image; same-origin API avoids runtime localhost surprises |
| Agent | FSM + A* + Pure Pursuit/PID + deterministic safety | Testable decisions; no language model controls actuators |
| Delivery | Docker Compose, CPU-first linux/amd64 | Reproducibility first; ARM/CUDA explicit tested variants |
| CI | GitHub Actions, pytest, Playwright, Trivy, secret scan | Local QA before push, CI before release |
| Ops | JSON logs, Prometheus/Grafana optional profile | Measured jobs, latency, errors and resource conditions |
| MLOps | Hash manifests + MLflow evaluation runs | Trace checkpoints/data/config; no training claim without a run |
| Hosting | Ubuntu VPS runbook, Caddy HTTPS, published digests | Server not supplied; no assumed purchase or successful deploy |

Candidate package/runtime versions are selected and locked by Claude after install and compatibility tests. Prefer Python 3.11 if the actual checkpoint and chosen dependencies work; isolate legacy dependency constraints in inference instead of downgrading the entire app. Use an active Node LTS verified at implementation time. Do not blindly install the upstream 2022 requirements over a modern environment.

Scope ladder: P0 real video product → P1 honest synthetic closed loop → P1b learned simulator adapter after validation → P2 optional CARLA/second detector/fine-tuning. Deliver all P0/P1 tasks before optional research. If learned sim fails, retain the P0 and synthetic labels instead of calling the fallback learned autonomy.

## Decision record 2026-10-10: second detector enabled (owner request)
The verified YOLOPv2 checkpoint detects one class (`vehicle`, class index 3). The owner asked that people, traffic lights
and signs be detected and that accuracy be measured. This meets the conditional trigger in docs/ML_PIPELINE.md for an
extra detector. RF-DETR-N (rfdetr 1.11.2, Apache 2.0 per the upstream model table, COCO-pretrained) is added beside
YOLOPv2 in the same inference service: YOLOPv2 keeps lane/drivable segmentation and `vehicle`; RF-DETR adds person,
bicycle, motorcycle, bus, truck, traffic_light and stop_sign. General traffic-sign categories other than stop sign remain
unsupported (no verified, licence-clear checkpoint). Traffic-light colour is a labelled HSV heuristic, not model output.
Detector accuracy is measured on a COCO val2017 street-scene subset through the production inference path; lane and
drivable accuracy remain not evaluated until licensed labels (e.g. owner-registered BDD100K) are supplied.
Details and wire format: docs/IMPLEMENTATION_SPEC.md section 2.1.
