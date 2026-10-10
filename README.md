# DreamDrive AI

Road-video perception and simulator evaluation platform. Two real modes:

1. **Uploaded-video analysis.** Upload an MP4/MOV dash-cam recording. A verified learned ensemble
   (YOLOPv2 + RF-DETR-N) finds drivable area, lane markings, vehicles, people, two-wheelers, traffic lights and stop
   signs, tracks them (ByteTrack) and writes an annotated H.264 video, a timeline and JSON/HTML reports with
   analysis-only recommendations (continue / slow down / stop). An uploaded video is an observation recording:
   nothing can change its next frame and no vehicle is controlled.
2. **Simulator-only closed loop (LiteRoad).** Choose a start and destination on a virtual road graph. The car renders
   its own camera image, perceives it with explicit pixel computer vision (`synthetic_rgb_cv`), and a deterministic
   FSM, A* planner, Pure Pursuit/PID controller and safety supervisor drive it. The UI shows the live maneuver
   (stop, brake, slow down, accelerate, cruise, turn left/right, hold) and the exact inputs each decision used.

No LLM is in the driving loop and DreamDrive never sends commands to a physical vehicle. The UI defaults to Uzbek
(Latin) with an English switch. The label "Simulyatsion prototip / Simulation prototype" is always visible.

## Quick start (Docker only)
Requirements: Docker Engine 25+ with Compose 2.24+ (Docker Desktop on Windows/macOS), ~10 GB free disk, internet for
the first model download. No host Python, Node or GPU needed.

```bash
git clone https://github.com/diyorbek20037773/dream_drive.git
cd dream_drive
make up          # Git Bash / WSL / Linux: builds, downloads + verifies models, starts, waits until ready
# open http://127.0.0.1:8080
```
Without make (PowerShell works too):
```powershell
$env:GIT_COMMIT = git rev-parse HEAD
docker compose up -d --build
```
`model-init` downloads both checkpoints from their official URLs and verifies size and SHA256 (522 MB in total;
interrupted downloads resume). Later starts work offline from the `model-cache` volume.

Published images (pull-only, no build): see [Release images](#release-images) and `compose.prod.yaml`.
Optional NVIDIA GPU inference: `make up GPU=1` (needs the NVIDIA container runtime; tested on an RTX 3050 6 GB).
Monitoring (Prometheus, Grafana, MLflow on 127.0.0.1): `make up OPS=1`.

### Sample input
Use your own forward-facing, landscape, daytime dash-cam clip (best), or the licensed clips that
`python scripts/fetch_fixtures.py` downloads into `tests/fixtures/cache/` (Wikimedia Commons, CC0 and CC BY 3.0; see
docs/DATA_SOURCES.md). Do not use videos you have no rights to. Limits: 100 MB, 120 s, 1920x1080.

## What is verified (measured, not claimed)
| Item | Result | Evidence |
|---|---|---|
| YOLOPv2 checkpoint | SHA256 `f2a8c837…1f8c`, 156,380,200 B, upstream commit 9388c6a | config/models.resolved.yaml |
| RF-DETR-N checkpoint | SHA256 `d8d6b9ee…440d`, 366,287,238 B, publisher MD5 matched, Apache 2.0 | config/models.resolved.yaml |
| Object detection (7 RF-DETR classes + YOLOPv2 vehicle) | COCO val2017 street subset, 300 images: mAP50 0.684, mAP50:95 0.462 | docs/EVALUATION.md |
| Drivable area (YOLOPv2) | KITTI Road training, 289 images: IoU 0.847, F1 0.917 (perspective space) | docs/EVALUATION_SEGMENTATION.md |
| Lane markings (YOLOPv2) | Roboflow lane-marking v2 (CC BY 4.0), 150 images: F1 0.455 exact, 0.615 at 4 px | docs/EVALUATION_LANES.md |
| LiteRoad closed loop | 15/15 expected outcomes over 5 scenarios x 3 seeds, 0 collisions | docs/SIMULATION_EVIDENCE.md |
| CPU inference (2 models) | about 0.6-1.0 s per frame on an i5-13420H in Docker; 10 s clip end-to-end in ~33 s | docs/MODEL_CARD.md |

Accuracy numbers are cross-dataset checks on the stated subsets, not real-road safety claims. Traffic-light colour is
an HSV heuristic (labelled as such). Not supported: traffic signs other than stop signs, metric distance/TTC from the
camera, night/rain validation, learned perception inside the simulator (it failed its validation and stays disabled).

## Architecture
`gateway` (Caddy) → `web` (React/TypeScript static app) and `api` (FastAPI). `worker` (RQ) runs video jobs, the
LiteRoad simulator and evaluations; `inference` (PyTorch, one process, both models) is internal only; `redis` holds
queues and events; SQLite (WAL) and artifacts live in the `app-data` volume. Details: docs/ARCHITECTURE.md,
docs/IMPLEMENTATION_SPEC.md, API contract in docs/API.md and `frontend/src/api/openapi.json`.

## Development and QA
All gates run in containers: `make lint typecheck test integration model-smoke scenarios e2e security` or the full
release gate `make qa` (evidence in `artifacts/qa/<commit>/`, non-zero on any failed or skipped gate).
More: docs/OPERATIONS.md (commands, Windows notes), docs/QA.md, docs/RUNBOOK.md (incidents), docs/DEPLOY.md (VPS).

## Release images
Published image tags, digests, the tested commit and the pull-only smoke result are recorded in
[docs/RELEASE.md](docs/RELEASE.md) once a release has been published and verified.

## Status and limits
docs/PROGRESS.md records what is implemented, verified and blocked. Server deployment is prepared (Caddy HTTPS,
pull-only `compose.prod.yaml`, backup/rollback scripts) but no server, domain or SSH access has been supplied, so
nothing is deployed publicly.

## Licences
The owner has not chosen a licence for DreamDrive's own code yet (no LICENSE file; all rights reserved until one is
added). Third-party code, model weights and datasets keep their own licences: docs/LICENSES.md,
docs/DATA_SOURCES.md, docs/MODEL_CARD.md. Model weights are downloaded at install time, never redistributed in images.
