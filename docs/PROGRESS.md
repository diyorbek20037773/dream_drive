# Implementation status

Living record of implemented, verified, blocked and remaining work. Times are UTC. Source-brief claims are not evidence;
only the "Evidence" column counts. Local evidence files live under `artifacts/` (gitignored) unless stated otherwise.

| Phase | Status | Evidence | Next action |
|---|---|---|---|
| 1. Environment audit | Done 2026-10-10 11:06 | docs/ENVIRONMENT.md. GPU is RTX 3050 6 GB (owner said 4050). Empty public GitHub repo; gh authed; Docker Hub login diyorbek7377 | – |
| 1. Checkpoint spike | Done 11:20 | YOLOPv2 SHA256 f2a8c837…1f8c; torch 2.14.1 CPU; class map {3: vehicle} | – |
| 2. Video vertical slice | Done | Real-weight run on a CC0 dash-cam clip (50 analysed frames, H.264 output re-opened, 0 contract errors); two-model run on a CC BY 3.0 Memphis street clip (people, traffic lights, stop signs; 6 controlled_stop recommendations on red-light heuristic) | Re-measure latency on an unloaded host |
| 2b. Second detector (owner request) | Done | RF-DETR-N (Apache 2.0) SHA256 d8d6b9ee…440d, publisher MD5 matched; offline start verified with `--network none`; model smoke 9/9 CPU; GPU fp16 smoke; docs/DECISIONS.md record | – |
| 3. Durable API/worker | Done | 553 backend unit/security/scenario tests pass on host; 400 in the Linux test image; live integration 7/7 (earlier stack) | Re-run in final QA |
| 4. UI | Done, live-verified for shell + simulation | Playwright vs live stack: 7/7 (shell, language, navigation, system, learned-mode reason, start→pause→resume→stop, obstacle stop); screenshots inspected | Video flows once the inference service is up |
| 5. LiteRoad pixel loop | Done | 107+ tests incl. pixel ablation and hidden-truth isolation; scenario suite 15/15 expected outcomes, 0 collisions; maneuver + decision-basis panel (owner request) | – |
| 6. Evaluation | Done (measured) | COCO val2017 street subset 300 images: mAP50 0.684, mAP50:95 0.462; KITTI Road 289 images: drivable IoU 0.847, F1 0.917; Roboflow rtqi/lane-marking v2 (CC BY 4.0) 150 images: lane F1 0.455 exact / 0.615 at 4 px tolerance. Records in evaluations/baseline/, method in docs/EVALUATION*.md | – |
| 6. Monitoring | Prepared | promtool rule tests pass (inactive→pending→firing); Grafana provisioning checked | Live scrape in final QA (OPS=1) |
| 7. Containers + full QA | In progress | `make up` stack healthy except inference (see log) | `make qa` on the final commit |
| 8. GitHub + registry | Not started | – | Push after local QA; CI; publish images |
| 9. Server deployment | Blocked: no server/domain/SSH supplied | deploy/ scripts, compose.prod.yaml, Caddyfile.prod validated locally | Owner supplies host/domain/SSH user |
| 10. Handoff | Not started | – | README, release manifest |

## Findings that changed the build
- 2026-10-10 16:38 model-init hit its 30 min total timeout at 95 % of the 366 MB RF-DETR checkpoint on a slow link and
  deleted the partial file. Fixed in b4efe0c: resumable downloads (HTTP Range), bounded retries, 1 h total timeout,
  4 new unit tests. The local stack volume was seeded with the locally verified checkpoint; model-init re-verifies it.
- The Docker Desktop engine was found stopped after a session restart (20:50 Tashkent) and was restarted.
- Learned (YOLOPv2) perception in LiteRoad failed its scenario validation (the orange obstacle is not a YOLOPv2 class),
  so `LEARNED_SIM_MODE_ENABLED` stays false and the UI shows the reason.

## Log
- 11:06 audit; 11:10 checkpoint downloaded; 11:20 real CPU spike; 11:27 parallel build started.
- 13:00 first build phase complete (7 components); 13:14 two-model inference ready; 13:34 maneuver panel.
- 13:44 COCO and KITTI evaluations measured; 15:58 commit fa64cd0; 16:10 lane evaluation measured.
- 16:18 live stack: simulation integration 4/4, Playwright 7/7; 16:22 baseline evaluations wired (bfc8ae9).
- 16:45 model-init resume fix (b4efe0c); stack rebuild in progress.
