# Implementation status

Living record of implemented, verified, blocked and remaining work. Times are UTC. Source-brief claims are not evidence;
only the "Evidence" column counts. Local evidence lives under `artifacts/` (gitignored); CI evidence is attached to the
GitHub Actions runs.

| Phase | Status | Evidence | Next action |
|---|---|---|---|
| 1. Environment audit | Done 2026-10-10 11:06 | docs/ENVIRONMENT.md. GPU is RTX 3050 6 GB (owner said 4050) | – |
| 1. Checkpoint spike | Done 11:20 | YOLOPv2 SHA256 f2a8c837…1f8c; torch 2.14.1 CPU; class map {3: vehicle} | – |
| 2. Video vertical slice | Done | Real-weight runs on a CC0 dash-cam clip and a CC BY 3.0 street clip; annotated H.264 re-opened; 0 contract errors | – |
| 2b. Second detector (owner request) | Done | RF-DETR-N SHA256 d8d6b9ee…440d (publisher MD5 matched); offline start with `--network none`; docs/DECISIONS.md | – |
| 3. Durable API/worker | Done | make qa unit/security/integration gates; restart reconciliation and backup/restore tested | – |
| 4. UI | Done | Playwright 36 tests on 3 viewports against the live stack (make qa e2e gate); screenshots inspected | – |
| 4b. Maneuver + decision basis (owner request) | Done | Live screenshots: "Tezlashish", "To‘xtash" with "To‘siq belgisi 0.78 ≥ 0.70 → To‘xtash" | – |
| 5. LiteRoad pixel loop | Done | pixel ablation + hidden-truth isolation tests; scenario suite 15/15 expected outcomes, 0 collisions | – |
| 6. Evaluation | Done (measured) | COCO street subset 300 img: mAP50 0.684, mAP50:95 0.462; KITTI Road 289 img: drivable IoU 0.847, F1 0.917; Roboflow lane-marking v2 150 img: F1 0.455 exact / 0.615 at 4 px. evaluations/baseline/, docs/EVALUATION*.md | – |
| 6. Monitoring | Prepared + rule-tested | promtool firing-transition tests; Grafana provisioning | Live OPS stack is optional |
| 7. Containers + full QA | Done | make qa 21/21 on 267f303 (local) and in CI | – |
| 8. GitHub + registry | Done | main = 267f303 pushed; CI run 38077565137 success; images published and pull-only smoke 14/14 (docs/RELEASE.md) | Optional GPU image |
| 9. Server deployment | Blocked: no server/domain/SSH supplied | deploy/ scripts, compose.prod.yaml, Caddyfile.prod; the exact deploy.sh path ran locally with HTTPS (internal CA) | Owner supplies host, domain, SSH user |
| 10. Handoff | Done | README.md, docs/RELEASE.md, MODEL_CARD, DATA_SOURCES, LICENSES, EVALUATION*, RUNBOOK, DEPLOY | – |

## Findings that changed the build
- model-init hit its 30 min total timeout at 95 % of the 366 MB RF-DETR checkpoint on a slow link; downloads now resume
  with HTTP Range and bounded retries (b4efe0c).
- Video decisions accepted tracked boxes with scores down to 0.1; decisions, burned-in boxes and report counts now need
  ≥ 0.5 (untracked) or ≥ 0.4 (confirmed track); low-score detections are listed separately (2ccdbdd).
- make qa found Windows path handling bugs, a non-hermetic test, missing manifests in the test image, an MLflow
  identity bug and a COOP console notice on the HTTP e2e runner (02f56eb).
- CI on a clean Linux runner found root-owned bind-mount directories, a gate-order dependency, npm ci deleting a
  scratch dir (26b686f) and CPU-dependent bytes when re-encoding a fixture clip; derived clips are now verified by
  decoded properties while the source SHA256 stays mandatory (267f303).
- Learned (YOLOPv2) perception in LiteRoad failed its scenario validation, so `LEARNED_SIM_MODE_ENABLED` stays false.
- Docker Desktop's engine was found stopped after a session restart and was restarted.

## Remaining / blocked
- Public deployment: needs a server (host/IP, OS), a domain with DNS, an SSH user and key path on the owner's machine.
- CI-based publishing: needs `DOCKERHUB_TOKEN` (and `DOCKERHUB_USERNAME=diyorbek7377`) set as GitHub secrets by the owner.
- Project code licence: not chosen by the owner yet.
- Not evaluated: night/rain, traffic signs other than stop signs, traffic-light colour accuracy (heuristic), metric
  distance/TTC, ARM images.
