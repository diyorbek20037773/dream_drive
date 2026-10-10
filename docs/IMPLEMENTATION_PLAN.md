# Ordered implementation and completion criteria

This plan instructs Claude to finish working software. It does not certify feasibility within the source deadline. Treat phases as gates; update PROGRESS after evidence, not after file creation.

1. **Audit and compatibility spike.** Check repository/status/remote/auth, WSL/Docker/GPU/RAM/disk. Preserve existing work. Prove checkpoint download/hash/class map/input-output/real frame on CPU, then optional CUDA. Output ENVIRONMENT.md and resolved manifest/locks. Timebox optional CARLA compatibility, not the required real inference.
2. **Video vertical slice.** Internal inference service, decode/invert masks+boxes, short MP4 annotation and structured report. Verify browser codec/timestamps. This precedes building an elaborate dashboard. No stub inference passes this phase.
3. **Durable API/worker.** Ownership, upload admission, queue, persisted states, cancellation, artifacts/ranges, events and restart cleanup. Complete endpoints/contract tests and frontend type generation.
4. **Polished UI.** Implement navigation/pages/i18n/media overlay/decision timeline/results/errors, using actual endpoints. Inspect three responsive screenshots and browser logs. Every exposed action works.
5. **LiteRoad pixel loop.** Renderer/dynamics → actual synthetic RGB perception → FSM/controller/supervisor → feedback. Destination selection and normal/safety reports. Add pixel ablation and hidden-truth isolation tests. Only enable learned simulator perception after its own scenario tests.
6. **Evaluation and monitoring.** Licensed labeled fixture/manifest, actual baseline, MLflow evidence, JSON logs, working Prometheus/Grafana dashboards, alert smoke and recovery behavior. Unsupported accuracy metrics remain null.
7. **Containers and full QA.** CPU default reproducibility, optional GPU profile, frozen locks, security scan, real-weight smoke, clean browser flows, scenario seeds and backup/restore. Commit final source, then test clean HEAD. Record results/artifacts.
8. **GitHub and registry.** After local QA push authorized target; verify CI for same SHA; build/tag/scan/publish tested images, capture digests and run fresh published-image smoke. Auth blockers do not stop independent implementation.
9. **Server readiness/deployment.** Prepare pull-only prod config/Caddy/secrets/backup/rollback scripts and verify them locally. Obtain actual owner-selected server/domain/access when available. Deploy and publicly smoke same digests. Without server, status is “deployment prepared; target not supplied”, not “deployed”.
10. **Handoff/freeze.** README, model/data/license docs, commands, supported hardware, actual limits, QA report and release manifest. If competition freeze applies, record exact submitted commit/deployed digest. Do not change frozen submission without owner instruction.

## Definition of done
A new user can clone and follow CPU instructions using Docker only, obtain verified model assets, upload a new supported video and get actual annotated output. They can select a virtual destination, run a genuine pixel-driven synthetic loop and see the declared source and measured result. All surfaced controls work. CI/release artifacts identify the same tested source. Published images are pulled and smoked. External deployment is complete only after actual target access and HTTPS check; otherwise the precise remaining blocker is disclosed.

## If the date leaves less than one day
Spend first hours on checkpoint/video and pixel-loop integration risks; next on API/UI; reserve final hours for portability/QA/publish. Stop optional CARLA, VLM, depth, fine-tuning and multiarch work when they threaten P0/P1. Do not drop required checks to look complete. Implemented/verified/blocked are separate states. Resume from PROGRESS after session compaction.
