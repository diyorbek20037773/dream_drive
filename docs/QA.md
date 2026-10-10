# Quality gates and evidence

QA belongs to the application Claude implements. Handoff syntax verification is separate. Require meaningful behavior tests; don't mirror implementation or inflate coverage with trivial getters. Mock tests are useful but cannot replace true inference and closed-loop acceptance.

## Required commands to implement
`make doctor`, `make lint`, `make typecheck`, `make test`, `make integration`, `make model-smoke`, `make e2e`, `make scenarios`, `make security`, `make qa`, `make smoke-published`, `make backup-test`. Add documented Windows PowerShell equivalents or route through WSL. Full release QA invokes all required gates, preserves outputs and fails nonzero. No `|| true`, suppressed failed subprocess or skipped required test may produce a green release.

## Gates
| Gate | Required evidence |
|---|---|
| Static | Python lint/type checks, frontend lint/tsc/build, generated schema/type consistency |
| Real model | Verified checkpoint on actual frame, nonempty masks with shape sanity, class/output audit; CPU and GPU separately if claimed |
| Video pipeline | New MP4 → real inference → annotated playable MP4 + JSON/report; timestamp and box/mask inverse transforms |
| Robustness | Invalid/corrupt/oversized/long/VFR/rotated video, model absent/bad hash, decode timeout, inference timeout, cancel/delete cleanup |
| Ownership | Two sessions cannot access each other's jobs, artifacts or SSE; traversal/CSRF/upload checks |
| Tracking | Per-run isolation, low-score second association, sampled timestamp gap, reset/occlusion behavior |
| Controller | Coordinate sign/units, PID anti-windup, clamp/NaN, FSM precedence, reachable/unreachable routes |
| Pixel separation | Hidden truth changed + same RGB gives same runtime command; RGB occlusion + same ego/route changes perception/control state |
| Simulation | Normal arrival, bend, obstacle stop, camera loss, pause/resume/operator stop; real state evolution and reports |
| Browser | Playwright full upload/download/seek/control flows, reconnect, responsive states, no JS/network errors |
| Portability | Fresh CPU Compose using only registry images and named volumes; no host runtime/GPU/source mount required |
| Security | Secret scanner, dependency audit, Trivy image scan, private admin ports and resource/input caps |
| Monitoring | Metrics scrape actual observations; simulated service-down/queue-age alert enters firing state without public alert spam |
| Recovery | Restart reconciliation, retained completed result, data backup and restore into a separate test volume |

A labeled evaluation subset is required for any accuracy claim. Include actual labeled-frame count and provenance. If no lane/track labels exist, corresponding metrics are null/not evaluated and do not gate a fabricated score. Release product must still have true inference smoke evidence. Evaluate synthetic scenarios independently of learned-video accuracy.

Suggested functional targets: 3 seeds of normal route finish within a stated tick budget with zero synthetic collisions; obstacle safety cases stop or fail visibly rather than falsely arrive; unobservable frame forces safety state before further unsafe stepping. These are test assertions for the bounded synthetic environment, not guarantees about real roads. Choose nonflaky numerical tolerances based on calibrated units and evidence.

Performance reports include hardware/driver/RAM, image digest, model SHA, input size, precision, warmups, samples, cold-start, P50/P95 latency, effective analyzed FPS, real-time factor and peak memory. Target values are requirements/hypotheses; observed values are separate. CPU may process offline video slower than playback. If CUDA is not tested, label it unverified and do not publish it as validated.

## Exact source release
Commit final source before full release QA, then run on clean HEAD. Evidence goes to ignored `artifacts/qa/<commit>/` and CI artifacts; do not modify tested source afterward. If fixes are made, commit new source and rerun relevant and final gates. Record source tree, commit, dependency/model identities, command exit codes, failed/skipped tests and screenshots. A release workflow runs required QA against the same commit before publication. Mutable latest tags are never the deploy source of truth.

CPU runner CI can run real model smoke with an authorized tiny fixture and verified model cache; no GPU is required. If downloads/weights are missing, this gate is blocked, not passed. GPU/CARLA tests can be separately marked unsupported, but an image/profile carrying a tested-support claim must have its own recorded smoke.
