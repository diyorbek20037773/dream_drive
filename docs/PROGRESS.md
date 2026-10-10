# Implementation status

Living record of implemented, verified, blocked and remaining work. Times are UTC unless stated.
Source-brief claims are not evidence; only the "Evidence" column counts.

| Phase | Status | Evidence | Next action |
|---|---|---|---|
| 1. Repository/environment audit | Done 2026-10-10 11:06 | docs/ENVIRONMENT.md; empty public GitHub repo; gh authed; Docker 29.7.2 + nvidia runtime; GPU is RTX 3050 6 GB (not 4050) | – |
| 1. Checkpoint compatibility spike | Done 2026-10-10 11:20 | SHA256 f2a8c837…1f8c, 156,380,200 B; torch 2.14.1+cpu jit.load OK; 14 real frames; class map {3: vehicle}; config/models.resolved.yaml | Formal model-smoke gate |
| 2. Video vertical slice | In progress | – | Inference service + pipeline being built |
| 3. Durable API/worker | In progress | – | – |
| 4. UI | In progress | – | – |
| 5. LiteRoad pixel loop | In progress | – | – |
| 6. Evaluation/monitoring | In progress | – | – |
| 7. Containers + full QA | Not started | – | after integration |
| 8. GitHub + registry | Not started | – | after local QA |
| 9. Server deployment | Blocked: no server/domain/SSH supplied | – | deploy files prepared and validated locally |
| 10. Handoff/freeze | Not started | – | – |

## Log
- 2026-10-10 11:06 audit; 11:10 checkpoint downloaded and hashed; 11:20 real CPU spike passed; 11:45 dependency locks
  resolved with uv (backend, inference); shared Pydantic contracts written and validated against contracts/*.json.
- 2026-10-10 ~11:55 parallel build of components started (inference, API contract/backend, video pipeline, LiteRoad, frontend, DevOps).
