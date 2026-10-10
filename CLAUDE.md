# DreamDrive engineering instructions

## Mission and facts
Build a working, reproducible DreamDrive AI application from these specifications.
Repository: https://github.com/diyorbek20037773/dream_drive
Docker Hub namespace confirmed by owner: diyorbek7377.
Developer GPU: NVIDIA RTX 4050, 6 GB VRAM. RAM and available disk are not yet known.
Developer environment: Windows with Ubuntu WSL2; verify the actual environment.
Server: not provisioned. Domain, SSH identity and server access are not supplied.
The source brief states a deadline of 2026-10-11 13:00 Asia/Tashkent.
Treat that date and competition rules as supplied context, not independently verified rules.
All application and model results are unimplemented until you generate evidence.
If the old source brief conflicts with the hardware-aware design, follow docs/DECISIONS.md.
The handoff author found no remote branch refs with git ls-remote; verify again.

## Read order
1. `docs/DECISIONS.md` and `docs/PRODUCT.md`.
2. `docs/ARCHITECTURE.md` and `docs/API.md` before backend work.
3. `docs/ML_PIPELINE.md` and `docs/SIMULATION.md` before ML/control work.
4. `docs/DESIGN.md` before frontend work.
5. `docs/QA.md`, `docs/DEVOPS.md`, `docs/MLOPS.md`, `docs/OBSERVABILITY.md`.
6. `docs/IMPLEMENTATION_PLAN.md`, `config/project.yaml`, `config/models.yaml`.
Read `docs/SOURCE_AUDIT_UZ.md` and the full source brief when checking provenance.
Do not load every long document into every task unnecessarily.

## Build rules
- Implement the application; do not finish with another plan, mockup or advice.
- Two real modes: uploaded-video analysis and simulator-only closed-loop driving.
- Never send actuator commands to a physical car. Do not promise unattended real driving.
- Uploaded MP4 is an observation recording; decisions cannot change its next frame.
- Never substitute simulator truth or scripted outcomes for pixel perception.
- Tag every perception and command with its source and mode.
- Never invent metrics, class maps, benchmark scores, hashes, image digests or deployment URLs.
- LiteRoad is the default simulator on this hardware. CARLA is an optional tested adapter.
- LiteRoad's bounded synthetic RGB perception is explicitly `synthetic_rgb_cv`.
- YOLOPv2 is the real-video baseline candidate, not proven to work on synthetic renders.
- Do not claim person/traffic-light recognition until the exact checkpoint supports it.
- No LLM in the driving/control loop. Explanation text derives from structured reason codes.
- Do not add RF-DETR, depth or a VLM to the default stack without measured need.
- Use deterministic FSM, route planning, controller and safety precedence.
- Use CPU inference by default for the portable image; optional CUDA image/profile.
- All user-facing controls must work; missing features are disabled with a clear reason.
- UI default is Uzbek Latin; expose an English switch. Source identifiers remain English.
- Logs, results and evaluation metadata must be honest about unsupported capabilities.

## Repository and execution
Inspect git status, remote URLs, branch refs and existing files before changing anything.
Preserve unrelated changes and existing project code; merge these instructions carefully.
Use a feature branch if code/history exists; for an empty repo create main normally.
Do not force-push, rewrite existing history or delete remote resources.
Use `docs/IMPLEMENTATION_PLAN.md` as the ordered execution plan.
Keep `docs/PROGRESS.md` current with implemented, verified, blocked and remaining work.
Run meaningful tests while building; final QA must test the exact committed source.
Generate real lockfiles from a working resolver; do not type plausible package versions.
Resolve all model revision/checkpoint/checksum fields before claiming real inference.
Keep model weights, uploads, datasets, generated results and credentials out of git.
Use approved public samples or owner-provided input; record provenance and licenses.
Use official technical documentation and pin vendored source commits with notices.

## User authorization and missing access
The owner requests GitHub push and Docker Hub publication after QA.
Proceed with these when existing authenticated credentials allow it; do not repeatedly ask.
First finish all local implementation and checks even when external credentials are absent.
Use `gh auth status` and Docker's existing credential flow without printing secrets.
Do not solicit passwords/tokens in chat or put them in source/config/logs.
When login is missing, provide one precise terminal login action and continue independent work.
Buying a server or spending money requires owner choice; no hosting purchase is authorized.
Server deployment is requested, but needs a target host/domain/access supplied by the owner.
Prepare deploy files and verify them locally; do not invent a successful external deployment.
Do not bypass native permission controls or ask for broad unrestricted permissions.

## Delivery gate
`make qa` must exit nonzero on failures; skips cannot count as passes.
Inspect desktop/mobile screenshots and ensure runtime buttons and error states work.
Run real-weight inference and a new uploaded video; do not use only mocked inference tests.
Test a clean CPU Docker deployment without host Python/Node or GPU.
Publish tested immutable images only after local QA and required CI succeed.
Use diyorbek7377/dreamdrive-web, dreamdrive-backend and dreamdrive-inference.
Optional GPU image: diyorbek7377/dreamdrive-inference-gpu.
Record commit, model SHA256, image digests, test evidence and supported platforms.
Server production config must pull published images by digest, with no local build section.
Do not claim ARM support until that image builds and passes inference on ARM.
Keep admin/monitoring services private; public uploads have quotas and retention.
Return working URLs only after a real browser/API check against the deployed service.
Finish with implemented features, evidence, limitations and exact remaining access blockers.
