# Build, registry and deployment requirements

## Local Compose and images
Implement `compose.yaml` for CPU default, `compose.gpu.yaml` override, `compose.ops.yaml` for monitoring/MLOps and `compose.prod.yaml` for registry-only deployment. Validate every combination with Docker Compose. Keep service names stable: gateway, web, api, worker, redis, inference. CPU needs no NVIDIA packages on the host.

Publish to Docker Hub:
- `diyorbek7377/dreamdrive-web`: static browser app.
- `diyorbek7377/dreamdrive-backend`: API/worker entrypoints, FFmpeg as needed, no loaded model.
- `diyorbek7377/dreamdrive-inference`: CPU runtime and real inference adapter.
- `diyorbek7377/dreamdrive-inference-gpu`: only if built/smoked, supported CUDA/driver contract.

Multi-stage builds, frozen locks, nonroot users, explicit workdirs, `.dockerignore`, bounded temp storage and OCI commit/version labels. Pin third-party services/base images by verified digest in the release manifest. A candidate version in a guide is not a real image digest. Do not distribute weights in an image until redistribution permissions are confirmed; otherwise use a checksum-verified one-command model-init service and persistent weights volume. First download needs network; subsequent startup must work from cache. Do not require the original developer's local model directory.

Use named volumes for app data, models, Redis persistence, Caddy state and ops data. Initialization creates ownership correctly for nonroot UID. Healthchecks and `depends_on` health ordering cannot replace application retries. Protect internal ports; local ops binds loopback. Restrict read-only filesystems where practical and permit only explicit writable volumes/tmpfs. Compose secrets or host protected env handles secrets; never bake them into layers or VITE variables.

Only claim linux/amd64 initially. Frontend multiarch may be easy, inference ARM wheels and checkpoint compatibility are not presumed. Use Buildx/QEMU only with successful platform-specific inference tests. CUDA driver/hardware remains a host requirement: Docker alone doesn't create a GPU.

## GitHub workflow behavior
Implement `validate.yml` and `release.yml` using official actions pinned to full verified commit SHAs. Do not put guessed SHAs into YAML. Untrusted PR validation has no publish credentials and no unsafe `pull_request_target` execution. Limit GITHUB_TOKEN permissions; secrets never appear in PR logs. Jobs have timeouts/concurrency controls, cache keys include lock/model hashes, artifacts include test results and sanitized screenshots.

Validation: backend/frontend checks, integration stack, real CPU model smoke, browser test, synthetic scenarios, secret/dependency scan, image build/scan and Compose config. Local QA precedes pushing implementation commits. CI for that exact commit must be green before release. Release tag/workflow_dispatch validates/tests the selected commit again or consumes verifiably bound success for exactly that SHA, builds once, creates SBOM/provenance, scans, pushes, retrieves registry digest and performs a pull-only smoke.

Docker Hub CI secrets: `DOCKERHUB_USERNAME` value diyorbek7377 and `DOCKERHUB_TOKEN` scoped for required repos. Set through GitHub UI or `gh secret set` stdin without exposing token values. GitHub CLI login does not automatically create Docker Hub secrets. Manual publication using an existing login is acceptable after the same QA gates; still record digests and smoke the pulled image.

Immutable tags: `sha-<full-or-recorded-short-commit>`; version tag when appropriate. Optional `latest` only after final acceptance. Never reuse an immutable tag for changed content. Images for the same release share one source commit and a release manifest listing model SHA and dependency/build identity.

## VPS readiness and runbook
No server has been provisioned. Recommend a CPU Ubuntu 24.04 LTS single-node starting candidate with 4 vCPU, 8 GB RAM and ~60 GB SSD for this bounded demo; this is a sizing hypothesis, not a measured guarantee or purchase recommendation. Choose actual cost/provider/region with the owner before spending. A GPU server is optional if offline CPU processing is too slow; CARLA may need a separate larger GPU instance.

Obtain host/IP, distro, CPU/RAM/disk, domain/DNS ownership, SSH user and existing local key path. Do not request private key contents. Verify SSH host fingerprint through a trusted source rather than disabling host checking. Use a dedicated deploy directory/user, protected secrets file, Docker Engine/Compose with documented versions, backups and disk limits. Enabling Docker-group access is privileged host access and should be recorded.

Production gateway: Caddy on ports 80/443 with domain DNS resolving to server and ACME connectivity. Same-origin routing `/api/*` to API, SSE with streaming/no-buffer behavior, static web otherwise. Public gateway must deny /metrics and operator UIs unless authenticated. Redis, inference, SQLite, MLflow and CARLA never have public ports. Firewall allows only SSH from intended administration sources and HTTP/HTTPS. IP-only HTTP staging is not a public HTTPS completion.

Deploy script: validate required env/domain/digests → preflight host/capacity/ports → backup current DB/artifacts → pull digests → model-init verification → compatible DB migration → start → ready/functional HTTPS smoke → atomically record successful manifest. It must refuse blanks, unknown digests or unsupported modes. Avoid `docker compose down -v`, blind prune and overwriting unrelated containers. Failed rollout restores last known-good manifest; incompatible DB migrations need explicit recovery/backup strategy.

Backup: coherent SQLite online backup (not arbitrary copying of active WAL DB), owned artifacts/metadata and persistent configurations. Redis queue snapshot alone is not a full backup. Test restoring a run/report to a separate Compose project. Store backup outside the server's failure domain when a destination is supplied. Retention and scheduling documented.

Run README commands from a clean checkout and published-image Compose. Add Bash and PowerShell/WSL copy-paste instructions. `make up`, `make down`, `make logs`, `make model-init`, `make smoke-published`, `make deploy-check`, `make backup`, `make restore-test`, `make rollback` must be implemented or not advertised. Production rollback selects recorded digests, never rebuilds source on the host.
