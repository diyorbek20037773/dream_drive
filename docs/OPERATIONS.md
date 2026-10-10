# Operations guide (local stack, QA, monitoring, release)

Docker Engine ≥ 25 with Compose ≥ 2.24 is the only runtime requirement: every Python/Node gate runs in a
container. Make and the scripts are bash (GNU make 4.x). Production deployment: docs/DEPLOY.md. Incidents:
docs/RUNBOOK.md.

## Windows (PowerShell users)
Run `make` from **Git Bash** (installed with Git for Windows) or from **WSL2 Ubuntu**, not from PowerShell:
```powershell
# PowerShell: open Git Bash in the repository and run a target
& "C:\Program Files\Git\bin\bash.exe" -lc "cd /d/myprojects/west_hack_dream_drive && make up"
# or via WSL (Docker Desktop WSL integration enabled for the distro)
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/d/myprojects/west_hack_dream_drive && make up"
```
Plain Docker Compose works from PowerShell too (no make needed):
```powershell
docker compose up -d --build            # CPU stack, http://127.0.0.1:8080
docker compose ps
docker compose logs -f api
docker compose down                     # keeps volumes
```
The Makefile exports `MSYS_NO_PATHCONV=1` so Git Bash does not rewrite container paths (`/evidence`, `/data`);
set it yourself when running the scripts' `docker` commands by hand in Git Bash.

## Local stack
| Command | What it does |
|---|---|
| `make doctor` | Docker/Compose/git/repo prerequisites (PASS/WARN/FAIL; non-zero on FAIL) |
| `make init-env` | `.env` from `.env.example` with generated `APP_SECRET`, `GRAFANA_ADMIN_PASSWORD` (never printed) |
| `make up` | build + start (volume-init → model-init → services), wait for health + gateway → api readiness |
| `make up OPS=1` / `GPU=1` | add Prometheus/Grafana/MLflow on 127.0.0.1 / CUDA inference override |
| `make ps`, `make logs SERVICE=worker`, `make down` | status, logs, stop (volumes are kept) |
| `make model-init` | download + SHA256-verify the checkpoint into the `model-cache` volume |
| `make smoke [UPLOAD=clip.mp4]` | gateway smoke; with UPLOAD a real end-to-end video job |
| `make backup`, `make restore-test [BACKUP=…]`, `make backup-test` | data backup and isolated restore verification |

Only the gateway publishes a port (`127.0.0.1:${DD_HTTP_PORT:-8080}`). Containers run as non-root uid 10001
(redis 999, web 101) with read-only root filesystems, dropped capabilities, `no-new-privileges`, tmpfs for `/tmp`,
CPU/memory/pid limits sized for a ~7.6 GiB Docker Desktop VM, and json-file log rotation. `volume-init` (one-shot,
root, only CHOWN/FOWNER/DAC_OVERRIDE, no network) hands the named volumes to uid 10001. The API does not wait for
inference: without a verified model it stays up and reports video analysis as unavailable with a reason.

## QA gates and evidence
`make qa` (scripts/qa.sh) runs, in order: doctor, deploy-check, ops-check, ops-selftest, build, lint,
typecheck, gen-types, unit, fixtures (licensed clip via scripts/fetch_fixtures.py), model-init, model-smoke,
stack-up, integration, scenarios (incl. the live learned-sim validation against the inference service),
smoke-upload (real video through the gateway), e2e (Playwright image matching `frontend/package-lock.json`),
eval-scenarios, security (tests/security, pip-audit, npm audit, gitleaks, trivy), backup-test, sbom. Gates with a failed
prerequisite are `blocked`; any failed, blocked, not-run gate or any junit `skipped` > 0 fails the run.
`DD_REQUIRE_LIVE=1` is exported so live tests fail instead of skipping. GPU/CARLA tests are deselected with
`-m "not gpu and not carla"` and are therefore not claimed.

Evidence: `artifacts/qa/<commit>/` (git-ignored): `NN-<gate>.log`, `<gate>/` (junit XML, Playwright report and
screenshots, model-smoke JSON, smoke JSON, evaluations, SBOMs, restore result), `gates.tsv`, `summary.md`,
`summary.json` (`release_evidence: false` for `QA_ALLOW_DIRTY=1` runs), `environment.txt`, `stack-logs.txt`,
`local-image-ids.txt`. Release QA must run on a clean, committed tree; commit fixes and rerun.

Individual gates: `make lint` (`lint-python`, `lint-frontend`, `lint-shell`), `make typecheck`, `make test`,
`make scenarios`, `make model-smoke`, `make integration`, `make e2e`, `make security`, `make eval-scenarios
SEEDS=1,2,3`, `make gen-types` / `make gen-types-check`, `make ops-check`, `make deploy-check`, `make sbom`.
Pass `EVIDENCE_DIR=./artifacts/...` to choose where junit/report files go (must start with `./` or `/`).

## Monitoring (ops profile)
`make up OPS=1` (needs `GRAFANA_ADMIN_PASSWORD` in `.env`; `make init-env` generates it):
- Prometheus http://127.0.0.1:9090 scrapes `api:8000/metrics`, `worker:9101/metrics`, `inference:8001/metrics`
  on the internal network; rules in `ops/prometheus/alerts.yml` (service down > 2 min, model not ready, worker
  heartbeat stale, oldest job above 15 min, failed jobs, data volume low + a guard when that metric is missing,
  artifact storage high). `make ops-check` runs `promtool check config/rules` and the unit tests in
  `alerts_test.yml` that prove inactive → pending → firing transitions. No alert destination is configured.
- Grafana http://127.0.0.1:3000 (user `admin`): provisioned datasource + dashboards (service health, jobs and
  queue, inference latency/throughput, simulation) generated by `scripts/gen_grafana_dashboards.py`; edit the
  generator, not the JSON, and `make ops-check` fails on drift.
- MLflow http://127.0.0.1:5000 (SQLite store + artifact volume). `make mlflow-log` pushes the application's
  evaluation JSON (`DATA_ROOT/evaluations`) with the resolved model manifest; only `measured` metrics become
  MLflow metrics, everything else is a status tag. With `ENABLE_MLFLOW=false` a tracking outage is a warning
  (exit 0); with `true` it fails (exit 2).
Metric/label names follow docs/OBSERVABILITY.md; panels for metrics a service does not export show "no data".

## CI and release
- `.github/workflows/validate.yml` (push to main, PRs): `static` job (compose combinations + deploy guards,
  promtool, dashboard drift, self-tests, shellcheck/actionlint, gitleaks) and `qa` job running `scripts/qa.sh`
  with the model cached by checkpoint SHA256 (model-init re-verifies; downloads on a miss) and the fixture clip
  cached by manifest hash; QA evidence (junit, Playwright screenshots, SBOMs) is uploaded. No secrets.
- `.github/workflows/release.yml` (workflow_dispatch with `commit`, or a `vX.Y.Z` tag): verifies via the GitHub
  API that `validate` succeeded for exactly that commit and that it is in `main`; builds each image once with
  buildx (linux/amd64, SBOM + max provenance), pushes immutable `sha-<commit>` tags (refuses to overwrite),
  writes `release.env` + `release-manifest.json` from registry digests, scans the pushed digests with Trivy and
  runs the pull-only smoke. Needs repository secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`:
  `gh secret set DOCKERHUB_USERNAME --body diyorbek7377` and `gh secret set DOCKERHUB_TOKEN` (paste at the prompt).
- Manual publication with an existing `docker login`: after `make qa` passed on a clean HEAD,
  `bash scripts/publish.sh [--version vX.Y.Z]` (same refusals: clean tree, passing release QA evidence, no
  existing sha- tag), then `make smoke-published RELEASE_ENV=artifacts/release/<commit>/release.env
  UPLOAD=tests/fixtures/cache/commons_dashcam_daylight_cc0_96s_10s.mp4`. If buildx reports that the `docker`
  driver cannot export attestations, create a builder first: `docker buildx create --use --name dreamdrive`.
- Only linux/amd64 is built and claimed. The optional `dreamdrive-inference-gpu` image is published only after
  its own recorded GPU smoke.

All third-party GitHub Actions are pinned to commit SHAs resolved with `gh api` (version in a comment); tool
images used by scripts are digest-pinned in `scripts/tool-images.env` (resolved with
`docker buildx imagetools inspect`). Update both by resolving again, never by hand-typing digests.

## Troubleshooting
- `make up` waits up to 10 minutes; the first start downloads the 156 MB checkpoint and warms up torch. On
  timeout the script prints unhealthy services and their last log lines.
- Port 8080 busy: set `DD_HTTP_PORT=8081` and `PUBLIC_ORIGIN=http://localhost:8081` in `.env`.
- `GRAFANA_ADMIN_PASSWORD is not set`: run `make init-env` (fills only blank values).
- Bind-mount errors mentioning `C:/Program Files/Git/...`: a docker command ran in Git Bash without
  `MSYS_NO_PATHCONV=1`.
- Evidence files owned by uid 10001 on Linux: the evidence directories are created world-writable by the
  Makefile so containers can write junit files; delete them normally from the host.
