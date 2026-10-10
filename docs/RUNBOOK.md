# Incident runbook

Applies to the local stack (`docker compose …` / `make`) and production (`deploy/compose.sh --env-file
/etc/dreamdrive/dreamdrive.env …`). Below, `dc` means whichever of the two you use. Alerts in
`ops/prometheus/alerts.yml` link to these sections. Before any recovery: **take a backup**
(`deploy/backup.sh`), never run `down -v`, `volume rm` or `docker system prune` on the live project, and keep
the evidence listed per incident (redact cookies, tokens and uploaded media before sharing).

Useful views: `dc ps`, `dc logs --tail 200 <service>` (JSON logs with request/job/run ids and error codes),
`GET /api/v1/capabilities` (model/device/queue state), Prometheus `/alerts` and Grafana "DreamDrive / Service
health" (ops profile, via SSH tunnel in production).

## Model not ready
- **Symptoms:** alert `DreamDriveModelNotReady`; capabilities `video_analysis.available=false` with a reason;
  uploads return 503 `model_not_ready`; inference container `unhealthy`. Simulation (synthetic_rgb_cv) keeps
  working because it does not use the learned model.
- **Diagnostics:** `dc logs inference` and `dc run --rm model-init` output; reason codes: `model_missing`
  (model-init never completed or the volume is new), `hash_mismatch` (corrupt/partial or wrong file),
  `load_failed` (torch/runtime incompatibility), `device_unavailable` (CUDA requested but absent).
  `docker run --rm -v dreamdrive_model-cache:/m:ro python:3.11-slim-bookworm sha256sum /m/*/*` vs
  `config/models.resolved.yaml`.
- **Recovery:** `dc run --rm model-init` (re-downloads, verifies SHA256, atomic rename; exits 0 when already
  valid) then `dc restart inference`. Never copy an unverified file into the volume and never enable mock
  inference. For `device_unavailable`, start without `compose.gpu.yaml` (CPU default).
- **Evidence:** inference + model-init logs, capabilities JSON, the volume's sha256 listing, image digest.

## GPU out of memory
- **Symptoms (GPU profile only):** inference logs `CUDA out of memory`; jobs fail with `inference_failed`;
  `inference_busy` responses rise; container may restart.
- **Diagnostics:** `nvidia-smi` on the host (other processes holding VRAM?), `dc logs inference`, the
  inference request outcome panel; confirm `INFERENCE_PRECISION` (fp16 expected on CUDA).
- **Recovery:** stop other GPU consumers; `dc restart inference`; if it recurs, run the CPU default (drop
  `compose.gpu.yaml`) — results are labeled with the CPU device, never silently mixed. Failed jobs stay failed;
  users re-upload. GPU support is only claimed with its own recorded smoke.
- **Evidence:** `nvidia-smi` output, inference logs around the error, affected job ids (from logs, not metrics).

## Corrupt video or failing jobs
- **Symptoms:** alert `DreamDriveJobFailuresElevated`; jobs end `failed` with `video_invalid`,
  `video_decode_failed`, `video_no_stream`, `video_too_long`, `video_too_large_dimensions` or
  `inference_failed`.
- **Diagnostics:** `dc logs worker | grep <job_id>`; the job's `error_code`/`error_message` via
  `GET /api/v1/jobs/{id}` (owner session) or the operator listing; check if failures share one client/session
  (abuse) or one error code (regression after a release → see Bad release).
- **Recovery:** invalid inputs are expected rejections; no action beyond telling the user the limits
  (MP4/MOV, ≤ 120 s, ≤ 1920×1080). For `inference_failed`, check Model not ready / GPU OOM. Abuse: lower
  `MAX_REQUESTS_PER_MINUTE` in the env file and redeploy the same release.
- **Evidence:** error codes and counts, worker log excerpt, release commit. Do not keep user uploads longer than
  the retention window.

## Queue backlog
- **Symptoms:** alert `DreamDriveQueueOldestJobAboveSLA`; capabilities queue counts high; uploads get 429
  `queue_full`/`session_queue_full`.
- **Diagnostics:** Grafana "Jobs and queue" (depth, oldest age, frames/s); `dc logs worker` for a stuck frame
  loop; inference latency panel (CPU saturation?); `docker stats`.
- **Recovery:** if the worker is wedged on one job, cancel it (`POST /api/v1/jobs/{id}/cancel`, or operator
  cleanup), then `dc restart worker` — startup reconciliation marks abandoned running jobs `interrupted`
  (never silently retried). If the load is genuine, wait (CPU processing is slower than playback) or move to a
  larger host / GPU profile after its smoke. Do not raise `MAX_QUEUED_TOTAL` beyond what the disk/CPU sustain.
- **Evidence:** queue depth/age graphs, worker logs, job ids and states before/after.

## Disk full
- **Symptoms:** alerts `DreamDriveDataVolumeLow` / `DreamDriveArtifactStorageHigh`; uploads fail; SQLite
  `database or disk is full`; Redis AOF write errors.
- **Diagnostics:** `df -h $(docker info -f '{{.DockerRootDir}}')`, `docker system df`,
  `docker run --rm -v dreamdrive_app-data:/d:ro python:3.11-slim-bookworm du -sh /d/*`, backup directory size,
  container log sizes (capped by the json-file driver settings).
- **Recovery:** run retention now (`POST /api/v1/admin/cleanup` as operator); lower
  `ARTIFACT_RETENTION_HOURS` and redeploy; move old backups off-site (`deploy/backup.sh --keep`); remove only
  **unused old images** you can identify (`docker image ls`, `docker image rm <old digest>`), never volumes. If
  `DreamDriveDiskMetricMissing` fires, the free-space metric is not exported yet; use `df` manually.
- **Evidence:** `df`/`du` output before and after, what was deleted and why.

## Worker crash or service down
- **Symptoms:** alerts `DreamDriveServiceDown` (api/worker/inference scrape down > 2 min) or
  `DreamDriveWorkerHeartbeatStale`; jobs stay queued; container restarts in `dc ps`.
- **Diagnostics:** `dc ps` (restart count, health), `dc logs --tail 300 <service>`, `docker inspect
  <container> --format '{{.State.OOMKilled}}'` (memory limit hit?), `dmesg | tail` on the host.
- **Recovery:** `dc up -d <service>` (restart policy is `unless-stopped`); after a worker crash, running jobs
  are reconciled to `interrupted` on startup and users re-submit. Repeated OOM: reduce `ANALYSIS_FPS` or the
  maximum video size, or give the VM more memory; do not remove limits. If it started with a new release, see
  Bad release.
- **Evidence:** logs, OOM flag, restart counts, release commit, alert timeline.

## Secret compromise
- **Symptoms:** a secret (APP_SECRET, ADMIN_PASSWORD, GRAFANA_ADMIN_PASSWORD, Docker Hub token, SSH key)
  appeared in a log, chat, commit, image or screenshot, or suspicious operator activity.
- **Diagnostics:** `bash scripts/security.sh gitleaks` (history), `docker history --no-trunc <image>`, search
  logs for the value **without pasting it anywhere new**; review `dc logs api` for operator logins.
- **Recovery:** rotate immediately: generate a new `APP_SECRET` (invalidates all sessions/CSRF tokens) and
  `ADMIN_PASSWORD` in `/etc/dreamdrive/dreamdrive.env` (chmod 600) and redeploy the same release; revoke and
  re-issue the Docker Hub token (`gh secret set DOCKERHUB_TOKEN` reads stdin) and GitHub tokens; replace SSH
  keys. If a secret was committed, rotate first; history rewriting needs the owner's explicit decision (no
  force-push by default). Images never contain secrets by design; if one did, delete that tag after publishing a
  fixed release and record it.
- **Evidence:** what leaked (name, not value), where, exposure window, rotation timestamps.

## Bad release
- **Symptoms:** smoke failure during `deploy.sh` (it already re-applied the previous release automatically), or
  regressions noticed after a successful deploy (new error codes, failing jobs, UI errors).
- **Diagnostics:** `deploy/rollback.sh --env-file … --list`; `deploy/state/history/<attempt>/smoke.json`;
  compare `capabilities.build.commit` with `deploy/state/current.json`; CI/QA evidence for that commit.
- **Recovery:** `deploy/rollback.sh --env-file /etc/dreamdrive/dreamdrive.env [--to <commit>]` — recorded
  digests only, never a rebuild, full backup + health + smoke. If the bad release changed data incompatibly,
  restore the pre-deploy backup that `deploy.sh` took (path in the history entry) after rolling back the images:
  ```bash
  dc stop api worker
  dc run --rm --no-deps -T --entrypoint python api -c "$(cat deploy/lib/backup_data.py)" restore --force \
    < /var/backups/dreamdrive/dreamdrive-backup-<UTC>.tar.gz
  dc up -d
  ```
  Then fix forward with a new commit through validate → release; never reuse a `sha-` tag.
- **Evidence:** both commits and image digests, smoke JSON, history entries, backup used, user-visible impact.
