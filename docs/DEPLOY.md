# Production deployment runbook (Ubuntu 24.04 VPS)

**Status: no server has been provisioned.** No host, domain, DNS zone or SSH access has been supplied, and no
hosting purchase is authorized. Everything below is prepared and validated locally (`make deploy-check`,
`make smoke-published`); DreamDrive is **not deployed** until these steps run on a real, owner-selected host and
the HTTPS smoke passes there.

## What gets deployed
- `compose.prod.yaml`: registry images **by digest only** (no build sections): `diyorbek7377/dreamdrive-web`,
  `dreamdrive-backend` (api, worker, volume-init), `dreamdrive-inference` (inference, model-init), plus
  `caddy:2.10-alpine@sha256:…` and `redis:7.4-alpine@sha256:…`. CPU inference only (linux/amd64).
- Caddy (`deploy/caddy/Caddyfile.prod`) on ports 80/443 with automatic HTTPS for `DOMAIN`: `/api/*` and
  `/health/*` to the API (SSE unbuffered), everything else to the web container, `/metrics` → 404, 110 MB body
  limit, HSTS and security headers. Redis, inference, SQLite, MLflow and Prometheus have no public ports.
- Named volumes: `app-data` (SQLite + artifacts), `model-cache` (verified weights), `redis-data`, `caddy-data`
  (certificates), `caddy-config`.

## Sizing hypothesis (not a measured guarantee)
Single node, CPU only: **4 vCPU, 8 GB RAM, 60 GB SSD**, Ubuntu 24.04 LTS, x86_64. Basis: container limits in
`compose.prod.yaml` (inference 3 GB, worker 2 GB, api 1 GB, small services < 1 GB), images ~3–4 GB, model 156 MB,
24 h artifact retention at ≤ 10 queued 100 MB uploads, Docker logs capped at 20 MB × 5 per service, backups.
CPU inference measured ~0.4–0.6 s/frame on the developer laptop (docs/ENVIRONMENT.md), so a 120 s clip at 5
analyzed FPS takes several minutes; that is expected. Choose provider/region/price with the owner first.

## Information needed from the owner
| Item | Notes |
|---|---|
| Server IP / hostname, distro | Ubuntu 24.04 LTS x86_64 recommended |
| SSH user and the **path** of an existing local key | never send private key contents |
| SSH host key fingerprint | from the provider console; do not disable host key checking |
| Domain and DNS control | A (and AAAA if IPv6) record → server |
| Off-site backup destination (optional) | e.g. an rclone remote; credentials stay on the server |
| Docker Hub publication | CI secrets `DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN`, or a local `docker login` (docs/OPERATIONS.md) |

## 1. DNS and firewall
```bash
dig +short A dreamdrive.example.org        # must print the server IP before the first deploy (ACME needs it)
sudo ufw default deny incoming
sudo ufw allow from <admin-ip>/32 to any port 22 proto tcp   # or 22/tcp if the admin IP is not fixed
sudo ufw allow 80/tcp && sudo ufw allow 443/tcp && sudo ufw allow 443/udp
sudo ufw enable
```
Docker-published ports bypass ufw rules (Docker manages iptables). That is why `compose.prod.yaml` publishes only
the gateway (80/443) and every ops UI binds to `127.0.0.1`; do not add other `ports:`.

## 2. Docker Engine and deploy user
Install Docker Engine + Compose plugin from Docker's official apt repository
(https://docs.docker.com/engine/install/ubuntu/). Requirements checked by `deploy.sh`: Engine ≥ 25, Compose ≥
2.24. Record `docker version` and `docker compose version` in the deployment notes.
```bash
sudo adduser --disabled-password --gecos "" dreamdrive
sudo usermod -aG docker dreamdrive      # docker group == root-equivalent access: record this decision
sudo install -d -o dreamdrive -g dreamdrive -m 750 /opt/dreamdrive /var/backups/dreamdrive
sudo install -d -o dreamdrive -g dreamdrive -m 700 /etc/dreamdrive
```

## 3. Checkout (no builds on the server)
```bash
sudo -iu dreamdrive
git clone https://github.com/diyorbek20037773/dream_drive /opt/dreamdrive/app
cd /opt/dreamdrive/app && git checkout <release-commit>   # compose/caddy/deploy files of the release
```

## 4. Secrets file (chmod 600, never in git)
```bash
umask 077
cat >/etc/dreamdrive/dreamdrive.env <<EOF
APP_ENV=production
DOMAIN=dreamdrive.example.org
PUBLIC_ORIGIN=https://dreamdrive.example.org
APP_SECRET=$(openssl rand -base64 48 | tr -d '\n/+=')
ADMIN_PASSWORD=
GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 24 | tr -d '\n/+=')
INFERENCE_CPUS=3
ARTIFACT_RETENTION_HOURS=24
EOF
chmod 600 /etc/dreamdrive/dreamdrive.env
```
`ADMIN_PASSWORD` stays blank (operator login disabled) unless the owner sets one. `deploy.sh` refuses group/world
readable env files, blank/short `APP_SECRET`, `APP_ENV` other than production and a `PUBLIC_ORIGIN` that is not
`https://DOMAIN`.

## 5. Release identity
The release workflow (`.github/workflows/release.yml`) publishes `sha-<commit>` images only for a commit whose
`validate` run succeeded, scans them, smoke-tests the pulled digests and uploads the artifact `release-<commit>`
containing `release.env` (digests) and `release-manifest.json`:
```bash
gh run download <run-id> -n release-<commit> -D /opt/dreamdrive/releases/<commit>
```
After a manual publication (`scripts/publish.sh`) the same files are in `artifacts/release/<commit>/`.

## 6. Deploy
```bash
cd /opt/dreamdrive/app
bash deploy/deploy.sh --env-file /etc/dreamdrive/dreamdrive.env --release /opt/dreamdrive/releases/<commit>/release.env --check
bash deploy/deploy.sh --env-file /etc/dreamdrive/dreamdrive.env --release /opt/dreamdrive/releases/<commit>/release.env
```
Order: validate (refuses blank, placeholder, malformed or unknown digests and tag-only images) → compose config
+ Caddyfile validation → host preflight (Docker/Compose versions, CPU, RAM, ≥ 15 GB free, ports 80/443) →
every digest must exist in the registry → backup of the current data (`deploy/backup.sh`) → `pull` by digest →
volume-init + model-init (download + SHA256 verify) → `up -d` (the API applies SQLite migrations at start) → wait
for health → HTTPS smoke (`deploy/smoke.sh`: health, web, `/metrics` denied, capabilities with the expected build
commit, session + CSRF, SSE) → atomic record of `deploy/state/current.json`, `current-release.env` and a history
entry. A failure after `up` re-applies the previously recorded release (never rebuilds) and exits non-zero.
`deploy.sh` never runs `down -v`, `volume rm` or `prune`. Add `--smoke-upload <clip.mp4>` for a real video job.

Then check in a browser at `https://DOMAIN` (upload a short clip, run a simulation) and keep the smoke JSON from
`deploy/state/history/<attempt>/smoke.json` as evidence.

## 7. Backups (coherent, off-site when configured)
`deploy/backup.sh` uses the sqlite3 online-backup API inside the backend image (never a raw copy of the live WAL
database), adds `artifacts/` and `evaluations/` and a manifest (table counts, integrity check, release), writes
`dreamdrive-backup-<UTC>.tar.gz` + `.sha256`, and keeps the newest 14 (`--keep`). Redis is not backed up
(transient queue; job history is in SQLite). Archives contain user uploads: keep them private and apply the
same retention expectations.
```bash
# /usr/local/bin/dreamdrive-offsite (optional, owner-supplied remote; receives the archive path as $1)
#   #!/bin/sh
#   set -e
#   rclone copy --immutable "$1" offsite:dreamdrive-backups
#   rclone copy --immutable "$1.sha256" offsite:dreamdrive-backups
# /etc/cron.d/dreamdrive-backup  (daily 03:15 UTC)
15 3 * * * dreamdrive cd /opt/dreamdrive/app && DD_BACKUP_DIR=/var/backups/dreamdrive \
  DD_BACKUP_OFFSITE_CMD=/usr/local/bin/dreamdrive-offsite bash /opt/dreamdrive/app/deploy/backup.sh \
  --env-file /etc/dreamdrive/dreamdrive.env >>/var/log/dreamdrive-backup.log 2>&1
```
`DD_BACKUP_OFFSITE_CMD` is optional and is called with the archive path as its last argument; if it is set, a
failed off-site copy fails the backup. Without it, backups stay on the server (same failure domain).

Restore test (monthly, and after every schema change) into a separate throw-away Compose project:
```bash
DD_BACKUP_DIR=/var/backups/dreamdrive bash deploy/restore-test.sh --env-file /etc/dreamdrive/dreamdrive.env \
  --require-job --boot-api
```
Disaster recovery onto a new host: deploy the same release, stop api/worker, then restore the archive into the
empty `app-data` volume with the same helper (`deploy/lib/backup_data.py restore`, see RUNBOOK "bad release").

## 8. Rollback
```bash
bash deploy/rollback.sh --env-file /etc/dreamdrive/dreamdrive.env --list
bash deploy/rollback.sh --env-file /etc/dreamdrive/dreamdrive.env            # newest earlier successful release
bash deploy/rollback.sh --env-file /etc/dreamdrive/dreamdrive.env --to <commit>
```
Rollback uses only recorded digests (`deploy/state/releases/<commit>.env`) and goes through the full deploy
path (backup, health, smoke). SQLite migrations are additive; if a release ever needs an incompatible migration,
restore the pre-deploy backup taken by `deploy.sh` instead of rolling back the image alone.

## 9. Operations on the server
```bash
bash deploy/compose.sh --env-file /etc/dreamdrive/dreamdrive.env ps
bash deploy/compose.sh --env-file /etc/dreamdrive/dreamdrive.env logs --tail 200 worker
bash deploy/compose.sh --env-file /etc/dreamdrive/dreamdrive.env --with-ops up -d prometheus grafana mlflow
ssh -L 3000:127.0.0.1:3000 -L 9090:127.0.0.1:9090 dreamdrive@<server>     # Grafana/Prometheus via SSH tunnel only
```
Incidents: docs/RUNBOOK.md.

## Local rehearsal of this exact path
`make smoke-published RELEASE_ENV=artifacts/release/<commit>/release.env UPLOAD=<clip.mp4>` runs `deploy.sh
--local` in a fresh Compose project with `DOMAIN=localhost` (Caddy internal CA), pull-only digests and fresh
volumes, then removes that project.

## Remaining blockers (as of 2026-10-10)
1. No server/host, SSH user/key path or domain has been supplied; nothing is deployed.
2. Docker Hub publication needs either the GitHub secrets `DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN`
   (`gh secret set DOCKERHUB_TOKEN` reads the token from stdin) or a local `docker login` by the owner.
3. HTTPS/ACME can only be verified on the real host with DNS pointing at it.
