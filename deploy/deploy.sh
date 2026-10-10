#!/usr/bin/env bash
# DreamDrive production deploy (pull-only, digests only). Steps:
#   1 validate env + release files (blank/placeholder/non-digest values are refused)
#   2 compose config + Caddyfile.prod validation           (--check stops here)
#   3 host preflight: Docker/Compose versions, CPU, RAM, disk, ports 80/443
#   4 verify every image digest exists in the registry
#   5 backup current data (deploy/backup.sh)                (skipped only when no database exists yet)
#   6 docker compose pull (by digest)  7 volume-init + model-init (download + SHA256 verify)
#   8 up -d (the api applies SQLite migrations at start)  9 wait for health  10 HTTPS smoke (deploy/smoke.sh)
#   11 atomically record deploy/state/current.json + current-release.env and a history entry
# A failure after step 8 re-applies the previously recorded release (never rebuilds) and exits non-zero.
# Never runs `down -v`, `volume rm` or `prune`.
#
#   deploy/deploy.sh --env-file /etc/dreamdrive/dreamdrive.env --release release.env
#   deploy/deploy.sh --env-file ENV --release REL --check [--placeholder-ok]   # validation only (make deploy-check)
#
# Options: --project NAME (default dreamdrive)  --with-ops (also compose.ops.yaml)  --smoke-upload FILE
#          --local (DOMAIN=localhost, internal CA, used by make smoke-published)  --kind deploy|rollback
set -Eeuo pipefail
DD_LOG_TAG=deploy
# shellcheck source=deploy/lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

usage() { sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

env_file="" release_file="" check=0 placeholder_ok=0 local_mode=0 with_ops=0 kind=deploy
project="${DD_PROJECT:-dreamdrive}" wait_s="${DD_DEPLOY_WAIT_SECONDS:-480}" smoke_upload=""
while (($#)); do
  case $1 in
    --env-file) env_file=$2; shift 2 ;;
    --release) release_file=$2; shift 2 ;;
    --check) check=1; shift ;;
    --placeholder-ok) placeholder_ok=1; shift ;;
    --local) local_mode=1; shift ;;
    --with-ops) with_ops=1; shift ;;
    --project) project=$2; shift 2 ;;
    --smoke-upload) smoke_upload=$2; shift 2 ;;
    --kind) kind=$2; shift 2 ;;
    -h | --help) usage; exit 0 ;;
    *) die "unknown argument: $1" 64 ;;
  esac
done
[[ -n $env_file ]] || die "--env-file is required (see --help)" 64
((placeholder_ok && !check)) && die "--placeholder-ok is only allowed together with --check" 64
[[ $kind == deploy || $kind == rollback ]] || die "--kind must be deploy or rollback" 64
[[ $project =~ ^[a-z0-9][a-z0-9_-]*$ ]] || die "invalid --project name" 64
[[ -z $smoke_upload || -f $smoke_upload ]] || die "--smoke-upload file not found: $smoke_upload" 64

# ---------------------------------------------------------------------------------------------------------------
# 1. validation
# ---------------------------------------------------------------------------------------------------------------
clear_release_env
load_env_file "$env_file"
[[ -n $release_file ]] && load_env_file "$release_file"
((placeholder_ok)) || check_secret_file_mode "$env_file"

errors=()
invalid() { errors+=("$1"); }
https_port=${DD_PUBLIC_HTTPS_PORT:-443}
if ((local_mode)); then
  [[ ${DOMAIN:-} == localhost ]] || invalid "--local requires DOMAIN=localhost"
  expected_origin="https://localhost"
  [[ $https_port == 443 ]] || expected_origin="https://localhost:$https_port"
  base_url=$expected_origin
else
  if ! is_hostname "${DOMAIN:-}"; then
    invalid "DOMAIN must be a lowercase DNS name such as app.example.org (blank or invalid)"
  elif [[ ${DOMAIN} =~ (^|\.)(example\.(com|org|net)|invalid|test|local|localhost)$ ]] && ((!placeholder_ok)); then
    invalid "DOMAIN '${DOMAIN}' is a reserved/placeholder name"
  fi
  expected_origin="https://${DOMAIN:-}"
  base_url=$expected_origin
fi
[[ ${APP_ENV:-} == production ]] || invalid "APP_ENV must be production"
[[ ${PUBLIC_ORIGIN:-} == "$expected_origin" ]] || invalid "PUBLIC_ORIGIN must equal $expected_origin"
secret=${APP_SECRET:-}
((${#secret} >= 32)) || invalid "APP_SECRET must be set (>= 32 characters; value not shown)"
unset secret
for key in WEB_IMAGE_DIGEST BACKEND_IMAGE_DIGEST INFERENCE_IMAGE_DIGEST; do
  value=${!key:-}
  if ! is_sha256_digest "$value"; then
    invalid "$key must be sha256:<64 lowercase hex> (got ${value:-blank})"
  elif is_placeholder_digest "$value" && ((!placeholder_ok)); then
    invalid "$key is the all-zero placeholder, not a published digest"
  fi
done
for key in GATEWAY_IMAGE REDIS_IMAGE; do
  value=${!key:-}
  if ! is_pinned_image "$value"; then
    invalid "$key must be pinned as name:tag@sha256:<64 hex> (got ${value:-blank})"
  elif [[ $value == *"@$DD_ZERO_DIGEST" ]] && ((!placeholder_ok)); then
    invalid "$key uses the all-zero placeholder digest"
  fi
done
is_git_commit "${GIT_COMMIT:-}" || invalid "GIT_COMMIT (release source commit) must be 40 lowercase hex"
namespace=${DOCKERHUB_NAMESPACE:-diyorbek7377}
[[ $namespace =~ ^[a-z0-9]+([._-][a-z0-9]+)*$ ]] || invalid "DOCKERHUB_NAMESPACE is invalid"
if ((${#errors[@]})); then
  for e in "${errors[@]}"; do log "invalid: $e"; done
  die "refusing to deploy: ${#errors[@]} validation error(s)" 2
fi
log "validated release ${GIT_COMMIT} for ${DOMAIN} (project $project, kind $kind)"

extra_files=()
((with_ops)) && extra_files+=("$DD_REPO_ROOT/compose.ops.yaml")
dd_compose_setup prod "$project" "$env_file" "$release_file" "${extra_files[@]}"

# ---------------------------------------------------------------------------------------------------------------
# 2. static checks
# ---------------------------------------------------------------------------------------------------------------
"${DD_COMPOSE[@]}" config -q || die "docker compose config failed"
caddy_image=$GATEWAY_IMAGE
if [[ $caddy_image == *"@$DD_ZERO_DIGEST" ]]; then
  caddy_image=${caddy_image%@*} # placeholder digest cannot be pulled; validate with the same tag
  warn "validating Caddyfile.prod with tag $caddy_image (placeholder digest)"
fi
docker run --rm --network none -e "DOMAIN=$DOMAIN" -v "$DD_REPO_HOST/deploy/caddy:/etc/caddy:ro" "$caddy_image" \
  caddy validate --config /etc/caddy/Caddyfile.prod --adapter caddyfile >/dev/null 2>&1 ||
  die "Caddyfile.prod failed validation with $caddy_image"
log "compose config and Caddyfile.prod are valid"
if ((check)); then
  log "check passed (no changes made)"
  exit 0
fi

# ---------------------------------------------------------------------------------------------------------------
# 3. host preflight
# ---------------------------------------------------------------------------------------------------------------
version_ge() { # version_ge HAVE WANT  (numeric dotted compare)
  local IFS=.
  local -a h w
  read -r -a h <<<"${1%%[-+]*}"
  read -r -a w <<<"$2"
  local i
  for i in 0 1 2; do
    ((${h[i]:-0} > ${w[i]:-0})) && return 0
    ((${h[i]:-0} < ${w[i]:-0})) && return 1
  done
  return 0
}
preflight() {
  docker info >/dev/null 2>&1 || die "Docker daemon is not reachable"
  local engine compose
  engine=$(docker version --format '{{.Server.Version}}')
  compose=$(docker compose version --short)
  version_ge "$engine" 25.0.0 || die "Docker Engine $engine < 25.0 (healthcheck start_interval needs 25+)"
  version_ge "${compose#v}" 2.24.0 || die "Docker Compose $compose < 2.24"
  log "docker engine $engine, compose $compose"
  if [[ $(uname -s) != Linux ]]; then
    warn "host capacity/port preflight only runs on Linux servers (this is $(uname -s))"
    return 0
  fi
  local cpus mem_kb root avail_kb port listeners ours
  cpus=$(nproc)
  mem_kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
  root=$(docker info --format '{{.DockerRootDir}}')
  avail_kb=$(df -Pk "$root" | awk 'NR == 2 {print $4}')
  log "host: ${cpus} CPU, $((mem_kb / 1024)) MiB RAM, $((avail_kb / 1024 / 1024)) GiB free on $root"
  ((cpus >= 2)) || die "at least 2 CPUs are required"
  ((mem_kb >= 3500000)) || die "at least ~3.5 GB RAM is required"
  ((avail_kb >= 15 * 1024 * 1024)) || die "at least 15 GB free disk is required on $root"
  ((cpus >= 4)) || warn "fewer than 4 CPUs: CPU video analysis will be slow"
  ((mem_kb >= 7500000)) || warn "less than 8 GB RAM: inference (3 GB) + worker (2 GB) limits may contend"
  if command -v ss >/dev/null 2>&1; then
    ours=$(docker ps -q --filter "label=com.docker.compose.project=$project" \
      --filter "label=com.docker.compose.service=gateway")
    for port in "${DD_PUBLIC_HTTP_PORT:-80}" "$https_port"; do
      listeners=$(ss -Hltn "sport = :$port" | wc -l)
      if ((listeners > 0)) && [[ -z $ours ]]; then
        die "port $port is already in use by another process (not this project's gateway)"
      fi
    done
  else
    warn "ss not found; port check skipped"
  fi
}
preflight

# ---------------------------------------------------------------------------------------------------------------
# 4. registry: every digest must exist (unknown digests are refused before anything changes)
# ---------------------------------------------------------------------------------------------------------------
images=(
  "$namespace/dreamdrive-web@$WEB_IMAGE_DIGEST"
  "$namespace/dreamdrive-backend@$BACKEND_IMAGE_DIGEST"
  "$namespace/dreamdrive-inference@$INFERENCE_IMAGE_DIGEST"
  "$GATEWAY_IMAGE"
  "$REDIS_IMAGE"
)
for ref in "${images[@]}"; do
  docker manifest inspect "$ref" >/dev/null 2>&1 || die "image digest not found in registry (or no access): $ref"
done
log "all ${#images[@]} image digests exist in the registry"

# ---------------------------------------------------------------------------------------------------------------
# 5. backup
# ---------------------------------------------------------------------------------------------------------------
backup_path=""
if docker volume inspect "${project}_app-data" >/dev/null 2>&1; then
  backup_release=$release_file
  [[ -f $DD_STATE_DIR/current-release.env ]] && backup_release="$DD_STATE_DIR/current-release.env"
  status=0
  backup_path=$(bash "$DD_REPO_ROOT/deploy/backup.sh" --env-file "$env_file" --release "$backup_release" \
    --project "$project") || status=$?
  if ((status == 3)); then
    log "no existing database in ${project}_app-data: nothing to back up"
    backup_path=""
  elif ((status != 0)); then
    die "pre-deploy backup failed (exit $status); nothing was changed"
  fi
else
  log "first deployment of project $project: no data volume to back up"
fi

# ---------------------------------------------------------------------------------------------------------------
# 6-7. pull + init (running containers are not touched yet)
# ---------------------------------------------------------------------------------------------------------------
"${DD_COMPOSE[@]}" pull --quiet || die "docker compose pull failed; nothing was changed"
"${DD_COMPOSE[@]}" run --rm model-init || die "model-init failed (download or SHA256 verification); nothing was changed"

# ---------------------------------------------------------------------------------------------------------------
# 8-11. start, verify, record
# ---------------------------------------------------------------------------------------------------------------
stamp=$(utc_stamp)
attempt_id="$stamp-${GIT_COMMIT:0:12}"
mkdir -p "$DD_STATE_DIR/history/$attempt_id" "$DD_STATE_DIR/releases"
services=(redis api worker inference web gateway)
((with_ops)) && services+=(prometheus grafana mlflow)

write_release_env() { # only release identity, never secrets
  local out=$1 k
  {
    printf '# DreamDrive release identity recorded by deploy.sh at %s\n' "$(utc_now)"
    printf 'DOCKERHUB_NAMESPACE=%s\n' "$namespace"
    for k in "${DD_RELEASE_KEYS[@]}"; do printf '%s=%s\n' "$k" "${!k}"; done
  } >"$out"
}
write_entry() { # write_entry STATUS FILE
  local status=$1 out=$2 backup_json=null
  [[ -z $backup_path ]] || backup_json="\"$backup_path\""
  cat >"$out" <<JSON
{
  "schema": "dreamdrive-deploy/1",
  "status": "$status",
  "kind": "$kind",
  "attempt_id": "$attempt_id",
  "recorded_utc": "$(utc_now)",
  "project": "$project",
  "domain": "$DOMAIN",
  "git_commit": "$GIT_COMMIT",
  "images": {
    "web": "${images[0]}",
    "backend": "${images[1]}",
    "inference": "${images[2]}",
    "gateway": "${images[3]}",
    "redis": "${images[4]}"
  },
  "release_env": "releases/$GIT_COMMIT.env",
  "compose_prod_sha256": "$(file_sha256 "$DD_REPO_ROOT/compose.prod.yaml")",
  "caddyfile_prod_sha256": "$(file_sha256 "$DD_REPO_ROOT/deploy/caddy/Caddyfile.prod")",
  "smoke": "history/$attempt_id/smoke.json",
  "backup": $backup_json
}
JSON
}

rollback_previous() {
  local reason=$1
  write_entry failed "$DD_STATE_DIR/history/$attempt_id.json"
  log "deploy failed: $reason"
  "${DD_COMPOSE[@]}" ps >&2 || warn "could not list services"
  local prev="$DD_STATE_DIR/current-release.env"
  if [[ ! -f $prev ]]; then
    die "no previously recorded release to restore; stack left as is (inspect: deploy/compose.sh ps/logs)"
  fi
  if grep -qx "GIT_COMMIT=$GIT_COMMIT" "$prev" && grep -qx "BACKEND_IMAGE_DIGEST=$BACKEND_IMAGE_DIGEST" "$prev"; then
    die "the failing release is the recorded current release; nothing older to restore automatically"
  fi
  log "re-applying the last recorded release ($(grep '^GIT_COMMIT=' "$prev"))"
  local -a prev_compose
  dd_compose_setup prod "$project" "$env_file" "$prev" "${extra_files[@]}"
  prev_compose=("${DD_COMPOSE[@]}")
  if "${prev_compose[@]}" up -d && wait_healthy "$project" "$wait_s" "${services[@]}"; then
    log "previous release restored and healthy"
  else
    log "previous release did NOT become healthy; manual recovery needed (docs/RUNBOOK.md#bad-release)"
  fi
  die "deploy of $GIT_COMMIT failed and was rolled back: $reason"
}

log "starting release $GIT_COMMIT"
"${DD_COMPOSE[@]}" up -d || rollback_previous "docker compose up failed"
wait_healthy "$project" "$wait_s" "${services[@]}" || rollback_previous "services not healthy within ${wait_s}s"

smoke_args=("$base_url" --expect-commit "$GIT_COMMIT" --evidence "$DD_STATE_DIR/history/$attempt_id")
((local_mode)) && smoke_args+=(--insecure)
[[ -n $smoke_upload ]] && smoke_args+=(--upload "$smoke_upload")
bash "$DD_REPO_ROOT/deploy/smoke.sh" "${smoke_args[@]}" >"$DD_STATE_DIR/history/$attempt_id/smoke.out" ||
  rollback_previous "HTTPS smoke test failed (see $DD_STATE_DIR/history/$attempt_id/smoke.json)"

# Atomic record: write temp files in the same directory, then rename over the current pointers.
write_release_env "$DD_STATE_DIR/releases/$GIT_COMMIT.env"
write_entry success "$DD_STATE_DIR/history/$attempt_id.json"
tmp_json=$(mktemp "$DD_STATE_DIR/.current.json.XXXXXX")
tmp_env=$(mktemp "$DD_STATE_DIR/.current-release.env.XXXXXX")
cp "$DD_STATE_DIR/history/$attempt_id.json" "$tmp_json"
cp "$DD_STATE_DIR/releases/$GIT_COMMIT.env" "$tmp_env"
mv -f "$tmp_env" "$DD_STATE_DIR/current-release.env"
mv -f "$tmp_json" "$DD_STATE_DIR/current.json"
log "deployed $GIT_COMMIT to $base_url; recorded $DD_STATE_DIR/current.json"
