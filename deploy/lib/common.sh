# shellcheck shell=bash
# Shared helpers for deploy/*.sh. Sourced, never executed. Bash 4+, works on Linux and Git Bash (Windows).
# Secrets are never printed: env files are parsed without `source`/eval and error messages name keys only.

# Git Bash would otherwise rewrite "/data"-style arguments passed to docker into Windows paths.
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'

DD_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Path form the Docker CLI/daemon understands for bind mounts: D:/... on Git Bash, unchanged on Linux.
DD_REPO_HOST="$(cd "$DD_REPO_ROOT" && { pwd -W 2>/dev/null || pwd; })"
DD_STATE_DIR="${DD_STATE_DIR:-$DD_REPO_ROOT/deploy/state}"
DD_BACKUP_DIR="${DD_BACKUP_DIR:-$DD_REPO_ROOT/backups}"
DD_ZERO_DIGEST="sha256:0000000000000000000000000000000000000000000000000000000000000000"
# Keys that identify a release (image digests); everything else in an env file is configuration/secrets.
DD_RELEASE_KEYS=(GIT_COMMIT WEB_IMAGE_DIGEST BACKEND_IMAGE_DIGEST INFERENCE_IMAGE_DIGEST GATEWAY_IMAGE REDIS_IMAGE)

log() { printf '%s [%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${DD_LOG_TAG:-dreamdrive}" "$*" >&2; }
warn() { log "WARNING: $*"; }
die() { log "ERROR: $1"; exit "${2:-1}"; }

utc_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
utc_stamp() { date -u +%Y%m%dT%H%M%SZ; }

is_sha256_digest() { [[ ${1:-} =~ ^sha256:[0-9a-f]{64}$ ]]; }
is_placeholder_digest() { [[ ${1:-} == "$DD_ZERO_DIGEST" ]]; }
# name[:tag]@sha256:<64 hex>
is_pinned_image() { [[ ${1:-} =~ ^[a-z0-9][a-z0-9._/-]*(:[A-Za-z0-9._-]+)?@sha256:[0-9a-f]{64}$ ]]; }
is_git_commit() { [[ ${1:-} =~ ^[0-9a-f]{40}$ ]]; }
is_hostname() { [[ ${1:-} =~ ^([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$ ]]; }

# load_env_file FILE: assign KEY=VALUE lines as (non-exported) shell variables. Docker Compose reads the same
# file via --env-file, so values are not exported into the environment here (no shadowing of the file).
load_env_file() {
  local file=$1 line key value n=0
  [[ -f $file ]] || die "env file not found: $file"
  [[ -r $file ]] || die "env file not readable: $file"
  while IFS= read -r line || [[ -n $line ]]; do
    n=$((n + 1))
    line=${line%$'\r'}
    [[ $line =~ ^[[:space:]]*(#|$) ]] && continue
    if [[ ! $line =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      die "$file line $n is not KEY=VALUE (content not shown)"
    fi
    key=${BASH_REMATCH[2]}
    value=${BASH_REMATCH[3]}
    if [[ $value =~ ^\"(.*)\"$ ]] || [[ $value =~ ^\'(.*)\'$ ]]; then
      value=${BASH_REMATCH[1]}
    fi
    printf -v "$key" '%s' "$value"
  done <"$file"
}

# Refuse env files readable by group/others (they hold APP_SECRET/ADMIN_PASSWORD). Linux only; Windows
# filesystems do not carry POSIX modes, so the check is reported as skipped there.
check_secret_file_mode() {
  local file=$1 mode
  if ! mode=$(stat -c '%a' "$file" 2>/dev/null); then
    warn "cannot read permissions of $file; ensure it is chmod 600"
    return 0
  fi
  case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*)
      warn "permission check skipped on Windows for $file"
      return 0
      ;;
  esac
  if ((8#$mode & 8#077)); then
    die "$file has mode $mode; secrets files must not be group/world accessible (chmod 600 $file)"
  fi
}

# Clear release keys inherited from the caller's environment so only the env files decide what is deployed.
clear_release_env() {
  local k
  for k in "${DD_RELEASE_KEYS[@]}"; do unset "$k"; done
}

# host_path FILE -> absolute path in the form docker/compose accept (D:/... on Git Bash). FILE must exist.
host_path() {
  local p=$1 dir
  dir=$(cd "$(dirname "$p")" && { pwd -W 2>/dev/null || pwd; })
  printf '%s/%s\n' "$dir" "$(basename "$p")"
}

# dd_compose_setup MODE PROJECT ENV_FILE RELEASE_FILE [extra -f files...]
#   MODE=dev  -> compose.yaml (local build images), env from .env if present
#   MODE=prod -> compose.prod.yaml (registry digests only), --env-file ENV_FILE --env-file RELEASE_FILE
# Sets the global array DD_COMPOSE.
dd_compose_setup() {
  local mode=$1 project=$2 env_file=${3:-} release_file=${4:-}
  shift $(($# < 4 ? $# : 4))
  DD_COMPOSE=(docker compose --project-name "$project" --project-directory "$DD_REPO_HOST")
  case $mode in
    dev)
      [[ -f $DD_REPO_ROOT/.env ]] && DD_COMPOSE+=(--env-file "$DD_REPO_HOST/.env")
      DD_COMPOSE+=(-f "$DD_REPO_HOST/compose.yaml")
      ;;
    prod)
      [[ -n $env_file ]] || die "prod mode needs --env-file"
      DD_COMPOSE+=(--env-file "$(host_path "$env_file")")
      [[ -n $release_file ]] && DD_COMPOSE+=(--env-file "$(host_path "$release_file")")
      DD_COMPOSE+=(-f "$DD_REPO_HOST/compose.prod.yaml")
      ;;
    *) die "unknown compose mode: $mode" ;;
  esac
  local extra
  for extra in "$@"; do DD_COMPOSE+=(-f "$(host_path "$extra")"); done
}

# container_health PROJECT SERVICE -> prints healthy|unhealthy|starting|running|exited|missing
container_health() {
  local project=$1 service=$2 id
  id=$(docker ps -aq --filter "label=com.docker.compose.project=$project" \
    --filter "label=com.docker.compose.service=$service" | head -n 1)
  [[ -n $id ]] || {
    echo missing
    return 0
  }
  docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$id"
}

# wait_healthy PROJECT TIMEOUT_SECONDS SERVICE... : every service must report healthy before the deadline.
wait_healthy() {
  local project=$1 timeout=$2
  shift 2
  local deadline=$((SECONDS + timeout)) svc state pending
  while :; do
    pending=()
    for svc in "$@"; do
      state=$(container_health "$project" "$svc")
      case $state in
        healthy) ;;
        unhealthy | exited | dead | missing)
          # unhealthy can recover while still inside its retries; keep waiting until the deadline.
          pending+=("$svc=$state")
          ;;
        *) pending+=("$svc=$state") ;;
      esac
    done
    ((${#pending[@]} == 0)) && return 0
    if ((SECONDS >= deadline)); then
      log "not healthy after ${timeout}s: ${pending[*]}"
      return 1
    fi
    sleep 5
  done
}

# sha256 of a file, portable across GNU coreutils and macOS.
file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

# Python 3 on the host is only used to run the stdlib smoke client.
host_python() {
  local py
  for py in python3 python; do
    if command -v "$py" >/dev/null 2>&1 && "$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
      echo "$py"
      return 0
    fi
  done
  return 1
}
