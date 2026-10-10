#!/usr/bin/env bash
# Environment check for building/running DreamDrive with Docker only. Prints PASS/WARN/FAIL lines and exits
# non-zero if any FAIL. Never prints secret values.
#   make doctor
set -Eeuo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
fails=0 warns=0
pass() { printf 'PASS  %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*"; warns=$((warns + 1)); }
fail() { printf 'FAIL  %s\n' "$*"; fails=$((fails + 1)); }

version_ge() { # version_ge HAVE WANT
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

if ((BASH_VERSINFO[0] >= 4)); then pass "bash ${BASH_VERSION}"; else fail "bash >= 4 required (have ${BASH_VERSION})"; fi
if make_version=$(make --version 2>/dev/null | head -n 1); then pass "$make_version"; else warn "GNU make not found"; fi

if command -v git >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1; then
  if commit=$(git rev-parse HEAD 2>/dev/null); then
    if [[ -z $(git status --porcelain) ]]; then
      pass "git commit $commit (clean tree)"
    else
      warn "git commit $commit with uncommitted changes: QA evidence is only release evidence on a clean tree"
    fi
  else
    warn "git repository has no commits yet"
  fi
else
  fail "not a git checkout (GIT_COMMIT build arg and QA evidence need git)"
fi

if ! command -v docker >/dev/null 2>&1; then
  fail "docker CLI not found"
elif ! docker info >/dev/null 2>&1; then
  fail "Docker daemon not reachable (start Docker Desktop / dockerd)"
else
  engine=$(docker version --format '{{.Server.Version}}')
  if version_ge "$engine" 25.0.0; then pass "Docker Engine $engine"; else fail "Docker Engine >= 25 required (have $engine)"; fi
  arch=$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')
  if [[ $arch == linux/amd64 ]]; then
    pass "engine platform $arch"
  else
    warn "engine platform $arch: only linux/amd64 images are built and supported"
  fi
  if compose=$(docker compose version --short 2>/dev/null); then
    if version_ge "${compose#v}" 2.24.0; then pass "Docker Compose $compose"; else fail "Docker Compose >= 2.24 required (have $compose)"; fi
  else
    fail "docker compose plugin not found"
  fi
  if bx=$(docker buildx version 2>/dev/null); then pass "buildx ${bx#github.com/docker/buildx }"; else warn "docker buildx not found (needed for release builds)"; fi
  mem=$(docker info --format '{{.MemTotal}}')
  cpus=$(docker info --format '{{.NCPU}}')
  mem_gib=$((mem / 1024 / 1024 / 1024))
  if ((mem >= 6 * 1024 * 1024 * 1024)); then
    pass "Docker VM: ${cpus} CPU, ${mem_gib} GiB RAM"
  else
    warn "Docker VM has ${mem_gib} GiB RAM; the CPU stack limits add up to ~7 GiB (inference 3 GiB, worker 2 GiB)"
  fi
  if docker info --format '{{json .Runtimes}}' | grep -q nvidia; then
    pass "nvidia container runtime present (optional GPU profile; CPU is the default)"
  else
    pass "no nvidia runtime (CPU default needs none)"
  fi
  root_dir=$(docker info --format '{{.DockerRootDir}}')
  pass "docker root dir $root_dir (check free space with: docker system df)"
fi

for f in compose.yaml compose.prod.yaml backend/Dockerfile inference/Dockerfile frontend/Dockerfile \
  config/models.resolved.yaml backend/uv.lock inference/uv.lock frontend/package-lock.json; do
  if [[ -f $f ]]; then pass "found $f"; else fail "missing $f"; fi
done
if [[ -f inference/Dockerfile.gpu ]]; then pass "found inference/Dockerfile.gpu (optional)"; else warn "no inference/Dockerfile.gpu (GPU profile unavailable)"; fi

if [[ -f .env ]]; then
  if grep -Eq '^APP_SECRET=.{32,}' .env; then pass ".env present with APP_SECRET set (value not shown)"; else warn ".env has no APP_SECRET; run make init-env"; fi
  if ! grep -Eq '^GRAFANA_ADMIN_PASSWORD=.+' .env; then warn ".env has no GRAFANA_ADMIN_PASSWORD (needed only for OPS=1); run make init-env"; fi
else
  warn ".env missing; run make init-env (make up does it automatically)"
fi

port="${DD_HTTP_PORT:-8080}"
if command -v curl >/dev/null 2>&1 && curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$port/" 2>/dev/null; then
  if docker ps --filter "label=com.docker.compose.project=dreamdrive" --filter "publish=$port" -q | grep -q .; then
    pass "port $port served by the dreamdrive gateway"
  else
    warn "port $port is already in use by something else; set DD_HTTP_PORT in .env"
  fi
else
  pass "port $port free"
fi

printf '\ndoctor: %d failure(s), %d warning(s)\n' "$fails" "$warns"
((fails == 0))
