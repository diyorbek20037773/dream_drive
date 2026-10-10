#!/usr/bin/env bash
# Regenerate the frontend API contract from the backend:
#   frontend/src/api/openapi.json  <- python -m app.export_openapi   (backend venv, or the backend image)
#   frontend/src/api/schema.d.ts   <- openapi-typescript             (frontend node_modules, or a node image)
#
# Usage: scripts/gen-types.sh            regenerate both files in place
#        scripts/gen-types.sh --check    regenerate into a scratch dir and fail if the committed files differ
#
# Environment overrides:
#   DD_BACKEND_PYTHON  python interpreter with the backend dependencies (default: backend/.venv)
#   DD_BACKEND_IMAGE   backend image used when no interpreter is found (default: dreamdrive-backend:local)
#   DD_NODE_IMAGE      node image used when npm is not installed (default: node:22-bookworm-slim)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
API_DIR="$FRONTEND/src/api"
MODE="write"

case "${1:-}" in
  "") ;;
  --check) MODE="check" ;;
  -h|--help) sed -n '2,13p' "${BASH_SOURCE[0]}"; exit 0 ;;
  *) echo "unknown argument: $1" >&2; exit 2 ;;
esac

die() { echo "gen-types: $*" >&2; exit 1; }

find_python() {
  if [[ -n "${DD_BACKEND_PYTHON:-}" ]]; then echo "$DD_BACKEND_PYTHON"; return 0; fi
  local candidate
  for candidate in "$BACKEND/.venv/bin/python" "$BACKEND/.venv/Scripts/python.exe" "$BACKEND/.venv/Scripts/python"; do
    if [[ -x "$candidate" ]]; then echo "$candidate"; return 0; fi
  done
  return 1
}

# export_openapi <output file>
export_openapi() {
  local out="$1" py
  if py="$(find_python)"; then
    local out_arg="$out"
    # A Windows interpreter (Git Bash) needs a native path: callers such as make export MSYS_NO_PATHCONV=1,
    # which stops MSYS from translating /d/... for it.
    if command -v cygpath >/dev/null 2>&1; then out_arg="$(cygpath -w "$out")"; fi
    (cd "$BACKEND" && APP_ENV="test" "$py" -m app.export_openapi --output "$out_arg") || die "OpenAPI export failed"
  elif command -v docker >/dev/null 2>&1; then
    local image="${DD_BACKEND_IMAGE:-dreamdrive-backend:local}"
    docker image inspect "$image" >/dev/null 2>&1 || die "no backend venv and image $image not found (build it first)"
    MSYS_NO_PATHCONV=1 docker run --rm --network none -e APP_ENV=test "$image" python -m app.export_openapi >"$out" \
      || die "OpenAPI export in $image failed"
  else
    die "need backend/.venv (or DD_BACKEND_PYTHON) or docker with the backend image"
  fi
  [[ -s "$out" ]] || die "OpenAPI export produced an empty file"
}

# gen_ts <openapi.json path relative to frontend/> <schema.d.ts path relative to frontend/>
gen_ts() {
  local in_rel="$1" out_rel="$2"
  if command -v npm >/dev/null 2>&1; then
    if [[ ! -x "$FRONTEND/node_modules/.bin/openapi-typescript" ]]; then
      (cd "$FRONTEND" && npm ci --no-audit --no-fund) || die "npm ci failed"
    fi
    (cd "$FRONTEND" && ./node_modules/.bin/openapi-typescript "$in_rel" -o "$out_rel") || die "openapi-typescript failed"
  elif command -v docker >/dev/null 2>&1; then
    local image="${DD_NODE_IMAGE:-node:22-bookworm-slim}"
    MSYS_NO_PATHCONV=1 docker run --rm -v "$FRONTEND:/work" -w /work "$image" \
      sh -c "npm ci --no-audit --no-fund && ./node_modules/.bin/openapi-typescript '$in_rel' -o '$out_rel'" \
      || die "openapi-typescript in $image failed"
  else
    die "need npm or docker to run openapi-typescript"
  fi
  [[ -s "$FRONTEND/$out_rel" ]] || die "openapi-typescript produced an empty file"
}

if [[ "$MODE" == "write" ]]; then
  mkdir -p "$API_DIR"
  export_openapi "$API_DIR/openapi.json"
  gen_ts "src/api/openapi.json" "src/api/schema.d.ts"
  echo "gen-types: wrote frontend/src/api/openapi.json and frontend/src/api/schema.d.ts"
  exit 0
fi

# --check: generate into a scratch dir under node_modules (git-ignored, visible to a containerised node too).
SCRATCH_REL="node_modules/.cache/dd-gen-types"
SCRATCH="$FRONTEND/$SCRATCH_REL"
rm -rf "$SCRATCH"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
export_openapi "$SCRATCH/openapi.json"
gen_ts "$SCRATCH_REL/openapi.json" "$SCRATCH_REL/schema.d.ts"

status=0
for name in openapi.json schema.d.ts; do
  if [[ ! -f "$API_DIR/$name" ]]; then
    echo "gen-types: frontend/src/api/$name is missing" >&2
    status=1
  elif ! diff -u --strip-trailing-cr "$API_DIR/$name" "$SCRATCH/$name" >"$SCRATCH/$name.diff"; then
    echo "gen-types: frontend/src/api/$name is stale (first lines of the diff):" >&2
    head -n 40 "$SCRATCH/$name.diff" >&2
    status=1
  fi
done
if [[ $status -ne 0 ]]; then
  echo "gen-types: run scripts/gen-types.sh and commit the result" >&2
  exit 1
fi
echo "gen-types: frontend/src/api/openapi.json and schema.d.ts are current"
