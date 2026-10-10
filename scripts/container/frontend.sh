#!/usr/bin/env bash
# Runs INSIDE node:22-bookworm-slim (compose.test.yaml service "node"); no host Node is needed.
#   frontend.sh TASK...   TASK = lint | typecheck | build | audit | test
# The read-only source (/src/frontend) is copied without host node_modules/dist, then `npm ci` installs exactly
# frontend/package-lock.json. Output files in /evidence are handed back to the host uid on Linux.
set -Eeuo pipefail
(($#)) || { echo "usage: frontend.sh lint|typecheck|build|audit|test ..." >&2; exit 64; }

restore_owner() {
  local rc=$?
  if [[ ${DD_HOST_UID:-0} != 0 ]] && ! chown -R "${DD_HOST_UID}:${DD_HOST_GID:-$DD_HOST_UID}" /evidence 2>/dev/null; then
    echo "frontend.sh: note: could not chown /evidence" >&2
  fi
  exit "$rc"
}
trap restore_owner EXIT

[[ -f /src/frontend/package.json ]] || { echo "frontend/package.json is missing" >&2; exit 1; }
[[ -f /src/frontend/package-lock.json ]] || { echo "frontend/package-lock.json is missing (npm ci needs it)" >&2; exit 1; }
mkdir -p /repo/frontend
tar -C /src/frontend --exclude=./node_modules --exclude=./dist --exclude=./test-results \
  --exclude=./playwright-report --exclude=./blob-report -cf - . | tar -C /repo/frontend -xf -
cd /repo/frontend
npm ci --no-audit --no-fund --loglevel=error

for task in "$@"; do
  echo "== npm task: $task"
  case $task in
    lint) npm run lint ;;
    typecheck) npm run typecheck ;;
    build) npm run build ;;
    # Runtime dependencies only: dev tooling is not shipped in the web image.
    audit) npm audit --omit=dev ;;
    test) npm test ;;
    *) echo "unknown task: $task" >&2; exit 64 ;;
  esac
done
