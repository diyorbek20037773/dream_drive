#!/usr/bin/env bash
# DreamDrive smoke test against a running gateway: health, web root, /metrics denied, capabilities (+ commit),
# session + CSRF, authenticated read, SSE, and optionally a real upload processed end-to-end.
#
#   deploy/smoke.sh https://dreamdrive.example.org --expect-commit <sha> [--upload clip.mp4]
#   deploy/smoke.sh http://127.0.0.1:8080 --upload tests/fixtures/<clip>.mp4 --evidence artifacts/qa/<sha>/smoke
#
# Needs python3 (stdlib only) on the host; on Linux without python3 it runs the same client in a
# python:3.11-slim container with host networking.
set -Eeuo pipefail
DD_LOG_TAG=smoke
# shellcheck source=deploy/lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

if [[ $# -lt 1 || $1 == -h || $1 == --help ]]; then
  sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 64
fi

if py=$(host_python); then
  exec "$py" "$DD_REPO_ROOT/deploy/lib/smoke.py" "$@"
fi
[[ $(uname -s) == Linux ]] || die "python3 not found; install Python 3.9+ to run the smoke client"
log "python3 not found on host; running the smoke client in python:3.11-slim-bookworm (host network)"
args=()
mounts=(-v "$DD_REPO_ROOT/deploy/lib/smoke.py:/smoke.py:ro")
prev=""
for a in "$@"; do
  # Files given to --upload/--evidence are mounted at the same absolute path inside the container.
  if [[ $prev == --upload || $prev == --evidence ]]; then
    abs=$(cd "$(dirname "$a")" && pwd)/$(basename "$a")
    [[ $prev == --evidence ]] && mkdir -p "$abs"
    mounts+=(-v "$abs:$abs")
    a=$abs
  fi
  args+=("$a")
  prev=$a
done
exec docker run --rm --network host --user "$(id -u):$(id -g)" "${mounts[@]}" python:3.11-slim-bookworm \
  python /smoke.py "${args[@]}"
