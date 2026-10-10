#!/usr/bin/env bash
# Wait until the local stack is healthy and the gateway -> api path answers /health/ready (make up).
#   scripts/wait-ready.sh [TIMEOUT_SECONDS]      default 600 (first start includes model warm-up)
# On timeout prints service states and the last log lines of unhealthy services, then exits non-zero.
set -Eeuo pipefail
DD_LOG_TAG=wait-ready
# shellcheck source=deploy/lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../deploy/lib/common.sh"
cd "$DD_REPO_ROOT"

timeout=${1:-600}
project=${DD_PROJECT:-dreamdrive}
services=(redis api worker inference web gateway)
[[ ${OPS:-0} == 1 ]] && services+=(prometheus grafana mlflow)

if ! wait_healthy "$project" "$timeout" "${services[@]}"; then
  docker compose -p "$project" ps >&2 || warn "could not list services"
  for svc in "${services[@]}"; do
    state=$(container_health "$project" "$svc")
    if [[ $state != healthy ]]; then
      log "---- last log lines of $svc ($state)"
      docker compose -p "$project" logs --no-color --tail 40 "$svc" >&2 || warn "no logs for $svc"
    fi
  done
  die "stack not healthy within ${timeout}s"
fi
# End-to-end through the gateway container (same routing as the browser; works without host curl).
docker compose -p "$project" exec -T gateway wget -q -O /dev/null http://127.0.0.1:8080/health/ready ||
  die "gateway -> api /health/ready failed"
log "stack ready: http://127.0.0.1:${DD_HTTP_PORT:-8080}"
