#!/usr/bin/env bash
# Run docker compose against the production stack with the recorded current release (deploy/state).
# For inspection and routine operations; deploys and rollbacks go through deploy.sh / rollback.sh.
#
#   deploy/compose.sh --env-file /etc/dreamdrive/dreamdrive.env ps
#   deploy/compose.sh --env-file /etc/dreamdrive/dreamdrive.env logs --tail 200 worker
#   deploy/compose.sh --env-file /etc/dreamdrive/dreamdrive.env --with-ops up -d prometheus grafana
#
# Refuses destructive volume operations (down -v / --volumes, volume rm); use them manually only after a backup.
set -Eeuo pipefail
DD_LOG_TAG=compose
# shellcheck source=deploy/lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

env_file="" project="${DD_PROJECT:-dreamdrive}" extra=()
while (($#)); do
  case $1 in
    --env-file) env_file=$2; shift 2 ;;
    --project) project=$2; shift 2 ;;
    --with-ops) extra+=("$DD_REPO_ROOT/compose.ops.yaml"); shift ;;
    -h | --help) sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) break ;;
  esac
done
[[ -n $env_file ]] || die "--env-file is required" 64
(($#)) || die "no compose command given" 64
for arg in "$@"; do
  case $arg in
    -v | --volumes | --volumes=*) die "refusing to remove volumes through this wrapper" ;;
  esac
done
release="$DD_STATE_DIR/current-release.env"
[[ -f $release ]] || die "no recorded release ($release); deploy first with deploy/deploy.sh"
dd_compose_setup prod "$project" "$env_file" "$release" "${extra[@]}"
exec "${DD_COMPOSE[@]}" "$@"
