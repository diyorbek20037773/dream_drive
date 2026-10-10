#!/usr/bin/env bash
# Restore a DreamDrive backup into a SEPARATE, throw-away Compose project and verify it: SQLite integrity, table
# counts equal to the backup manifest, artifact file counts, and (with --require-job) at least one job row and a
# report.json for succeeded jobs. --boot-api additionally starts redis+api of that project on the restored data
# and requires /health/ready. The live stack and its volumes are never touched.
#
#   deploy/restore-test.sh --dev [--backup FILE] [--require-job] [--boot-api]          (make restore-test)
#   deploy/restore-test.sh --env-file /etc/dreamdrive/dreamdrive.env [--release FILE] [--backup FILE] ...
#
# Default backup: newest archive in ./backups (DD_BACKUP_DIR). --evidence DIR copies the JSON result there.
# --keep leaves the restore project running for inspection (remove it later with: docker compose -p NAME down -v).
set -Eeuo pipefail
DD_LOG_TAG=restore-test
# shellcheck source=deploy/lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

mode="" env_file="" release_file="" backup="" require_job=0 boot_api=0 keep=0 evidence=""
while (($#)); do
  case $1 in
    --dev) mode=dev; shift ;;
    --env-file) mode=prod; env_file=$2; shift 2 ;;
    --release) release_file=$2; shift 2 ;;
    --backup) backup=$2; shift 2 ;;
    --require-job) require_job=1; shift ;;
    --boot-api) boot_api=1; shift ;;
    --keep) keep=1; shift ;;
    --evidence) evidence=$2; shift 2 ;;
    -h | --help) sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1" 64 ;;
  esac
done
[[ -n $mode ]] || die "choose --dev or --env-file FILE" 64

if [[ -z $backup ]]; then
  backup=$(find "$DD_BACKUP_DIR" -maxdepth 1 -type f -name 'dreamdrive-backup-*.tar.gz' 2>/dev/null | sort | tail -n 1)
  [[ -n $backup ]] || die "no backup archive found in $DD_BACKUP_DIR (run deploy/backup.sh first)"
fi
[[ -f $backup ]] || die "backup not found: $backup"
if [[ -f $backup.sha256 ]]; then
  want=$(cut -d' ' -f1 <"$backup.sha256")
  have=$(file_sha256 "$backup")
  [[ $want == "$have" ]] || die "checksum mismatch for $backup"
  log "checksum ok ($have)"
else
  warn "no .sha256 next to $backup; integrity is still verified by gzip/tar and the restore checks"
fi

if [[ $mode == prod && -z $release_file && -f $DD_STATE_DIR/current-release.env ]]; then
  release_file="$DD_STATE_DIR/current-release.env"
fi
project="dreamdrive-restoretest-$(date -u +%Y%m%d%H%M%S)"
dd_compose_setup "$mode" "$project" "$env_file" "$release_file"

cleanup() {
  local rc=$?
  if ((keep)); then
    log "keeping project $project (remove with: docker compose -p $project down -v)"
  elif [[ $project == dreamdrive-restoretest-* ]]; then
    # Only the throw-away restore project is removed, including its own volumes.
    "${DD_COMPOSE[@]}" down --volumes --remove-orphans >/dev/null 2>&1 ||
      warn "could not remove restore project $project"
  fi
  exit "$rc"
}
trap cleanup EXIT

log "restoring $(basename "$backup") into separate project $project"
"${DD_COMPOSE[@]}" run --rm --no-deps -T volume-init >/dev/null

script=$(cat "$DD_REPO_ROOT/deploy/lib/backup_data.py")
restore_args=(restore)
((require_job)) && restore_args+=(--require-job)
result_file=$(mktemp)
status=0
"${DD_COMPOSE[@]}" run --rm --no-deps -T --entrypoint python api -c "$script" "${restore_args[@]}" \
  <"$backup" >"$result_file" || status=$?
cat "$result_file"
if [[ -n $evidence ]]; then
  mkdir -p "$evidence"
  cp "$result_file" "$evidence/restore-test.json"
fi
rm -f "$result_file"
((status == 0)) || die "restore verification failed (exit $status)"

if ((boot_api)); then
  log "booting redis + api of $project on the restored data"
  "${DD_COMPOSE[@]}" up -d api >/dev/null
  wait_healthy "$project" 180 redis api || die "restored api did not become healthy"
  "${DD_COMPOSE[@]}" exec -T api python -c \
    "import urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=10); print(r.status)" ||
    die "restored api /health/ready failed"
  log "restored api is ready"
fi
log "restore test passed"
