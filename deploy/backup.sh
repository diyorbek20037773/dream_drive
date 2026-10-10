#!/usr/bin/env bash
# Coherent DreamDrive data backup: SQLite online backup (sqlite3 backup API inside the backend image, never a
# raw copy of the live WAL database) + artifacts/ + evaluations/ + backup-manifest.json, as one tar.gz.
#
#   deploy/backup.sh --dev                                              # local compose.yaml stack (make backup)
#   deploy/backup.sh --env-file /etc/dreamdrive/dreamdrive.env [--release FILE]   # production stack
#
# Options: --project NAME (default dreamdrive), --out DIR (default ./backups, or DD_BACKUP_DIR),
#          --keep N (newest N archives kept in DIR, default 14 or DD_BACKUP_KEEP).
# Offsite copy: if DD_BACKUP_OFFSITE_CMD is set (e.g. 'rclone copy'), it is run as: $DD_BACKUP_OFFSITE_CMD <archive>
# and a failure fails the backup. Nothing leaves the host unless you configure it.
# Prints the archive path on stdout. Redis is not backed up: queued work is transient; job history is in SQLite.
set -Eeuo pipefail
DD_LOG_TAG=backup
# shellcheck source=deploy/lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

mode="" env_file="" release_file="" project="${DD_PROJECT:-dreamdrive}"
out_dir="$DD_BACKUP_DIR" keep="${DD_BACKUP_KEEP:-14}"
while (($#)); do
  case $1 in
    --dev) mode=dev; shift ;;
    --env-file) mode=prod; env_file=$2; shift 2 ;;
    --release) release_file=$2; shift 2 ;;
    --project) project=$2; shift 2 ;;
    --out) out_dir=$2; shift 2 ;;
    --keep) keep=$2; shift 2 ;;
    -h | --help) sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1" 64 ;;
  esac
done
[[ -n $mode ]] || die "choose --dev or --env-file FILE" 64
[[ $keep =~ ^[1-9][0-9]*$ ]] || die "--keep must be a positive integer" 64

release_json='{"mode":"dev"}'
if [[ $mode == prod ]]; then
  check_secret_file_mode "$env_file"
  # Default to the release that is currently deployed, so the backup is taken with the running image.
  if [[ -z $release_file && -f $DD_STATE_DIR/current-release.env ]]; then
    release_file="$DD_STATE_DIR/current-release.env"
  fi
  [[ -n $release_file ]] || die "no --release given and no $DD_STATE_DIR/current-release.env"
  clear_release_env
  load_env_file "$release_file"
  release_json=$(printf '{"mode":"prod","git_commit":"%s","backend_image_digest":"%s"}' \
    "${GIT_COMMIT:-unknown}" "${BACKEND_IMAGE_DIGEST:-unknown}")
elif commit=$(git -C "$DD_REPO_ROOT" rev-parse HEAD 2>/dev/null); then
  release_json=$(printf '{"mode":"dev","git_commit":"%s"}' "$commit")
fi

dd_compose_setup "$mode" "$project" "$env_file" "$release_file"
docker volume inspect "${project}_app-data" >/dev/null 2>&1 ||
  die "volume ${project}_app-data does not exist: nothing to back up" 3

mkdir -p "$out_dir"
if ! chmod 700 "$out_dir" 2>/dev/null; then warn "could not chmod 700 $out_dir"; fi
stamp=$(utc_stamp)
final="$out_dir/dreamdrive-backup-$stamp.tar.gz"
part="$out_dir/.dreamdrive-backup-$stamp.tar.gz.part"
errlog="$out_dir/.dreamdrive-backup-$stamp.log"
trap 'rm -f "$part" "$errlog"' EXIT

script=$(cat "$DD_REPO_ROOT/deploy/lib/backup_data.py")
log "backing up project $project (volume ${project}_app-data) -> $final"
status=0
"${DD_COMPOSE[@]}" run --rm --no-deps -T -e "DD_BACKUP_RELEASE_JSON=$release_json" \
  --entrypoint python api -c "$script" backup >"$part" 2>"$errlog" || status=$?
if ((status != 0)); then
  cat "$errlog" >&2
  # 3 = no database yet (fresh volume); deploy.sh treats that as "nothing to back up".
  if grep -q '"database_missing"' "$errlog"; then die "no database in ${project}_app-data yet" 3; fi
  die "backup command failed (exit $status)"
fi

gzip -t "$part" || die "archive failed gzip integrity test"
# Relative names: GNU tar would read "C:/..." (Git Bash) as a remote host.
part_name=$(basename "$part")
last_member=$(cd "$out_dir" && tar -tzf "$part_name" | tail -n 1)
[[ $last_member == backup-manifest.json ]] || die "archive is incomplete (manifest is not the last member)"
(cd "$out_dir" && tar -xzOf "$part_name" backup-manifest.json) >"$final.manifest.json"
mv -f "$part" "$final"
if ! chmod 600 "$final" "$final.manifest.json" 2>/dev/null; then warn "could not chmod 600 the archive"; fi
(cd "$out_dir" && printf '%s  %s\n' "$(file_sha256 "$(basename "$final")")" "$(basename "$final")" \
  >"$(basename "$final").sha256")
log "backup ok: $(du -h "$final" | cut -f1) sha256=$(cut -d' ' -f1 <"$final.sha256")"

if [[ -n ${DD_BACKUP_OFFSITE_CMD:-} ]]; then
  log "offsite copy via DD_BACKUP_OFFSITE_CMD"
  # Intentionally word-split: the variable holds a command plus its fixed arguments.
  # shellcheck disable=SC2086
  $DD_BACKUP_OFFSITE_CMD "$final" || die "offsite copy failed; local archive kept at $final"
fi

# Retention: only our own archive names in this directory, newest $keep kept.
mapfile -t archives < <(find "$out_dir" -maxdepth 1 -type f -name 'dreamdrive-backup-*.tar.gz' | sort)
if ((${#archives[@]} > keep)); then
  for old in "${archives[@]:0:${#archives[@]}-keep}"; do
    log "retention: removing $(basename "$old")"
    rm -f "$old" "$old.sha256" "$old.manifest.json"
  done
fi
printf '%s\n' "$final"
