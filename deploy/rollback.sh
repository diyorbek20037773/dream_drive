#!/usr/bin/env bash
# Roll DreamDrive back to a previously deployed release recorded by deploy.sh (deploy/state/history/*.json and
# deploy/state/releases/<commit>.env). Only recorded registry digests are used; nothing is ever rebuilt.
# The rollback goes through deploy.sh (validation, backup, pull by digest, model-init, health, HTTPS smoke) and is
# recorded in the history with kind "rollback".
#
#   deploy/rollback.sh --env-file /etc/dreamdrive/dreamdrive.env --list
#   deploy/rollback.sh --env-file /etc/dreamdrive/dreamdrive.env            # newest successful older release
#   deploy/rollback.sh --env-file /etc/dreamdrive/dreamdrive.env --to <40-hex commit>
#
# Extra options are passed to deploy.sh (e.g. --project NAME, --with-ops, --local).
set -Eeuo pipefail
DD_LOG_TAG=rollback
# shellcheck source=deploy/lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

env_file="" target="" list=0 passthrough=()
while (($#)); do
  case $1 in
    --env-file) env_file=$2; shift 2 ;;
    --to) target=$2; shift 2 ;;
    --list) list=1; shift ;;
    --project) passthrough+=("$1" "$2"); shift 2 ;;
    --with-ops | --local) passthrough+=("$1"); shift ;;
    -h | --help) sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1" 64 ;;
  esac
done

history_dir="$DD_STATE_DIR/history"
[[ -d $history_dir ]] || die "no deployment history in $history_dir"

# json_field FILE KEY -> value of a top-level "KEY": "value" line written by deploy.sh
json_field() { sed -n "s/^  \"$2\": \"\\([^\"]*\\)\",\\{0,1\\}\$/\\1/p" "$1" | head -n 1; }

mapfile -t entries < <(find "$history_dir" -maxdepth 1 -type f -name '*.json' | sort)
current_commit=""
[[ -f $DD_STATE_DIR/current.json ]] && current_commit=$(json_field "$DD_STATE_DIR/current.json" git_commit)

if ((list)); then
  printf '%-22s %-8s %-9s %s\n' "recorded_utc" "status" "kind" "git_commit"
  for e in "${entries[@]}"; do
    printf '%-22s %-8s %-9s %s\n' "$(json_field "$e" recorded_utc)" "$(json_field "$e" status)" \
      "$(json_field "$e" kind)" "$(json_field "$e" git_commit)"
  done
  printf 'current: %s\n' "${current_commit:-none}"
  exit 0
fi
[[ -n $env_file ]] || die "--env-file is required" 64

if [[ -n $target ]]; then
  is_git_commit "$target" || die "--to needs a full 40-hex commit"
  found=0
  for e in "${entries[@]}"; do
    if [[ $(json_field "$e" git_commit) == "$target" && $(json_field "$e" status) == success ]]; then found=1; fi
  done
  ((found)) || die "commit $target has no successful recorded deployment; refusing to deploy unknown digests"
else
  for ((i = ${#entries[@]} - 1; i >= 0; i--)); do
    e=${entries[i]}
    commit=$(json_field "$e" git_commit)
    if [[ $(json_field "$e" status) == success && $commit != "$current_commit" ]]; then
      target=$commit
      break
    fi
  done
  [[ -n $target ]] || die "no earlier successful release recorded (current: ${current_commit:-none})"
fi

release="$DD_STATE_DIR/releases/$target.env"
[[ -f $release ]] || die "recorded release file missing: $release"
log "rolling back from ${current_commit:-none} to $target using recorded digests in $release"
exec bash "$DD_REPO_ROOT/deploy/deploy.sh" --env-file "$env_file" --release "$release" --kind rollback "${passthrough[@]}"
