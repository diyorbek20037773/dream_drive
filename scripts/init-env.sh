#!/usr/bin/env bash
# Create the local .env from .env.example and fill blank secrets with freshly generated random values.
# Existing non-blank values are never changed; secret values are never printed.
#
#   scripts/init-env.sh                 # .env in the repository root (make init-env)
#   DD_ENV_FILE=/path/.env scripts/init-env.sh
#
# Generated: APP_SECRET (48 random bytes, base64url), GRAFANA_ADMIN_PASSWORD (24 random bytes).
# ADMIN_PASSWORD stays blank on purpose: operator login is disabled until you choose a password yourself.
set -Eeuo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${DD_ENV_FILE:-$root/.env}"
example="$root/.env.example"
[[ -f $example ]] || { echo "init-env: $example not found" >&2; exit 1; }

random_token() { # random_token BYTES -> base64url without padding, from the kernel CSPRNG
  head -c "$1" /dev/urandom | base64 | tr -d '\n=' | tr '+/' '-_'
}

umask 077
if [[ ! -f $env_file ]]; then
  cp "$example" "$env_file"
  echo "init-env: created $env_file from .env.example"
fi
chmod 600 "$env_file" 2>/dev/null || echo "init-env: note: could not chmod 600 $env_file (non-POSIX filesystem)" >&2

# set_if_blank KEY VALUE: replace "KEY=" (blank) in place, or append when the key is absent.
set_if_blank() {
  local key=$1 value=$2 tmp
  if grep -Eq "^${key}=.+" "$env_file"; then
    echo "init-env: $key already set (unchanged)"
    return 0
  fi
  tmp=$(mktemp "${env_file}.XXXXXX")
  if grep -Eq "^${key}=" "$env_file"; then
    awk -v k="$key" -v v="$value" 'BEGIN { FS = OFS = "=" } $1 == k && $2 == "" { print k "=" v; next } { print }' \
      "$env_file" >"$tmp"
  else
    cat "$env_file" >"$tmp"
    printf '%s=%s\n' "$key" "$value" >>"$tmp"
  fi
  mv -f "$tmp" "$env_file"
  echo "init-env: generated $key (value not shown)"
}

set_if_blank APP_SECRET "$(random_token 48)"
set_if_blank GRAFANA_ADMIN_PASSWORD "$(random_token 24)"
echo "init-env: done. Secrets live only in $env_file (git-ignored)."
