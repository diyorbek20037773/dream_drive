#!/usr/bin/env bash
# Static deployment validation (make deploy-check). No containers are started and nothing is changed.
#  1. `docker compose config -q` for every supported file combination
#  2. deploy/deploy.sh --check with clearly fake placeholder values (all-zero digests, *.invalid domain)
#  3. negative cases: deploy.sh must REFUSE placeholder digests, blank digests, tag-only images and a missing
#     APP_SECRET when not in placeholder mode (proves the guards work)
#  4. Caddyfile (local) validation
# The placeholder files are generated in a temp dir and deleted; they contain no real secrets or digests.
set -Eeuo pipefail
DD_LOG_TAG=deploy-check
# shellcheck source=deploy/lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../deploy/lib/common.sh"
cd "$DD_REPO_ROOT"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
zero=$DD_ZERO_DIGEST
cat >"$tmp/dev.env" <<'EOF'
# deploy-check placeholder (not a real password)
GRAFANA_ADMIN_PASSWORD=deploy-check-placeholder
EOF
cat >"$tmp/prod.env" <<'EOF'
# deploy-check placeholders: NOT real secrets.
APP_ENV=production
DOMAIN=deploy-check.invalid
PUBLIC_ORIGIN=https://deploy-check.invalid
APP_SECRET=deploy-check-placeholder-not-a-secret-0000000000
GRAFANA_ADMIN_PASSWORD=deploy-check-placeholder
EOF
cat >"$tmp/release.env" <<EOF
# deploy-check placeholders: NOT real digests (all zeros).
GIT_COMMIT=0000000000000000000000000000000000000000
WEB_IMAGE_DIGEST=$zero
BACKEND_IMAGE_DIGEST=$zero
INFERENCE_IMAGE_DIGEST=$zero
GATEWAY_IMAGE=caddy:2.10-alpine@$zero
REDIS_IMAGE=redis:7.4-alpine@$zero
EOF
chmod 600 "$tmp"/*.env

failures=0
check() { # check NAME CMD...
  local name=$1
  shift
  if "$@" >"$tmp/out.log" 2>&1; then
    printf 'PASS  %s\n' "$name"
  else
    printf 'FAIL  %s\n' "$name"
    sed 's/^/      /' "$tmp/out.log" | tail -n 20
    failures=$((failures + 1))
  fi
}
refuse() { # refuse NAME PATTERN CMD... : the command must exit non-zero AND report PATTERN
  local name=$1 pattern=$2
  shift 2
  if "$@" >"$tmp/out.log" 2>&1; then
    printf 'FAIL  %s (was accepted)\n' "$name"
    failures=$((failures + 1))
  elif ! grep -q -- "$pattern" "$tmp/out.log"; then
    printf 'FAIL  %s (refused, but not for the expected reason: %s)\n' "$name" "$pattern"
    sed 's/^/      /' "$tmp/out.log" | tail -n 10
    failures=$((failures + 1))
  else
    printf 'PASS  %s (refused: %s)\n' "$name" "$(grep -m 1 -- "$pattern" "$tmp/out.log" | sed 's/^.*\] //')"
  fi
}

dev=(docker compose --project-directory "$DD_REPO_HOST" --env-file "$(host_path "$tmp/dev.env")")
prod=(docker compose --project-directory "$DD_REPO_HOST" --env-file "$(host_path "$tmp/prod.env")"
  --env-file "$(host_path "$tmp/release.env")")
check "compose.yaml" "${dev[@]}" -f compose.yaml config -q
check "compose.yaml + compose.gpu.yaml" "${dev[@]}" -f compose.yaml -f compose.gpu.yaml config -q
check "compose.yaml + compose.ops.yaml" "${dev[@]}" -f compose.yaml -f compose.ops.yaml config -q
check "compose.yaml + compose.gpu.yaml + compose.ops.yaml" \
  "${dev[@]}" -f compose.yaml -f compose.gpu.yaml -f compose.ops.yaml config -q
check "compose.yaml + compose.test.yaml (profile test)" \
  "${dev[@]}" -f compose.yaml -f compose.test.yaml --profile test config -q
check "compose.prod.yaml" "${prod[@]}" -f compose.prod.yaml config -q
check "compose.prod.yaml + compose.ops.yaml" "${prod[@]}" -f compose.prod.yaml -f compose.ops.yaml config -q

# Production compose must stay registry-only and expose no host ports except the gateway.
check "compose.prod.yaml has no build sections" bash -c '! grep -nE "^[[:space:]]+build:" compose.prod.yaml'
check "compose.prod.yaml images are digest-pinned" bash -c \
  "${prod[*]@Q} -f compose.prod.yaml config --images | grep -v '@sha256:' | (! grep .)"
check "only the gateway publishes ports (dev + prod)" bash -c \
  "for f in compose.yaml compose.prod.yaml; do
     awk '/^  [a-z-]+:\$/ {svc=\$1} /^    ports:/ {print FILENAME\": \"svc}' \$f; done | grep -v 'gateway:' | (! grep .)"
check "ops UIs bind to 127.0.0.1 only" bash -c \
  "grep -A2 -E '^    ports:' compose.ops.yaml | grep -E '^\s+- ' | grep -v '\"127.0.0.1:' | (! grep .)"

check "deploy.sh --check --placeholder-ok" \
  bash "$DD_REPO_ROOT/deploy/deploy.sh" --env-file "$tmp/prod.env" --release "$tmp/release.env" --check --placeholder-ok

# Negative cases: the real deploy path must refuse these before touching anything. Where the domain is not the
# subject, a non-reserved name is used (never contacted: --check stops before any network access).
deploy_sh="$DD_REPO_ROOT/deploy/deploy.sh"
variant() { # variant SRC DEST SED_EXPR
  sed "$3" "$1" >"$2"
  chmod 600 "$2"
}
variant "$tmp/prod.env" "$tmp/prod-domain.env" 's/deploy-check[.]invalid/dreamdrive.deploy-check.net/g'
refuse "deploy.sh refuses all-zero placeholder digests" "all-zero placeholder" \
  bash "$deploy_sh" --env-file "$tmp/prod-domain.env" --release "$tmp/release.env" --check
refuse "deploy.sh refuses a reserved placeholder domain" "reserved/placeholder name" \
  bash "$deploy_sh" --env-file "$tmp/prod.env" --release "$tmp/release.env" --check
variant "$tmp/release.env" "$tmp/blank.env" 's/^WEB_IMAGE_DIGEST=.*/WEB_IMAGE_DIGEST=/'
refuse "deploy.sh refuses a blank digest" "WEB_IMAGE_DIGEST must be sha256" \
  bash "$deploy_sh" --env-file "$tmp/prod.env" --release "$tmp/blank.env" --check --placeholder-ok
variant "$tmp/release.env" "$tmp/short.env" 's/^BACKEND_IMAGE_DIGEST=.*/BACKEND_IMAGE_DIGEST=sha256:abc123/'
refuse "deploy.sh refuses a malformed digest" "BACKEND_IMAGE_DIGEST must be sha256" \
  bash "$deploy_sh" --env-file "$tmp/prod.env" --release "$tmp/short.env" --check --placeholder-ok
variant "$tmp/release.env" "$tmp/tagonly.env" 's/^REDIS_IMAGE=.*/REDIS_IMAGE=redis:7.4-alpine/'
refuse "deploy.sh refuses a tag-only third-party image" "REDIS_IMAGE must be pinned" \
  bash "$deploy_sh" --env-file "$tmp/prod.env" --release "$tmp/tagonly.env" --check --placeholder-ok
variant "$tmp/prod.env" "$tmp/nosecret.env" 's/^APP_SECRET=.*/APP_SECRET=/'
refuse "deploy.sh refuses a missing APP_SECRET" "APP_SECRET must be set" \
  bash "$deploy_sh" --env-file "$tmp/nosecret.env" --release "$tmp/release.env" --check --placeholder-ok
variant "$tmp/prod.env" "$tmp/origin.env" 's|^PUBLIC_ORIGIN=.*|PUBLIC_ORIGIN=http://deploy-check.invalid|'
refuse "deploy.sh refuses a non-HTTPS PUBLIC_ORIGIN" "PUBLIC_ORIGIN must equal" \
  bash "$deploy_sh" --env-file "$tmp/origin.env" --release "$tmp/release.env" --check --placeholder-ok
variant "$tmp/prod.env" "$tmp/devmode.env" 's/^APP_ENV=.*/APP_ENV=development/'
refuse "deploy.sh refuses APP_ENV other than production" "APP_ENV must be production" \
  bash "$deploy_sh" --env-file "$tmp/devmode.env" --release "$tmp/release.env" --check --placeholder-ok
refuse "deploy.sh refuses --placeholder-ok outside --check" "only allowed together with --check" \
  bash "$deploy_sh" --env-file "$tmp/prod.env" --release "$tmp/release.env" --placeholder-ok

check "Caddyfile (local) validates" docker run --rm --network none -v "$DD_REPO_HOST/deploy/caddy:/etc/caddy:ro" \
  caddy:2.10-alpine caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile

printf '\ndeploy-check: %d failure(s)\n' "$failures"
((failures == 0))
