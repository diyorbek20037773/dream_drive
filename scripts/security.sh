#!/usr/bin/env bash
# DreamDrive security gate (make security). Every check runs; the script fails if any check failed.
#   tests      tests/security in the backend test image against the running stack (DD_REQUIRE_LIVE=1)
#   pip-audit  backend + inference locked runtime dependencies (uv export from uv.lock -> pip-audit)
#   npm-audit  frontend runtime dependencies (npm audit --omit=dev) in a Node container
#   gitleaks   secret scan of the full git history (reviewed false positives: ops/security/gitleaksignore)
#   trivy      image scan of dreamdrive-{web,backend,inference}: fail on fixable HIGH/CRITICAL
#
#   scripts/security.sh [CHECK...]      default: all checks, in the order above
# Evidence: ${DD_EVIDENCE_DIR:-./artifacts/qa/local}/ (requirements exports, per-check logs).
set -Euo pipefail
DD_LOG_TAG=security
# shellcheck source=deploy/lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../deploy/lib/common.sh"
# shellcheck source=scripts/tool-images.env
source "$DD_REPO_ROOT/scripts/tool-images.env"
cd "$DD_REPO_ROOT" || exit 1

checks=("$@")
((${#checks[@]})) || checks=(tests pip-audit npm-audit gitleaks trivy)
tag=${DD_TAG:-dev}
evidence=${DD_EVIDENCE_DIR:-./artifacts/qa/local}
export DD_EVIDENCE_DIR=$evidence
mkdir -p "$evidence"
chmod 0777 "$evidence" 2>/dev/null || warn "could not chmod $evidence"
test_compose=(docker compose -f compose.yaml -f compose.test.yaml --profile test)

run_tests() {
  "${test_compose[@]}" run --rm --no-deps -e DD_REQUIRE_LIVE=1 backend-test \
    python -m pytest -q -p no:cacheprovider ../tests/security --junitxml=/evidence/junit-security.xml
}

run_pip_audit() {
  local component rc=0
  for component in backend inference; do
    log "uv export ($component/uv.lock, runtime dependencies only)"
    # Local version labels such as +cpu (PyTorch CPU wheels) are stripped so the vulnerability database matches
    # the upstream release; the exported file keeps exact pins.
    docker run --rm -e UV_CACHE_DIR=/tmp/uv -v "$DD_REPO_HOST/$component:/src:ro" -w /src "$UV_IMAGE" \
      export --frozen --no-dev --no-hashes --no-emit-project --format requirements-txt |
      sed -E 's/^([A-Za-z0-9._-]+==[^ ;+]+)\+[A-Za-z0-9.]+/\1/' >"$evidence/requirements-$component.txt" ||
      return 1
    [[ -s $evidence/requirements-$component.txt ]] || return 1
    log "pip-audit $component"
    "${test_compose[@]}" run --rm --no-deps backend-test python -m pip_audit --progress-spinner off \
      --no-deps --disable-pip -r "/evidence/requirements-$component.txt" || rc=1
  done
  return "$rc"
}

run_npm_audit() {
  "${test_compose[@]}" run --rm --no-deps node audit
}

run_gitleaks() {
  docker run --rm -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0='*' \
    -v "$DD_REPO_HOST:/repo:ro" "$GITLEAKS_IMAGE" git /repo --redact --no-banner --no-color \
    --gitleaks-ignore-path /repo/ops/security/gitleaksignore
}

run_trivy() {
  local image rc=0
  for image in "dreamdrive-web:$tag" "dreamdrive-backend:$tag" "dreamdrive-inference:$tag"; do
    if ! docker image inspect "$image" >/dev/null 2>&1; then
      log "image $image not found (run make build)"
      rc=1
      continue
    fi
    log "trivy $image (fail on fixable HIGH/CRITICAL)"
    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v dreamdrive-trivy-cache:/root/.cache/ \
      "$TRIVY_IMAGE" image --scanners vuln --severity HIGH,CRITICAL --ignore-unfixed --exit-code 1 \
      --no-progress --timeout 20m "$image" || rc=1
  done
  return "$rc"
}

declare -A results
failed=0
for check in "${checks[@]}"; do
  log "== $check"
  case $check in
    tests) run_tests ;;
    pip-audit) run_pip_audit ;;
    npm-audit) run_npm_audit ;;
    gitleaks) run_gitleaks ;;
    trivy) run_trivy ;;
    *) die "unknown check: $check" 64 ;;
  esac
  rc=$?
  if ((rc == 0)); then results[$check]=pass; else results[$check]="FAIL($rc)"; failed=1; fi
done

printf '\nsecurity summary:\n'
for check in "${checks[@]}"; do printf '  %-10s %s\n' "$check" "${results[$check]}"; done
((failed == 0))
