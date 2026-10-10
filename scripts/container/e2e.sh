#!/usr/bin/env bash
# Runs INSIDE mcr.microsoft.com/playwright:v<version>-noble (compose.test.yaml service "e2e") against the
# running gateway (DD_BASE_URL=http://gateway:8080). Browsers come preinstalled in the image, so the
# @playwright/test version in package-lock.json must equal the image version (checked below).
# Reports: /evidence/junit-e2e.xml, /evidence/playwright-report, attachments/traces in /evidence/test-results,
# full-page screenshots (written by the specs to test-results/screenshots) in /evidence/screenshots.
set -Eeuo pipefail

restore_owner() {
  local rc=$?
  if [[ -d /repo/frontend/test-results/screenshots ]]; then
    mkdir -p /evidence/screenshots
    if ! cp -r /repo/frontend/test-results/screenshots/. /evidence/screenshots/; then
      echo "e2e.sh: note: could not copy screenshots" >&2
    fi
  fi
  if [[ ${DD_HOST_UID:-0} != 0 ]] && ! chown -R "${DD_HOST_UID}:${DD_HOST_GID:-$DD_HOST_UID}" /evidence 2>/dev/null; then
    echo "e2e.sh: note: could not chown /evidence" >&2
  fi
  exit "$rc"
}
trap restore_owner EXIT

[[ -f /src/frontend/package-lock.json ]] || { echo "frontend/package-lock.json is missing" >&2; exit 1; }
mkdir -p /repo/frontend
tar -C /src/frontend --exclude=./node_modules --exclude=./dist --exclude=./test-results \
  --exclude=./playwright-report --exclude=./blob-report -cf - . | tar -C /repo/frontend -xf -
cd /repo/frontend
npm ci --no-audit --no-fund --loglevel=error

installed=$(node -p "JSON.parse(require('fs').readFileSync('node_modules/@playwright/test/package.json', 'utf8')).version")
if [[ $installed != "${PLAYWRIGHT_VERSION:-}" ]]; then
  echo "Playwright mismatch: package-lock has $installed, image is ${PLAYWRIGHT_VERSION:-unset}" >&2
  exit 1
fi
echo "Playwright $installed against ${DD_BASE_URL}"
npx playwright test --reporter=list,junit,html --output=/evidence/test-results "$@"
