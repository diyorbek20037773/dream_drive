#!/usr/bin/env bash
# Print the exact @playwright/test version pinned in frontend/package-lock.json (no Node needed).
# The e2e container image mcr.microsoft.com/playwright:v<version>-noble must match it.
set -Eeuo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lock="$root/frontend/package-lock.json"
[[ -f $lock ]] || { echo "playwright-version: $lock not found" >&2; exit 1; }
version=$(grep -A 3 '"node_modules/@playwright/test": {' "$lock" | sed -n 's/.*"version": "\([0-9][^"]*\)".*/\1/p' | head -n 1)
[[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "playwright-version: @playwright/test not found in $lock" >&2; exit 1; }
printf '%s\n' "$version"
