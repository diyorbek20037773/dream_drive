#!/usr/bin/env bash
# SPDX SBOMs for the DreamDrive images with Syft (anchore/syft container; reads images from the local Docker
# daemon). Release images additionally get BuildKit SBOM + provenance attestations in .github/workflows/release.yml.
#
#   scripts/sbom.sh [IMAGE...]        default: dreamdrive-{web,backend,inference}:${DD_TAG:-dev}
# Output: ${DD_EVIDENCE_DIR:-artifacts/sbom/<commit>}/<name>.spdx.json
set -Eeuo pipefail
DD_LOG_TAG=sbom
# shellcheck source=deploy/lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../deploy/lib/common.sh"

# shellcheck source=scripts/tool-images.env
source "$DD_REPO_ROOT/scripts/tool-images.env"
syft_image=$SYFT_IMAGE
tag=${DD_TAG:-dev}
images=("$@")
((${#images[@]})) || images=("dreamdrive-web:$tag" "dreamdrive-backend:$tag" "dreamdrive-inference:$tag")
commit=$(git -C "$DD_REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
out="${DD_EVIDENCE_DIR:-$DD_REPO_ROOT/artifacts/sbom/$commit}"
mkdir -p "$out"

for image in "${images[@]}"; do
  docker image inspect "$image" >/dev/null 2>&1 || die "image $image not found locally (run make build)"
  name=$(printf '%s' "$image" | tr '/:@' '___')
  log "syft $image -> $out/$name.spdx.json"
  docker run --rm -v /var/run/docker.sock:/var/run/docker.sock "$syft_image" "docker:$image" -o spdx-json \
    >"$out/$name.spdx.json.part"
  [[ -s $out/$name.spdx.json.part ]] || die "empty SBOM for $image"
  mv -f "$out/$name.spdx.json.part" "$out/$name.spdx.json"
done
log "SBOMs written to $out"
