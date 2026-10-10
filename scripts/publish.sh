#!/usr/bin/env bash
# Manual publication with an existing `docker login` (same rules as .github/workflows/release.yml), for when the
# CI secrets are not configured. Refuses unless:
#   - the working tree is clean and HEAD is the commit being published,
#   - artifacts/qa/<commit>/summary.json says "result": "pass" and "release_evidence": true (make qa on HEAD),
#   - none of the immutable <namespace>/dreamdrive-*:sha-<commit> tags exists yet.
# Builds each image once with buildx for linux/amd64 (SBOM + max provenance attestations), pushes it, then resolves
# the registry digests into artifacts/release/<commit>/release.env + release-manifest.json.
#
#   scripts/publish.sh [--namespace diyorbek7377] [--version vX.Y.Z]
# Next: make smoke-published RELEASE_ENV=artifacts/release/<commit>/release.env UPLOAD=<clip.mp4>
set -Eeuo pipefail
DD_LOG_TAG=publish
# shellcheck source=deploy/lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../deploy/lib/common.sh"
cd "$DD_REPO_ROOT" || exit 1

namespace="${DOCKERHUB_NAMESPACE:-diyorbek7377}" version=""
while (($#)); do
  case $1 in
    --namespace) namespace=$2; shift 2 ;;
    --version) version=$2; shift 2 ;;
    -h | --help) sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1" 64 ;;
  esac
done
[[ -z $version || $version =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "--version must be vMAJOR.MINOR.PATCH" 64

[[ -z $(git status --porcelain) ]] || die "working tree is not clean; publish only committed, QA-tested source"
commit=$(git rev-parse HEAD)
summary="artifacts/qa/$commit/summary.json"
[[ -f $summary ]] || die "no QA evidence for $commit ($summary); run make qa on this commit first"
grep -q '"result": "pass"' "$summary" || die "QA did not pass for $commit (see artifacts/qa/$commit/summary.md)"
grep -q '"release_evidence": true' "$summary" || die "QA evidence for $commit is not release evidence (dirty run)"
docker buildx version >/dev/null 2>&1 || die "docker buildx is required"

images=("web|frontend/Dockerfile|web" "backend|backend/Dockerfile|runtime" "inference|inference/Dockerfile|runtime")
for spec in "${images[@]}"; do
  IFS='|' read -r name _ _ <<<"$spec"
  if docker buildx imagetools inspect "$namespace/dreamdrive-$name:sha-$commit" >/dev/null 2>&1; then
    die "$namespace/dreamdrive-$name:sha-$commit already exists; immutable tags are never overwritten"
  fi
done

build_version=${version:-sha-$commit}
for spec in "${images[@]}"; do
  IFS='|' read -r name dockerfile target <<<"$spec"
  tags=(-t "$namespace/dreamdrive-$name:sha-$commit")
  [[ -n $version ]] && tags+=(-t "$namespace/dreamdrive-$name:$version")
  log "building + pushing $namespace/dreamdrive-$name:sha-$commit ($dockerfile, target $target)"
  docker buildx build --platform linux/amd64 -f "$dockerfile" --target "$target" \
    --build-arg "GIT_COMMIT=$commit" --build-arg "BUILD_VERSION=$build_version" --build-arg BUILD_DIRTY=0 \
    --provenance=mode=max --sbom=true "${tags[@]}" --push .
done

bash "$DD_REPO_ROOT/scripts/release-manifest.sh" --commit "$commit" --namespace "$namespace"
log "published $commit. Next: make smoke-published RELEASE_ENV=artifacts/release/$commit/release.env UPLOAD=<clip.mp4>"
