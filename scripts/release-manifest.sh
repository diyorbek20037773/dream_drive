#!/usr/bin/env bash
# Resolve the registry digests of a published DreamDrive release and write the deploy inputs:
#   <out>/release.env            GIT_COMMIT, *_IMAGE_DIGEST, GATEWAY_IMAGE, REDIS_IMAGE (input of deploy.sh)
#   <out>/release-manifest.json  the same plus model identity and supported platforms
#
#   scripts/release-manifest.sh --commit <40-hex> [--namespace diyorbek7377] [--out artifacts/release/<commit>]
#                               [--with-gpu]
# Digests are read from the registry (docker buildx imagetools inspect), never typed by hand. Images must have
# been pushed as <namespace>/dreamdrive-{web,backend,inference}:sha-<commit> by the release workflow.
set -Eeuo pipefail
DD_LOG_TAG=release-manifest
# shellcheck source=deploy/lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../deploy/lib/common.sh"

commit="" namespace="${DOCKERHUB_NAMESPACE:-diyorbek7377}" out="" with_gpu=0
gateway_tag="caddy:2.10-alpine" redis_tag="redis:7.4-alpine"
while (($#)); do
  case $1 in
    --commit) commit=$2; shift 2 ;;
    --namespace) namespace=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --with-gpu) with_gpu=1; shift ;;
    -h | --help) sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1" 64 ;;
  esac
done
is_git_commit "$commit" || die "--commit must be a full 40-hex commit" 64
out=${out:-$DD_REPO_ROOT/artifacts/release/$commit}
mkdir -p "$out"

digest_of() {
  local ref=$1 d
  d=$(docker buildx imagetools inspect "$ref" --format '{{.Manifest.Digest}}') || die "cannot inspect $ref"
  is_sha256_digest "$d" || die "unexpected digest for $ref: $d"
  printf '%s' "$d"
}

web=$(digest_of "$namespace/dreamdrive-web:sha-$commit")
backend=$(digest_of "$namespace/dreamdrive-backend:sha-$commit")
inference=$(digest_of "$namespace/dreamdrive-inference:sha-$commit")
gateway=$(digest_of "$gateway_tag")
redis=$(digest_of "$redis_tag")
gpu_line="" gpu_json="null"
if ((with_gpu)); then
  gpu=$(digest_of "$namespace/dreamdrive-inference-gpu:sha-$commit")
  gpu_line="INFERENCE_GPU_IMAGE_DIGEST=$gpu"
  gpu_json="\"$namespace/dreamdrive-inference-gpu@$gpu\""
fi
model_sha=$(sed -n 's/^ *"checkpoint_sha256": "\([0-9a-f]\{64\}\)".*/\1/p' "$DD_REPO_ROOT/config/models.resolved.yaml" | head -n 1)
is_sha256_digest "sha256:$model_sha" || die "could not read checkpoint_sha256 from config/models.resolved.yaml"

{
  printf '# DreamDrive release %s (resolved from the registry at %s)\n' "$commit" "$(utc_now)"
  printf 'DOCKERHUB_NAMESPACE=%s\n' "$namespace"
  printf 'GIT_COMMIT=%s\n' "$commit"
  printf 'WEB_IMAGE_DIGEST=%s\n' "$web"
  printf 'BACKEND_IMAGE_DIGEST=%s\n' "$backend"
  printf 'INFERENCE_IMAGE_DIGEST=%s\n' "$inference"
  printf 'GATEWAY_IMAGE=%s@%s\n' "$gateway_tag" "$gateway"
  printf 'REDIS_IMAGE=%s@%s\n' "$redis_tag" "$redis"
  if [[ -n $gpu_line ]]; then printf '%s\n' "$gpu_line"; fi
} >"$out/release.env"

cat >"$out/release-manifest.json" <<JSON
{
  "schema": "dreamdrive-release/1",
  "git_commit": "$commit",
  "resolved_utc": "$(utc_now)",
  "platforms": ["linux/amd64"],
  "images": {
    "web": "$namespace/dreamdrive-web@$web",
    "backend": "$namespace/dreamdrive-backend@$backend",
    "inference": "$namespace/dreamdrive-inference@$inference",
    "inference_gpu": $gpu_json
  },
  "tags": {
    "web": "$namespace/dreamdrive-web:sha-$commit",
    "backend": "$namespace/dreamdrive-backend:sha-$commit",
    "inference": "$namespace/dreamdrive-inference:sha-$commit"
  },
  "third_party": {
    "gateway": "$gateway_tag@$gateway",
    "redis": "$redis_tag@$redis"
  },
  "model": {
    "id": "yolopv2",
    "checkpoint_sha256": "$model_sha",
    "source": "config/models.resolved.yaml (weights are downloaded by model-init, not shipped in images)"
  }
}
JSON
log "wrote $out/release.env and $out/release-manifest.json"
printf '%s\n' "$out/release.env"
