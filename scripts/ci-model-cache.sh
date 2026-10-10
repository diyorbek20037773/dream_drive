#!/usr/bin/env bash
# Move the verified model between the Docker volume <project>_model-cache and a host directory that
# actions/cache persists (key = checkpoint SHA256 from config/models.resolved.yaml). Layout-agnostic: the whole
# volume content is archived. model-init still re-verifies size + SHA256 on every start, so a restored cache can
# never bypass verification; on a cache miss model-init downloads from the pinned upstream URL.
#
#   scripts/ci-model-cache.sh restore DIR    # DIR/model-cache.tar -> volume (if present)
#   scripts/ci-model-cache.sh save DIR       # volume -> DIR/model-cache.tar, only if it holds the pinned checkpoint
set -Eeuo pipefail
DD_LOG_TAG=ci-model-cache
# shellcheck source=deploy/lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../deploy/lib/common.sh"
# shellcheck source=scripts/tool-images.env
source "$DD_REPO_ROOT/scripts/tool-images.env"

(($# == 2)) || die "usage: ci-model-cache.sh restore|save DIR" 64
action=$1 dir=$2
volume="${DD_PROJECT:-dreamdrive}_model-cache"
model_sha=$(sed -n 's/^ *"checkpoint_sha256": "\([0-9a-f]\{64\}\)".*/\1/p' "$DD_REPO_ROOT/config/models.resolved.yaml" | head -n 1)
is_sha256_digest "sha256:$model_sha" || die "checkpoint_sha256 not found in config/models.resolved.yaml"
mkdir -p "$dir"
dir_host=$(cd "$dir" && { pwd -W 2>/dev/null || pwd; })

case $action in
  restore)
    if [[ ! -f $dir/model-cache.tar ]]; then
      log "no cached model archive in $dir (model-init will download)"
      exit 0
    fi
    docker volume create "$volume" >/dev/null
    docker run --rm --network none -v "$volume:/models" -v "$dir_host:/cache:ro" "$PYTHON_IMAGE" \
      sh -c 'tar -C /models -xf /cache/model-cache.tar && chown -R 10001:10001 /models'
    log "restored model cache into volume $volume (model-init re-verifies it)"
    ;;
  save)
    if docker run --rm --network none -v "$volume:/models:ro" "$PYTHON_IMAGE" \
      sh -c 'find /models -type f -exec sha256sum {} +' | grep -q "^$model_sha "; then
      docker run --rm --network none -v "$volume:/models:ro" -v "$dir_host:/cache" "$PYTHON_IMAGE" \
        tar -C /models -cf /cache/model-cache.tar .
      log "saved verified model ($model_sha) from $volume to $dir"
    else
      rm -rf "$dir"
      log "volume $volume does not hold the pinned checkpoint; nothing cached"
    fi
    ;;
  *) die "unknown action: $action" 64 ;;
esac
