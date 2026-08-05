#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Certify an exact CUDA candidate digest with the checkpoint-backed GPU smoke.

Usage:
  BACKEND_IMAGE=ghcr.io/chenglabresearch/ouroboros-autoseg-backend@sha256:<digest> \
  INPUT_STACK=/path/to/straightened-stack.tif \
    backend/scripts/certify_cuda_candidate_gpu.sh

The smoke always runs with BUILD_IMAGE=0. After it passes, the script records
the exact digest as gpu-certified-<digest> in the same registry repository so
the main workflow can promote it without rebuilding. Set
PUBLISH_GPU_CERTIFICATION=0 to run the gate without publishing that marker.
All biological_video_smoke_gpu.sh environment controls remain available.
USAGE
}

die() {
  printf '[certify] ERROR: %s\n' "$*" >&2
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

BACKEND_IMAGE="${BACKEND_IMAGE:-}"
[[ -n "${BACKEND_IMAGE}" ]] || {
  usage
  exit 2
}
[[ "${BACKEND_IMAGE}" =~ @sha256:([0-9a-f]{64})$ ]] \
  || die "BACKEND_IMAGE must identify an exact @sha256 digest"
digest_hex="${BASH_REMATCH[1]}"
[[ -n "${INPUT_STACK:-}" ]] || die "INPUT_STACK is required"
image_without_digest="${BACKEND_IMAGE%@sha256:*}"
image_name="${image_without_digest##*/}"
if [[ "${image_name}" == *:* ]]; then
  image_repository="${image_without_digest%:*}"
else
  image_repository="${image_without_digest}"
fi
canonical_image="${image_repository}@sha256:${digest_hex}"

BUILD_IMAGE=0 \
REQUIRE_IMAGE_DIGEST=1 \
BACKEND_IMAGE="${canonical_image}" \
  "${SCRIPT_DIR}/biological_video_smoke_gpu.sh"

if [[ "${PUBLISH_GPU_CERTIFICATION:-1}" == "1" ]]; then
  command -v docker >/dev/null 2>&1 || die "Missing required command: docker"
  certification_tag="${image_repository}:gpu-certified-${digest_hex}"
  printf '[certify] Recording passed digest as %s\n' "${certification_tag}"
  docker buildx imagetools create \
    --tag "${certification_tag}" \
    "${canonical_image}"
  recorded_digest="$(docker buildx imagetools inspect \
    "${certification_tag}" --format '{{.Manifest.Digest}}')"
  [[ "${recorded_digest}" == "sha256:${digest_hex}" ]] \
    || die "Certification marker resolved to ${recorded_digest}, expected sha256:${digest_hex}"
  printf '[certify] GPU certification recorded for %s\n' "${canonical_image}"
else
  printf '[certify] GPU smoke passed; certification marker publication was disabled\n'
fi
