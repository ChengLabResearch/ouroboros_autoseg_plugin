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

The smoke always runs with BUILD_IMAGE=0 and never writes registry state. Its
exit status and ARTIFACT_DIR are the pre-release evidence. All
biological_video_smoke_gpu.sh environment controls remain available.
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
[[ -n "${INPUT_STACK:-}" ]] || die "INPUT_STACK is required"

BUILD_IMAGE=0 \
REQUIRE_IMAGE_DIGEST=1 \
BACKEND_IMAGE="${BACKEND_IMAGE}" \
  "${SCRIPT_DIR}/biological_video_smoke_gpu.sh"

printf '[certify] Exact-digest GPU smoke passed for %s\n' "${BACKEND_IMAGE}"
