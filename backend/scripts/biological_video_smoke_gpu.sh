#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_CHECKPOINT_URL="https://huggingface.co/ChongCong/Medical-SAM3/resolve/main/checkpoint_3D.pt"
DEFAULT_IMAGE="ouroboros-autoseg-backend:video-smoke-cuda"

usage() {
  cat <<'USAGE'
Run the Medical SAM3 biological-stack video smoke on the CUDA backend.

Usage:
  INPUT_STACK=/path/to/straightened-stack.tif backend/scripts/biological_video_smoke_gpu.sh
  backend/scripts/biological_video_smoke_gpu.sh /path/to/straightened-stack.tif

Common options:
  BUILD_IMAGE=0                 Use BACKEND_IMAGE instead of building this checkout.
  BACKEND_IMAGE=name:tag         CUDA backend image to build/run.
  CUDA_COMPUTE_CAP=75            CUDA compute capability used for the local image build.
  CHECKPOINT_PATH=/path/model.pt Use an existing Medical SAM3 3D checkpoint.
  CHECKPOINT_URL=https://...     Override the default checkpoint_3D.pt URL.
  HOST_PORT=18788               Host port mapped to backend port 8686.
  VOLUME_NAME=ouroboros-volume   Docker volume used as the plugin shared volume.
  OUTPUT_NAME=mask_stack.tif     Output filename written under Segmentation/.
  OUTPUT_DIR=/path/out           Also copy the output mask stack to this host dir.
  KEEP_CONTAINER=1               Leave the backend container running after the script exits.

The script builds/runs the CUDA Docker target with --gpus all, stages the input
stack and Medical SAM3 3D checkpoint into the Docker volume, submits a
VideoPredictor job, polls it to completion, and verifies that the mask stack was
written. If Python tifffile is available, it also checks output geometry.
USAGE
}

log() {
  printf '[smoke] %s\n' "$*"
}

die() {
  printf '[smoke] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

resolve_path() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
}

json_value() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
value = payload
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}

job_progress() {
  python3 - "$1" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
steps = payload.get("steps", [])
summary = ", ".join(
    f"{step.get('name', '?')}={step.get('progress', '?')}%" for step in steps
)
print(summary or "no step progress")
PY
}

cleanup() {
  local status=$?
  if [[ -n "${VALIDATION_DIR:-}" && -d "${VALIDATION_DIR}" ]]; then
    rm -rf "${VALIDATION_DIR}"
  fi
  if [[ "${KEEP_CONTAINER:-0}" != "1" ]]; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi
  exit "${status}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

INPUT_STACK="${INPUT_STACK:-${1:-}}"
[[ -n "${INPUT_STACK}" ]] || {
  usage
  exit 2
}

require_command docker
require_command curl
require_command python3

INPUT_STACK="$(resolve_path "${INPUT_STACK}")"
[[ -f "${INPUT_STACK}" ]] || die "Input stack does not exist: ${INPUT_STACK}"

BACKEND_IMAGE="${BACKEND_IMAGE:-${DEFAULT_IMAGE}}"
BUILD_IMAGE="${BUILD_IMAGE:-1}"
CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-75}"
CANDLE_SAM3_COMMIT="${CANDLE_SAM3_COMMIT:-770d20ca8db4f834ba4c89c845bca196fbfc97ea}"
CHECKPOINT_URL="${CHECKPOINT_URL:-${DEFAULT_CHECKPOINT_URL}}"
CHECKPOINT_CACHE="${CHECKPOINT_CACHE:-${XDG_CACHE_HOME:-${HOME}/.cache}/ouroboros-autoseg/medical_sam3_checkpoint_3D.pt}"
HOST_PORT="${HOST_PORT:-18788}"
VOLUME_NAME="${VOLUME_NAME:-ouroboros-volume}"
CONTAINER_NAME="${CONTAINER_NAME:-ouroboros-autoseg-video-smoke}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-10}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-7200}"
OVERLAY_ANNOTATION_POINTS="${OVERLAY_ANNOTATION_POINTS:-false}"

INPUT_DIR="$(dirname -- "${INPUT_STACK}")"
INPUT_NAME="$(basename -- "${INPUT_STACK}")"
INPUT_STEM="${INPUT_NAME%.*}"
OUTPUT_NAME="${OUTPUT_NAME:-${INPUT_STEM}_video_smoke_mask.tif}"

if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  CHECKPOINT_PATH="$(resolve_path "${CHECKPOINT_PATH}")"
  [[ -f "${CHECKPOINT_PATH}" ]] || die "Checkpoint does not exist: ${CHECKPOINT_PATH}"
else
  mkdir -p "$(dirname -- "${CHECKPOINT_CACHE}")"
  if [[ ! -f "${CHECKPOINT_CACHE}" ]]; then
    log "Downloading Medical SAM3 3D checkpoint"
    curl_args=(-L --fail --continue-at - --output "${CHECKPOINT_CACHE}" "${CHECKPOINT_URL}")
    if [[ -n "${HF_TOKEN:-}" ]]; then
      curl_args=(-H "Authorization: Bearer ${HF_TOKEN}" "${curl_args[@]}")
    fi
    curl "${curl_args[@]}"
  else
    log "Using cached checkpoint: ${CHECKPOINT_CACHE}"
  fi
  CHECKPOINT_PATH="${CHECKPOINT_CACHE}"
fi

CHECKPOINT_DIR="$(dirname -- "${CHECKPOINT_PATH}")"
CHECKPOINT_NAME="$(basename -- "${CHECKPOINT_PATH}")"

if [[ "${BUILD_IMAGE}" != "0" ]]; then
  log "Building CUDA backend image: ${BACKEND_IMAGE}"
  docker build \
    -f "${BACKEND_DIR}/Dockerfile" \
    --target cuda-runtime \
    --build-arg CANDLE_FEATURES=cuda \
    --build-arg CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP}" \
    --build-arg CANDLE_SAM3_COMMIT="${CANDLE_SAM3_COMMIT}" \
    -t "${BACKEND_IMAGE}" \
    "${BACKEND_DIR}"
else
  docker image inspect "${BACKEND_IMAGE}" >/dev/null 2>&1 \
    || die "Docker image not found or Docker is not accessible: ${BACKEND_IMAGE}"
fi

if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
  log "Checking Docker GPU access with ${BACKEND_IMAGE}"
  docker run --rm --gpus all --entrypoint nvidia-smi "${BACKEND_IMAGE}" >/dev/null
fi

log "Staging input and checkpoint in Docker volume: ${VOLUME_NAME}"
docker volume create "${VOLUME_NAME}" >/dev/null
docker run --rm \
  --entrypoint /bin/sh \
  -v "${VOLUME_NAME}:/volume" \
  -v "${INPUT_DIR}:/input:ro" \
  -v "${CHECKPOINT_DIR}:/checkpoint:ro" \
  -e INPUT_NAME="${INPUT_NAME}" \
  -e CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
  -e OUTPUT_NAME="${OUTPUT_NAME}" \
  "${BACKEND_IMAGE}" \
  -ceu '
    plugin_root=/volume/sam3-segmentation
    mkdir -p "${plugin_root}/chkpts" "${plugin_root}/Segmentation"
    rm -f "${plugin_root}/${INPUT_NAME}" "${plugin_root}/Segmentation/${OUTPUT_NAME}"
    cp "/input/${INPUT_NAME}" "${plugin_root}/${INPUT_NAME}"
    cp "/checkpoint/${CHECKPOINT_NAME}" "${plugin_root}/chkpts/medical_sam3.pt"
  '

trap cleanup EXIT

log "Starting CUDA backend container: ${CONTAINER_NAME}"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  -p "127.0.0.1:${HOST_PORT}:8686" \
  -v "${VOLUME_NAME}:/ouroboros-volume" \
  -e VOLUME_MOUNT_PATH=/ouroboros-volume \
  -e VOLUME_SERVER_URL=http://host.docker.internal:3001 \
  --add-host host.docker.internal:host-gateway \
  "${BACKEND_IMAGE}" >/dev/null

log "Waiting for backend on http://127.0.0.1:${HOST_PORT}"
for _ in $(seq 1 60); do
  if curl -fs "http://127.0.0.1:${HOST_PORT}/" >/dev/null 2>&1; then
    break
  fi
  if ! docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null | grep -qx true; then
    docker logs --tail 200 "${CONTAINER_NAME}" >&2 || true
    die "Backend container exited before becoming ready"
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:${HOST_PORT}/" >/dev/null || die "Backend did not become ready"

log "Submitting VideoPredictor smoke job"
REQUEST_BODY="$(
  python3 - "${INPUT_NAME}" "${OUTPUT_NAME}" "${OVERLAY_ANNOTATION_POINTS}" <<'PY'
import json
import sys

input_name, output_name, overlay = sys.argv[1:4]
print(json.dumps({
    "file_path": f"/smoke/{input_name}",
    "output_file": f"/smoke/{output_name}",
    "model_type": "medical_sam3",
    "predictor_type": "VideoPredictor",
    "overlay_annotation_points": overlay.lower() == "true",
}))
PY
)"
SUBMIT_RESPONSE="$(
  curl -fsS \
    -X POST "http://127.0.0.1:${HOST_PORT}/process-stack" \
    -H 'Content-Type: application/json' \
    --data-binary "${REQUEST_BODY}"
)"
JOB_ID="$(json_value "${SUBMIT_RESPONSE}" "job_id")"
log "Job accepted: ${JOB_ID}"

deadline=$((SECONDS + TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  sleep "${POLL_INTERVAL_SECONDS}"
  STATUS_RESPONSE="$(curl -fsS "http://127.0.0.1:${HOST_PORT}/status/${JOB_ID}")"
  JOB_STATUS="$(json_value "${STATUS_RESPONSE}" "status")"
  log "status=${JOB_STATUS}; $(job_progress "${STATUS_RESPONSE}")"
  case "${JOB_STATUS}" in
    completed)
      break
      ;;
    error)
      docker logs --tail 200 "${CONTAINER_NAME}" >&2 || true
      die "Smoke job failed"
      ;;
  esac
done

[[ "${JOB_STATUS:-}" == "completed" ]] || {
  docker logs --tail 200 "${CONTAINER_NAME}" >&2 || true
  die "Smoke job timed out after ${TIMEOUT_SECONDS}s"
}

log "Verifying output in Docker volume"
docker run --rm \
  --entrypoint /bin/sh \
  -v "${VOLUME_NAME}:/volume" \
  -e OUTPUT_NAME="${OUTPUT_NAME}" \
  "${BACKEND_IMAGE}" \
  -ceu 'test -s "/volume/sam3-segmentation/Segmentation/${OUTPUT_NAME}"'

VALIDATION_DIR="$(mktemp -d)"
docker run --rm \
  --entrypoint /bin/sh \
  -v "${VOLUME_NAME}:/volume" \
  -v "${VALIDATION_DIR}:/out" \
  -e OUTPUT_NAME="${OUTPUT_NAME}" \
  "${BACKEND_IMAGE}" \
  -ceu 'cp "/volume/sam3-segmentation/Segmentation/${OUTPUT_NAME}" "/out/${OUTPUT_NAME}"'

if python3 -c 'import tifffile' >/dev/null 2>&1; then
  python3 - "${INPUT_STACK}" "${VALIDATION_DIR}/${OUTPUT_NAME}" <<'PY'
import sys
import tifffile


def geometry(path):
    with tifffile.TiffFile(path) as tif:
        if not tif.pages:
            raise SystemExit(f"{path} has no TIFF pages")
        pages = len(tif.pages)
        shape = tif.pages[0].shape
        if len(shape) < 2:
            raise SystemExit(f"{path} first page has unsupported shape {shape!r}")
        return pages, int(shape[0]), int(shape[1])


source = geometry(sys.argv[1])
output = geometry(sys.argv[2])
if source != output:
    raise SystemExit(f"output geometry {output} does not match input {source}")
print(f"[smoke] output geometry verified: frames={output[0]}, height={output[1]}, width={output[2]}")
PY
else
  log "Python tifffile is not installed; skipped output geometry validation"
fi

if [[ -n "${OUTPUT_DIR:-}" ]]; then
  mkdir -p "${OUTPUT_DIR}"
  cp "${VALIDATION_DIR}/${OUTPUT_NAME}" "${OUTPUT_DIR}/${OUTPUT_NAME}"
  log "Copied output to ${OUTPUT_DIR}/${OUTPUT_NAME}"
fi

log "GPU biological-stack smoke completed successfully"
