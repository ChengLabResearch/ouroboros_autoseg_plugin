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
  REUSE_STAGED_CHECKPOINT=1     Reuse medical_sam3.pt already in VOLUME_NAME.
  HOST_PORT=18788               Host port mapped to backend port 8686.
  VOLUME_NAME=ouroboros-volume   Docker volume used as the plugin shared volume.
  OUTPUT_NAME=mask_stack.tif     Output filename written under Segmentation/.
  OUTPUT_DIR=/path/out           Also copy the output mask stack to this host dir.
  ARTIFACT_DIR=/path/artifacts   Store revisions, telemetry, and bounded backend logs here.
  TELEMETRY_INTERVAL_SECONDS=1  Sampling interval for GPU, RSS, and elapsed-time CSV rows.
  SAM3_VIDEO_STATE_PROFILE=cpu-offload  State profile: gpu-resident (B) or cpu-offload (C).
  SAM3_VIDEO_FEATURE_CACHE_ENTRIES=1    Feature-cache capacity benchmark control: 1 or 2.
  SAM3_TRACKER_TRIM_PAST_NON_COND_MEM=true  Enable the mask-memory trim control.
  SAM3_MAX_NON_COND_TRACKER_STATES=32       Opt in to bounded non-conditioning history.
  SAM3_VIDEO_HOTSTART_DELAY=4               Opt in to a bounded hotstart certification control.
  SAM3_COMPUTE_DTYPE=f32                     Model compute dtype: f32 or f16.
  SAM3_RETAINED_STATE_DTYPE=bf16             Retained mask-memory dtype: f32 or bf16.
  TIFF_VALIDATOR_PYTHON=python3  Python interpreter with tifffile and numpy installed.
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

capture_backend_log() {
  [[ -n "${ARTIFACT_DIR:-}" ]] || return 0
  docker logs --tail "${LOG_TAIL_LINES}" "${CONTAINER_NAME}" \
    >"${ARTIFACT_DIR}/backend.log" 2>&1 || true
}

telemetry_loop() {
  local start_epoch="$1"
  while docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null | grep -qx true; do
    local now elapsed gpu rss
    now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    elapsed=$(( $(date +%s) - start_epoch ))
    gpu="$(nvidia-smi \
      --id="${CUDA_DEVICE_ORDINAL}" \
      --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
    rss="$(docker stats --no-stream --format '{{.MemUsage}}' "${CONTAINER_NAME}" 2>/dev/null || true)"
    printf '%s,%s,%s,"%s"\n' "${now}" "${elapsed}" "${gpu}" "${rss}" \
      >>"${TELEMETRY_CSV}"
    sleep "${TELEMETRY_INTERVAL_SECONDS}"
  done
}

cleanup() {
  local status=$?
  if [[ -n "${TELEMETRY_PID:-}" ]]; then
    kill "${TELEMETRY_PID}" >/dev/null 2>&1 || true
    wait "${TELEMETRY_PID}" 2>/dev/null || true
  fi
  capture_backend_log
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
CANDLE_SAM3_COMMIT="${CANDLE_SAM3_COMMIT:-a92008576b6d8618b2182c577832399035a6fcb5}"
PLUGIN_GIT_SHA="${PLUGIN_GIT_SHA:-$(git -C "${BACKEND_DIR}/.." rev-parse HEAD)}"
PLUGIN_DIRTY="$(git -C "${BACKEND_DIR}/.." status --porcelain | awk 'NF { found=1 } END { print found ? "true" : "false" }')"
CHECKPOINT_URL="${CHECKPOINT_URL:-${DEFAULT_CHECKPOINT_URL}}"
CHECKPOINT_CACHE="${CHECKPOINT_CACHE:-${XDG_CACHE_HOME:-${HOME}/.cache}/ouroboros-autoseg/medical_sam3_checkpoint_3D.pt}"
HOST_PORT="${HOST_PORT:-18788}"
VOLUME_NAME="${VOLUME_NAME:-ouroboros-volume}"
CONTAINER_NAME="${CONTAINER_NAME:-ouroboros-autoseg-video-smoke}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-10}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-7200}"
TELEMETRY_INTERVAL_SECONDS="${TELEMETRY_INTERVAL_SECONDS:-1}"
CUDA_DEVICE_ORDINAL="${CUDA_DEVICE_ORDINAL:-0}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-500}"
OVERLAY_ANNOTATION_POINTS="${OVERLAY_ANNOTATION_POINTS:-false}"
REUSE_STAGED_CHECKPOINT="${REUSE_STAGED_CHECKPOINT:-0}"
SAM3_VIDEO_STATE_PROFILE="${SAM3_VIDEO_STATE_PROFILE:-cpu-offload}"
SAM3_VIDEO_FEATURE_CACHE_ENTRIES="${SAM3_VIDEO_FEATURE_CACHE_ENTRIES:-1}"
SAM3_TRACKER_TRIM_PAST_NON_COND_MEM="${SAM3_TRACKER_TRIM_PAST_NON_COND_MEM:-true}"
SAM3_MAX_NON_COND_TRACKER_STATES="${SAM3_MAX_NON_COND_TRACKER_STATES:-}"
SAM3_VIDEO_HOTSTART_DELAY="${SAM3_VIDEO_HOTSTART_DELAY:-0}"
SAM3_COMPUTE_DTYPE="${SAM3_COMPUTE_DTYPE:-f32}"
SAM3_RETAINED_STATE_DTYPE="${SAM3_RETAINED_STATE_DTYPE:-bf16}"
if [[ -x "${BACKEND_DIR}/.venv/bin/python" ]]; then
  DEFAULT_TIFF_VALIDATOR_PYTHON="${BACKEND_DIR}/.venv/bin/python"
else
  DEFAULT_TIFF_VALIDATOR_PYTHON="python3"
fi
TIFF_VALIDATOR_PYTHON="${TIFF_VALIDATOR_PYTHON:-${DEFAULT_TIFF_VALIDATOR_PYTHON}}"

INPUT_DIR="$(dirname -- "${INPUT_STACK}")"
INPUT_NAME="$(basename -- "${INPUT_STACK}")"
INPUT_STEM="${INPUT_NAME%.*}"
OUTPUT_NAME="${OUTPUT_NAME:-${INPUT_STEM}_video_smoke_mask.tif}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${OUTPUT_DIR:-/tmp/autoseg-smoke-${INPUT_STEM}}}"
mkdir -p "${ARTIFACT_DIR}"
TELEMETRY_CSV="${ARTIFACT_DIR}/telemetry.csv"
printf '%s\n' 'timestamp_utc,elapsed_seconds,gpu_index,gpu_utilization_percent,gpu_memory_used_mib,gpu_memory_total_mib,container_memory_usage' \
  >"${TELEMETRY_CSV}"

if [[ "${REUSE_STAGED_CHECKPOINT}" == "1" ]]; then
  CHECKPOINT_PATH=""
  CHECKPOINT_NAME="medical_sam3.pt"
elif [[ -n "${CHECKPOINT_PATH:-}" ]]; then
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

if [[ "${REUSE_STAGED_CHECKPOINT}" != "1" ]]; then
  CHECKPOINT_DIR="$(dirname -- "${CHECKPOINT_PATH}")"
  CHECKPOINT_NAME="$(basename -- "${CHECKPOINT_PATH}")"
fi

if [[ "${BUILD_IMAGE}" != "0" ]]; then
  log "Building CUDA backend image: ${BACKEND_IMAGE}"
  docker build \
    -f "${BACKEND_DIR}/Dockerfile" \
    --target cuda-runtime \
    --build-arg CANDLE_FEATURES=cuda \
    --build-arg CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP}" \
    --build-arg CANDLE_SAM3_COMMIT="${CANDLE_SAM3_COMMIT}" \
    --build-arg PLUGIN_GIT_SHA="${PLUGIN_GIT_SHA}" \
    -t "${BACKEND_IMAGE}" \
    "${BACKEND_DIR}"
else
  docker image inspect "${BACKEND_IMAGE}" >/dev/null 2>&1 \
    || die "Docker image not found or Docker is not accessible: ${BACKEND_IMAGE}"
fi

docker volume create "${VOLUME_NAME}" >/dev/null
if [[ "${REUSE_STAGED_CHECKPOINT}" == "1" ]]; then
  CHECKPOINT_SHA256="$(
    docker run --rm \
      --entrypoint /bin/sh \
      -v "${VOLUME_NAME}:/volume" \
      "${BACKEND_IMAGE}" \
      -ceu 'sha256sum /volume/sam3-segmentation/chkpts/medical_sam3.pt' \
      | awk '{print $1}'
  )"
else
  CHECKPOINT_SHA256="$(sha256sum "${CHECKPOINT_PATH}" | awk '{print $1}')"
fi

cat >"${ARTIFACT_DIR}/revisions.env" <<EOF
plugin_sha=${PLUGIN_GIT_SHA}
plugin_dirty=${PLUGIN_DIRTY}
candle_sha=${CANDLE_SAM3_COMMIT}
input_sha256=$(sha256sum "${INPUT_STACK}" | awk '{print $1}')
checkpoint_sha256=${CHECKPOINT_SHA256}
image_id=$(docker image inspect --format '{{.Id}}' "${BACKEND_IMAGE}")
cuda_compute_cap=${CUDA_COMPUTE_CAP}
cuda_device_ordinal=${CUDA_DEVICE_ORDINAL}
sam3_video_state_profile=${SAM3_VIDEO_STATE_PROFILE}
sam3_video_feature_cache_entries=${SAM3_VIDEO_FEATURE_CACHE_ENTRIES}
sam3_tracker_trim_past_non_cond_mem=${SAM3_TRACKER_TRIM_PAST_NON_COND_MEM}
sam3_max_non_cond_tracker_states=${SAM3_MAX_NON_COND_TRACKER_STATES:-unbounded}
sam3_video_hotstart_delay=${SAM3_VIDEO_HOTSTART_DELAY}
sam3_compute_dtype=${SAM3_COMPUTE_DTYPE}
sam3_retained_state_dtype=${SAM3_RETAINED_STATE_DTYPE}
gpu=$(nvidia-smi --id="${CUDA_DEVICE_ORDINAL}" --query-gpu=name --format=csv,noheader 2>/dev/null || printf unavailable)
driver_version=$(nvidia-smi --id="${CUDA_DEVICE_ORDINAL}" --query-gpu=driver_version --format=csv,noheader 2>/dev/null || printf unavailable)
cuda_runtime=$(docker run --rm --entrypoint /bin/sh "${BACKEND_IMAGE}" -c 'printf %s "${CUDA_VERSION:-unknown}"')
EOF

if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
  log "Checking Docker GPU access with ${BACKEND_IMAGE}"
  docker run --rm --gpus all --entrypoint nvidia-smi "${BACKEND_IMAGE}" >/dev/null
fi

log "Staging input and checkpoint in Docker volume: ${VOLUME_NAME}"
if [[ "${REUSE_STAGED_CHECKPOINT}" == "1" ]]; then
  docker run --rm \
    --entrypoint /bin/sh \
    -v "${VOLUME_NAME}:/volume" \
    -v "${INPUT_DIR}:/input:ro" \
    -e INPUT_NAME="${INPUT_NAME}" \
    -e OUTPUT_NAME="${OUTPUT_NAME}" \
    "${BACKEND_IMAGE}" \
    -ceu '
      plugin_root=/volume/sam3-segmentation
      test -s "${plugin_root}/chkpts/medical_sam3.pt"
      mkdir -p "${plugin_root}/Segmentation"
      rm -f "${plugin_root}/${INPUT_NAME}" "${plugin_root}/Segmentation/${OUTPUT_NAME}"
      cp "/input/${INPUT_NAME}" "${plugin_root}/${INPUT_NAME}"
    '
else
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
fi

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
  -e CUDA_DEVICE_ORDINAL="${CUDA_DEVICE_ORDINAL}" \
  -e SAM3_VIDEO_STATE_PROFILE="${SAM3_VIDEO_STATE_PROFILE}" \
  -e SAM3_VIDEO_FEATURE_CACHE_ENTRIES="${SAM3_VIDEO_FEATURE_CACHE_ENTRIES}" \
  -e SAM3_TRACKER_TRIM_PAST_NON_COND_MEM="${SAM3_TRACKER_TRIM_PAST_NON_COND_MEM}" \
  -e SAM3_MAX_NON_COND_TRACKER_STATES="${SAM3_MAX_NON_COND_TRACKER_STATES}" \
  -e SAM3_VIDEO_HOTSTART_DELAY="${SAM3_VIDEO_HOTSTART_DELAY}" \
  -e SAM3_COMPUTE_DTYPE="${SAM3_COMPUTE_DTYPE}" \
  -e SAM3_RETAINED_STATE_DTYPE="${SAM3_RETAINED_STATE_DTYPE}" \
  --add-host host.docker.internal:host-gateway \
  "${BACKEND_IMAGE}" >/dev/null

RUN_START_EPOCH="$(date +%s)"
telemetry_loop "${RUN_START_EPOCH}" &
TELEMETRY_PID=$!

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
      capture_backend_log
      tail -n "${LOG_TAIL_LINES}" "${ARTIFACT_DIR}/backend.log" >&2 || true
      die "Smoke job failed"
      ;;
  esac
done

[[ "${JOB_STATUS:-}" == "completed" ]] || {
  capture_backend_log
  tail -n "${LOG_TAIL_LINES}" "${ARTIFACT_DIR}/backend.log" >&2 || true
  die "Smoke job timed out after ${TIMEOUT_SECONDS}s"
}

capture_backend_log
grep -F 'device=cuda' "${ARTIFACT_DIR}/backend.log" >/dev/null \
  || die "Backend log does not prove CUDA model loading"
grep -F "cuda_ordinal=${CUDA_DEVICE_ORDINAL}" "${ARTIFACT_DIR}/backend.log" >/dev/null \
  || die "Backend log does not prove the requested CUDA ordinal"
grep -F "plugin_sha=${PLUGIN_GIT_SHA}" "${ARTIFACT_DIR}/backend.log" >/dev/null \
  || die "Backend log does not contain the expected plugin SHA"
grep -F "candle_sha=${CANDLE_SAM3_COMMIT}" "${ARTIFACT_DIR}/backend.log" >/dev/null \
  || die "Backend log does not contain the expected Candle SHA"

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

"${TIFF_VALIDATOR_PYTHON}" -c 'import numpy, tifffile' >/dev/null 2>&1 \
  || die "TIFF validation requires numpy and tifffile in ${TIFF_VALIDATOR_PYTHON}"
"${TIFF_VALIDATOR_PYTHON}" - "${INPUT_STACK}" "${VALIDATION_DIR}/${OUTPUT_NAME}" <<'PY'
import sys
import numpy
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
with tifffile.TiffFile(sys.argv[2]) as tif:
    for index, page in enumerate(tif.pages):
        values = page.asarray()
        if values.dtype != "uint8":
            raise SystemExit(f"output page {index} has dtype {values.dtype}, expected uint8")
        unique = set(int(value) for value in numpy.unique(values))
        if not unique.issubset({0, 255}):
            raise SystemExit(f"output page {index} is not binary: {sorted(unique)!r}")
print(f"[smoke] output geometry verified: frames={output[0]}, height={output[1]}, width={output[2]}")
PY

if [[ -n "${OUTPUT_DIR:-}" ]]; then
  mkdir -p "${OUTPUT_DIR}"
  cp "${VALIDATION_DIR}/${OUTPUT_NAME}" "${OUTPUT_DIR}/${OUTPUT_NAME}"
  log "Copied output to ${OUTPUT_DIR}/${OUTPUT_NAME}"
fi

log "GPU biological-stack smoke completed successfully; artifacts=${ARTIFACT_DIR}"
