#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_SCRIPT="${SCRIPT_DIR}/biological_video_smoke_gpu.sh"

usage() {
  cat <<'USAGE'
Benchmark the explicit CANDLE-2 low-memory video state profiles.

Usage:
  INPUT_STACK=/path/input.tif CHECKPOINT_PATH=/path/checkpoint.pt \
    backend/scripts/benchmark_low_memory_profiles_gpu.sh
  INPUT_STACK=/path/input.tif REUSE_STAGED_CHECKPOINT=1 \
    VOLUME_NAME=existing-volume backend/scripts/benchmark_low_memory_profiles_gpu.sh

Controls:
  FRAME_COUNTS="32 128 512"            Workloads to generate from the source stack.
  STATE_PROFILES="gpu-resident cpu-offload"  Variant B followed by variant C.
  FEATURE_CACHE_ENTRIES="1 2"          Feature-cache capacities to compare.
  RESULTS_DIR=/path/results             CSV, logs, outputs, and exact revisions.
  BACKEND_IMAGE=name:tag                Image reused for all matrix entries.
  BUILD_IMAGE=1                         Build once before the first matrix entry.
  VOLUME_NAME=candle2-low-memory-profile Shared checkpoint/output Docker volume.
  REUSE_STAGED_CHECKPOINT=1             Avoid another private checkpoint copy.
  AVAILABLE_DISK_GIB_MIN=30             Refuse to start below this host-space floor.
  MAX_NON_COND_TRACKER_STATES=32        Opt-in bounded-history control; empty is unbounded.
  VIDEO_HOTSTART_DELAY=4                Opt-in bounded hotstart control; default 0.

The source TIFF pages are repeated deterministically. Its JSON image description,
including annotation_points, is preserved while the declared stack shape is
updated. This harness intentionally keeps the legacy transformer-owned frame
adapter as the control; https://github.com/den-sq/sam_parity/issues/41 is not
selected into this matrix.
USAGE
}

[[ "${1:-}" != "-h" && "${1:-}" != "--help" ]] || {
  usage
  exit 0
}

INPUT_STACK="${INPUT_STACK:-${1:-}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
[[ -f "${INPUT_STACK}" ]] || { usage >&2; exit 2; }
REUSE_STAGED_CHECKPOINT="${REUSE_STAGED_CHECKPOINT:-0}"
if [[ "${REUSE_STAGED_CHECKPOINT}" != "1" && ! -f "${CHECKPOINT_PATH}" ]]; then
  usage >&2
  exit 2
fi

FRAME_COUNTS="${FRAME_COUNTS:-32 128 512}"
STATE_PROFILES="${STATE_PROFILES:-gpu-resident cpu-offload}"
FEATURE_CACHE_ENTRIES="${FEATURE_CACHE_ENTRIES:-1 2}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/candle2-low-memory-profile}"
BACKEND_IMAGE="${BACKEND_IMAGE:-ouroboros-autoseg-backend:candle2-low-memory-cuda}"
VOLUME_NAME="${VOLUME_NAME:-candle2-low-memory-profile}"
BUILD_IMAGE="${BUILD_IMAGE:-1}"
AVAILABLE_DISK_GIB_MIN="${AVAILABLE_DISK_GIB_MIN:-30}"
TIFF_PYTHON="${TIFF_PYTHON:-python3}"
MAX_NON_COND_TRACKER_STATES="${MAX_NON_COND_TRACKER_STATES:-}"
VIDEO_HOTSTART_DELAY="${VIDEO_HOTSTART_DELAY:-0}"
mkdir -p "${RESULTS_DIR}/inputs" "${RESULTS_DIR}/runs"

available_kib=$(df --output=avail "${RESULTS_DIR}" | tail -1 | tr -d ' ')
minimum_kib=$((AVAILABLE_DISK_GIB_MIN * 1024 * 1024))
if (( available_kib < minimum_kib )); then
  printf 'Refusing benchmark: %s GiB free is below the %s GiB floor.\n' \
    "$((available_kib / 1024 / 1024))" "${AVAILABLE_DISK_GIB_MIN}" >&2
  exit 1
fi

printf '%s\n' \
  'frame_count,state_profile,feature_cache_entries,status,elapsed_seconds,peak_gpu_memory_mib,peak_gpu_utilization_percent,output_sha256' \
  >"${RESULTS_DIR}/summary.csv"

run_index=0
for frame_count in ${FRAME_COUNTS}; do
  generated="${RESULTS_DIR}/inputs/fixture_${frame_count}.tif"
  "${TIFF_PYTHON}" - "${INPUT_STACK}" "${generated}" "${frame_count}" <<'PY'
import json
import sys
import numpy as np
import tifffile

source, output, frame_count = sys.argv[1], sys.argv[2], int(sys.argv[3])
with tifffile.TiffFile(source) as tif:
    frames = tif.asarray()
    description = json.loads(tif.pages[0].description or "{}")
if frames.shape[0] == 0:
    raise SystemExit("source TIFF has no frames")
indices = np.arange(frame_count) % frames.shape[0]
generated = frames[indices]
shape = list(description.get("shape", list(generated.shape)))
shape[0] = frame_count
description["shape"] = shape
tifffile.imwrite(output, generated, description=json.dumps(description, separators=(",", ":")))
PY

  for state_profile in ${STATE_PROFILES}; do
    for feature_cache_entries in ${FEATURE_CACHE_ENTRIES}; do
      run_index=$((run_index + 1))
      run_name="f${frame_count}-${state_profile}-cache${feature_cache_entries}"
      run_dir="${RESULTS_DIR}/runs/${run_name}"
      mkdir -p "${run_dir}"
      status=passed
      if ! INPUT_STACK="${generated}" \
        CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
        BUILD_IMAGE="${BUILD_IMAGE}" \
        BACKEND_IMAGE="${BACKEND_IMAGE}" \
        VOLUME_NAME="${VOLUME_NAME}" \
        REUSE_STAGED_CHECKPOINT="${REUSE_STAGED_CHECKPOINT}" \
        CONTAINER_NAME="candle2-low-memory-${run_name}" \
        HOST_PORT="$((18800 + run_index))" \
        OUTPUT_NAME="${run_name}.tif" \
        ARTIFACT_DIR="${run_dir}" \
        OUTPUT_DIR="${run_dir}" \
        SAM3_VIDEO_STATE_PROFILE="${state_profile}" \
        SAM3_VIDEO_FEATURE_CACHE_ENTRIES="${feature_cache_entries}" \
        SAM3_MAX_NON_COND_TRACKER_STATES="${MAX_NON_COND_TRACKER_STATES}" \
        SAM3_VIDEO_HOTSTART_DELAY="${VIDEO_HOTSTART_DELAY}" \
        "${SMOKE_SCRIPT}"; then
        status=failed
      fi
      BUILD_IMAGE=0
      elapsed=$(awk -F, 'END {print $2+0}' "${run_dir}/telemetry.csv" 2>/dev/null || printf 0)
      peak_memory=$(awk -F, 'NR>1 {gsub(/ /,"",$5); if ($5+0>m) m=$5+0} END {print m+0}' "${run_dir}/telemetry.csv" 2>/dev/null || printf 0)
      peak_util=$(awk -F, 'NR>1 {gsub(/ /,"",$4); if ($4+0>m) m=$4+0} END {print m+0}' "${run_dir}/telemetry.csv" 2>/dev/null || printf 0)
      output_sha=$(sha256sum "${run_dir}/${run_name}.tif" 2>/dev/null | awk '{print $1}' || true)
      printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${frame_count}" "${state_profile}" "${feature_cache_entries}" "${status}" \
        "${elapsed}" "${peak_memory}" "${peak_util}" "${output_sha}" \
        >>"${RESULTS_DIR}/summary.csv"
      [[ "${status}" == passed ]] || exit 1
    done
  done
done
