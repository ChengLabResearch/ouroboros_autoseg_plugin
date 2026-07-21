# CANDLE-2 low-memory video profile

Tracking issue: https://github.com/den-sq/sam_parity/issues/33

The temporary production default is variant C: compact low-memory outputs with
tracker state offloaded to CPU, zero frame prefetch, and one cached feature entry.
Variant B keeps tracker state on the GPU and is available as a benchmark control.
It must not become the default until the 512-frame workload demonstrates both a
14 GiB peak-VRAM ceiling and a post-warmup memory plateau.

Bounded tracker history from https://github.com/den-sq/sam_parity/issues/37 is
also opt-in while its mask certification is pending. Set
`SAM3_MAX_NON_COND_TRACKER_STATES=32` for the bounded benchmark control; leave it
empty for compatibility/unbounded mode. Candle rejects a bound smaller than its
effective mask-memory, object-pointer, and refinement windows (16 for Medical
SAM3), always retains prompt/conditioning states, and reports retained state,
output-index, low-resolution-mask, and hotstart-queue metrics separately.

| Setting | Variant B | Variant C (default) |
| --- | --- | --- |
| `SAM3_VIDEO_STATE_PROFILE` | `gpu-resident` | `cpu-offload` |
| `offload_state_to_cpu` | false | true |
| `SAM3_VIDEO_FEATURE_CACHE_ENTRIES` | 1 or 2 | 1 or 2 |
| frame prefetch | 0 | 0 |

The backend logs the selected profile, loaded-frame count, feature-cache count,
and separate CPU/device byte totals for frames, tracker state, and cached output.
The GPU smoke harness also records host/container RSS, CUDA memory, utilization,
elapsed time, exact revisions, and output hashes.

`trim_past_non_cond_mem_for_eval` is enabled as the mask-memory control. This
reduces the non-conditioning mask-memory window; it does **not** bound the full
`tracker_states` map or its F32 `288 x 288` low-resolution masks. Until bounded
tracker history lands in https://github.com/den-sq/sam_parity/issues/37, CPU
offload is therefore a temporary capacity measure rather than a claim that
retained history is bounded.
Set `SAM3_TRACKER_TRIM_PAST_NON_COND_MEM=false` only for the required untrimmed
reference run; the default is `true`, and the selected masks must remain equal.

The Medical production default has `hotstart_delay=0`. Certification can opt in
to a bounded callback-queue control with `SAM3_VIDEO_HOTSTART_DELAY=4`; the
reported current/peak queue frames and bytes remain separate from tracker-history
growth. Production remains at delay zero unless the environment explicitly
selects a different value.

## Benchmark matrix

Run `backend/scripts/benchmark_low_memory_profiles_gpu.sh` with a local Medical
SAM3 checkpoint and representative annotated TIFF. It generates deterministic
32-, 128-, and 512-frame workloads, runs B before C, compares feature-cache
capacity one versus two, and writes per-run logs plus `summary.csv`.
Set `REUSE_STAGED_CHECKPOINT=1` and `VOLUME_NAME` to reuse a private checkpoint
already staged in Docker without creating another 10 GB checkpoint copy.

The matrix deliberately retains the existing transformer-owned frame adapter.
The caller-owned lazy adapter in https://github.com/den-sq/sam_parity/issues/41
is deferred until its preprocessing golden passes. This keeps media-adapter
ownership and process-start effects out of the state-offload attribution.

Output D2H streaming is common to both variants. The difference between B and C
therefore measures repeated tracker-state offload/reload overhead; the external
CUDA telemetry remains the total transfer/control observation until Candle
exposes direction-specific PCIe counters.
