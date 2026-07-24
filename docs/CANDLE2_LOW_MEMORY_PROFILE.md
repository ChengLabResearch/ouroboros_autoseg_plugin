# CANDLE-2 low-memory video profile

Tracking issue: https://github.com/den-sq/sam_parity/issues/35

The temporary CANDLE-2.9 candidate default is bounded variant B: F32 compute and
retained state, tracker state resident on the GPU, zero frame prefetch, one
cached feature entry, and at most 32 non-conditioning tracker states. Variant C
remains available as an explicit CPU-offload fallback for systems without the
required GPU headroom.

Bounded tracker history from https://github.com/den-sq/sam_parity/issues/37 is
enabled by default with `SAM3_MAX_NON_COND_TRACKER_STATES=32`. Set the variable
explicitly empty only for compatibility/unbounded reference runs. Candle rejects
a bound smaller than its effective mask-memory, object-pointer, and refinement
windows (16 for Medical SAM3), always retains prompt/conditioning states, and
reports retained state, output-index, low-resolution-mask, and hotstart-queue
metrics separately.

| Setting | Variant B (default) | Variant C (fallback) |
| --- | --- | --- |
| `SAM3_VIDEO_STATE_PROFILE` | `gpu-resident` | `cpu-offload` |
| `offload_state_to_cpu` | false | true |
| `SAM3_VIDEO_FEATURE_CACHE_ENTRIES` | 1 or 2 | 1 or 2 |
| `SAM3_MAX_NON_COND_TRACKER_STATES` | 32 | 32 |
| `SAM3_RETAINED_STATE_DTYPE` | `f32` | `f32` |
| frame prefetch | 0 | 0 |

The backend logs the selected profile, loaded-frame count, feature-cache count,
and separate CPU/device byte totals for frames, tracker state, and cached output.
The GPU smoke harness also records host/container RSS, CUDA memory, utilization,
elapsed time, exact revisions, and output hashes.

`trim_past_non_cond_mem_for_eval` is enabled as the mask-memory control. The
separate 32-state limit bounds the full non-conditioning `tracker_states` map
and its F32 `288 x 288` low-resolution masks; prompt/conditioning states remain
retained.
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

## Provisional default evidence

The exact CANDLE-2.9 candidate produced byte-identical 64-frame F32 output in
two GPU-resident controls and one CPU-offload control. Under matched heat-soaked
conditions, GPU-resident propagation took 326.59 seconds versus 384.43 seconds
for CPU offload (15.0% less time). It retained 33 total tracker states, including
32 non-conditioning states. Peak reported GPU memory was 10,647 MiB, below the
14 GiB gate.

The follow-on 512-frame run retained a flat approximately 10.5 GiB VRAM plateau
through 25% progress, but its speed result was invalidated when the host driver
dropped the GPU to P3/300 MHz and requested a 50 W cap instead of its 90 W
default. This provisional profile selection does not claim that the sustained
throughput acceptance criterion is complete.
