# SAM3 Video Smoke Checks

AUTO-4 keeps the production video path on conservative runtime settings:

- `VideoMemoryProfile::LowMemory`
- no frame or state CPU offload
- no prefetch ahead or behind
- at most two feature cache entries

The unit tests lock those options and validate that video propagation output has
one mask per staged frame, with the same width and height as the straightened
input volume.

Straightened-stack `annotation_points` metadata currently describes one
biological trajectory. The first prompt allocates its SAM3 object ID and every
later z-annotation reuses that ID, including multiple positive points grouped on
one frame. A future multi-object annotation schema must provide explicit track
IDs; independent rows must not implicitly allocate independent objects.

## GPU Biological-Stack Smoke

Use the smoke script for the checkpoint/dataset-dependent portion of AUTO-4. It
builds the CUDA backend target, verifies Docker GPU access, stages the input
stack and Medical SAM3 3D checkpoint into the plugin Docker volume, submits a
`VideoPredictor` request, polls the job, and verifies that the output mask stack
is present.

```bash
INPUT_STACK=local_data/psd_anno_thing.tif \
  backend/scripts/biological_video_smoke_gpu.sh
```

Useful overrides:

```bash
CHECKPOINT_PATH=/path/to/checkpoint_3D.pt \
OUTPUT_DIR=/tmp/autoseg-smoke \
HOST_PORT=18788 \
  backend/scripts/biological_video_smoke_gpu.sh /path/to/straightened-stack.tif
```

The script defaults to
`https://huggingface.co/ChongCong/Medical-SAM3/resolve/main/checkpoint_3D.pt`,
because the video loader needs the tracker tensors present in the 3D Medical
SAM3 checkpoint. A valid smoke run should:

- finish without frame-load panics or out-of-memory errors
- preserve progress updates through polling or UI reconnects
- write an output mask stack with the same frame count, width, and height as the
  straightened input
- keep normal output clean when `overlay_annotation_points` is false

Each run writes a `revisions.env`, bounded `backend.log`, and synchronized
`telemetry.csv` under `ARTIFACT_DIR` (or `OUTPUT_DIR` when set). The telemetry
rows record elapsed time, GPU utilization, GPU memory, and container memory.
The harness also verifies that the model-load log contains the requested CUDA
ordinal plus the exact plugin and Candle revisions, and that every output TIFF
page is `uint8` with binary values only.

Use a separate run with `overlay_annotation_points` enabled when generating
prompt-marker figure artifacts.
