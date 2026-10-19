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

Video propagation is consumed through Candle's streaming callback. Each yielded
CUDA frame is thresholded and copied into its final CPU `FrameMask` slot before
the callback returns; the plugin never retains a full `VideoOutput`. Frame
indices are validated for direction, range, duplicates, and omissions, and the
session is explicitly closed on both success and failure. Low-memory runs log
`cached_output_frames` after propagation and require it to be zero.

The final CPU result is intentionally still resident until TIFF writing
completes. For 4,901 binary 200 x 200 pages, the mask pixel buffers account for
about 187 MiB, plus vector metadata. The TIFF writer currently makes a
comparable transient page-data copy, so full-stack host RSS acceptance must
include both allocations.

## GPU Biological-Stack Smoke

Use the smoke script for the checkpoint/dataset-dependent portion of AUTO-4. It
builds the CUDA backend target, verifies Docker GPU access, and stages the
Medical SAM3 3D checkpoint into the plugin Docker volume. It submits real host
input/output paths to `VideoPredictor`, polls the job, and validates the mask
stack returned to the host through the backend's volume-transfer requests.

The standalone harness starts `smoke_volume_server.py`, a diagnostic fixture
implementing the production `/copy-to-volume` and `/copy-to-host` protocol. It
accepts only this run's input and output paths and plugin folder. Its Docker
adapter maps the backend's logical `ouroboros-volume` to the harness's physical
`VOLUME_NAME`, so isolated benchmark volumes still work without production
configuration changes. The fixture uses the already selected backend image
and host Python standard library, listens on an ephemeral port reachable from
the backend through `host.docker.internal`, and stops when the harness exits.

Inputs are absent from the volume before submission; host output validation
uses only files returned by the backend. The harness requires successful input
and output transfer events in `transfers.jsonl`, including the separate overlay
when requested. This catches regressions that manual input staging and direct
output extraction would conceal. `volume-server.log` records fixture requests.
Run its focused protocol tests without Docker or a GPU:

```bash
python3 -m unittest discover -s backend/scripts/tests
```

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

Each run writes a `revisions.env`, bounded `backend.log`, `transfers.jsonl`,
`volume-server.log`, and synchronized `telemetry.csv` under `ARTIFACT_DIR` (or
`OUTPUT_DIR` when set). The telemetry
rows record elapsed time, GPU utilization, GPU memory, and container memory.
The harness also verifies that the model-load log contains the requested CUDA
ordinal plus the exact plugin and Candle revisions, and that every output TIFF
page is `uint8` with binary values only.

Use a separate run with `overlay_annotation_points` enabled when generating
prompt-marker figure artifacts.

## CANDLE-2 Streamed Control

The 2026-07-18 cc7.5 control at plugin `03d94e6b17e8889b4db4a54a7e3c87744d2679d8`
and Candle `770d20ca8db4f834ba4c89c845bca196fbfc97ea` ran the 16-page,
200 x 200 Medical-SAM3 fixture in 92 seconds (0.174 fps), with 11,536 MiB peak
VRAM and 327.0 MiB peak container RSS. The log recorded
`cached_output_frames=0`. The streamed output SHA-256 was
`bf6d366d33adc09053ba6da7065c88916d6780a133bb2779204f458af751c67e`,
byte-identical to both collected old-pin controls.
