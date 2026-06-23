# SAM3 Video Smoke Checks

AUTO-4 keeps the production video path on conservative runtime settings:

- `VideoMemoryProfile::LowMemory`
- no frame or state CPU offload
- no prefetch ahead or behind
- at most two feature cache entries

The unit tests lock those options and validate that video propagation output has
one mask per staged frame, with the same width and height as the straightened
input volume.

## Manual Biological-Stack Smoke

After staging a SAM3 checkpoint and a small straightened stack with
`annotation_points` metadata in the plugin volume, start the backend and submit a
video job:

```bash
curl -X POST http://127.0.0.1:8686/process-stack \
  -H 'Content-Type: application/json' \
  -d '{
    "file_path": "/host/path/straightened-stack.tif",
    "output_file": "/host/path/segmented.tif",
    "model_type": "medical_sam3",
    "predictor_type": "VideoPredictor",
    "overlay_annotation_points": false
  }'
```

Then poll `/status/{job_id}`. A valid smoke run should:

- finish without frame-load panics or out-of-memory errors
- preserve progress updates through polling or UI reconnects
- write an output mask stack with the same frame count, width, and height as the
  straightened input
- keep normal output clean when `overlay_annotation_points` is false

Use a separate run with `overlay_annotation_points` enabled when generating
prompt-marker figure artifacts.
