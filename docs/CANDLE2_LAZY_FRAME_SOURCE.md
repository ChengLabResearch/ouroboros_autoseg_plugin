# CANDLE-2 lazy frame-source boundary

This integration implements https://github.com/den-sq/sam_parity/issues/41.

## Ownership

- `candle-transformers` accepts caller-owned `FrameSource` implementations and only handles tensors.
- The plugin owns staged-frame discovery, JPEG decode, RGB conversion, bicubic resize, normalization input, cache eviction, and source lifetime.
- The plugin adapter accepts only `.jpg` and `.jpeg`, matching its staged biological-stack contract and compiled JPEG support. TIFF remains an input-volume format handled before staging; PNG, BMP, WebP, and video-container claims are intentionally absent from this adapter.
- Candle examples own their JPEG/PNG and ffmpeg adapters and PNG debug-artifact sink.

## Memory contract

`StagedJpegFrameSource` starts with zero decoded frames. `prefetch` and `get_frame` decode on demand, `evict_except` releases all frames outside the predictor window, and `close` clears the cache. The low-memory production profile remains zero-prefetch with a one-entry visual-feature cache.

## Adapter profiling

Run the decoder/resize/normalization path independently of model inference:

```bash
cargo run --release --manifest-path backend/Cargo.toml \
  --bin sam3_frame_source_bench -- /path/to/staged-jpeg-frames 1
```

The JSON result reports the adapter and legacy-equivalent decode/resize/normalize throughput, their percentage delta, plus peak and final loaded-frame counts and bytes. End-to-end CUDA certification remains in `backend/scripts/biological_video_smoke_gpu.sh`.

No checkpoint, input stack, decoded frame, or output mask is committed by these tools.
