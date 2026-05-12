# Rust Backend Scaffold

This crate is an additive scaffold for the planned Python-to-Rust migration. It lives alongside the current Python backend so the existing plugin runtime remains unchanged for now.

What is in place:

- `backend/Cargo.toml`
- `backend/src/api/*` mirroring the current HTTP surface
- `backend/src/app_state.rs` for in-memory jobs and startup status
- `backend/src/services/*` for startup, checkpoints, volume operations, pipeline orchestration, and job execution
- `backend/src/inference/*` for the future Candle SAM2 and SAM3 adapters
- `backend/src/imaging/*` for TIFF inspection, annotation handling, preprocessing, overlay, and output seams
- Rust tests for the Phase 2 non-ML infrastructure:
  - mixed host-path parsing
  - volume-server payloads and error handling
  - checkpoint status/download logic
  - startup refresh behavior
- Rust tests for the Phase 3 imaging and prompt plumbing:
  - TIFF stack inspection and annotation metadata loading
  - fallback/default annotation generation and interpolation
  - TIFF mask-stack writing
  - staged TIFF and JPEG frame generation
  - pipeline preparation for ImagePredictor and VideoPredictor inputs

What is intentionally still stubbed:

- Candle SAM2 image and video inference
- Candle SAM3 image and video inference
- Volume transfer and checkpoint-backed end-to-end execution inside the Rust pipeline

Recommended next steps:

1. Wire the Phase 3 staging path into the real volume-transfer pipeline.
2. Implement the image inference path before the video path.
3. Switch Docker and compose files only after the Rust backend can satisfy the current frontend contract.
