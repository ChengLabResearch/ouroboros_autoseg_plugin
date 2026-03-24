# Rust Backend Scaffold

This crate is an additive scaffold for the planned Python-to-Rust migration. It lives alongside the current Python backend so the existing plugin runtime remains unchanged for now.

What is in place:

- `backend/Cargo.toml`
- `backend/src/api/*` mirroring the current HTTP surface
- `backend/src/app_state.rs` for in-memory jobs and startup status
- `backend/src/services/*` for startup, checkpoints, volume operations, pipeline orchestration, and job execution
- `backend/src/inference/*` for the future Candle SAM2 and SAM3 adapters
- `backend/src/imaging/*` for TIFF, annotation, preprocessing, overlay, and output seams
- Rust tests for the Phase 2 non-ML infrastructure:
  - mixed host-path parsing
  - volume-server payloads and error handling
  - checkpoint status/download logic
  - startup refresh behavior

What is intentionally still stubbed:

- TIFF inspection and frame staging
- TIFF output writing
- Candle SAM2 image and video inference
- Candle SAM3 image and video inference
- Full pipeline execution

Recommended next steps:

1. Implement TIFF inspection and output writing.
2. Wire the Phase 2 volume/checkpoint infrastructure into the real Rust pipeline.
3. Implement the image inference path before the video path.
4. Switch Docker and compose files only after the Rust backend can satisfy the current frontend contract.
