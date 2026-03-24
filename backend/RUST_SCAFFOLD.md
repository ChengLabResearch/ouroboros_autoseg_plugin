# Rust Backend Scaffold

This crate is an additive scaffold for the planned Python-to-Rust migration. It lives alongside the current Python backend so the existing plugin runtime remains unchanged for now.

What is in place:

- `backend/Cargo.toml`
- `backend/src/api/*` mirroring the current HTTP surface
- `backend/src/app_state.rs` for in-memory jobs and startup status
- `backend/src/services/*` for startup, checkpoints, volume operations, pipeline orchestration, and job execution
- `backend/src/inference/*` for the future Candle SAM2 and SAM3 adapters
- `backend/src/imaging/*` for TIFF, annotation, preprocessing, overlay, and output seams

What is intentionally still stubbed:

- TIFF inspection and frame staging
- TIFF output writing
- Candle SAM2 image and video inference
- Candle SAM3 image and video inference
- Full pipeline execution

Recommended next steps:

1. Add `cargo fmt` and `cargo check` to local development.
2. Port the pure utility logic first:
   - annotation interpolation
   - fallback prompts
   - overlay markers
   - mixed-path parsing
3. Implement TIFF inspection and output writing.
4. Implement the image inference path before the video path.
5. Switch Docker and compose files only after the Rust backend can satisfy the current frontend contract.
