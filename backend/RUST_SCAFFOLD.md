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

## Candle dependencies

The backend uses two git dependencies, both pinned by commit in `Cargo.toml` and
`Cargo.lock`:

- `candle-core` and `candle-nn` from [den-sq/candle](https://github.com/den-sq/candle),
  a fork of Candle with general changes (MIT OR Apache-2.0);
- `sam3` (crate `candle-sam3`) from [den-sq/candle-sam3](https://github.com/den-sq/candle-sam3),
  the SAM 3 engine, distributed under the SAM License.

`candle-sam3` pins the same fork commit, so both use one set of Candle types. When
updating either pin, keep the fork `rev` here equal to the one in `candle-sam3`'s
`Cargo.toml`, and keep the Dockerfile's `CANDLE_SAM3_COMMIT` equal to the
`candle-sam3` `rev`.

No sibling checkout is needed:

```bash
cd ouroboros_autoseg_plugin/backend
cargo build --release
cargo test
```

To build against local checkouts of either repository, use a
[`[patch]`](https://doc.rust-lang.org/cargo/reference/overriding-dependencies.html)
section in an uncommitted `.cargo/config.toml`.
