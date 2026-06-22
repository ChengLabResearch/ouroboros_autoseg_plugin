# Checkpoint Verification

The production Rust backend loads SAM3 checkpoints through
`candle_transformers::models::sam3::Sam3CheckpointSource::upstream_pth`.

## Medical-SAM3 Load Check

Stage the Medical-SAM3 checkpoint and point `OUROBOROS_SAM3_CHECKPOINT` at the
local file:

```bash
export OUROBOROS_SAM3_CHECKPOINT=/path/to/medical_sam3.pt
cargo test --ignored sam3_medical_checkpoint_load -- --nocapture
```

The ignored test loads the checkpoint through the same production loader used by
`/process`, runs the image path on a tiny synthetic fixture, and asserts that the
output mask has the input geometry and uses the established `uint8` `0/255`
convention.

For a command-line smoke check:

```bash
cargo run --release --bin sam3_checkpoint_smoke -- \
  --checkpoint "$OUROBOROS_SAM3_CHECKPOINT" \
  --mask-out /tmp/sam3_smoke_mask.tif
```

The smoke binary exits non-zero when the checkpoint cannot be loaded or the mask
geometry is invalid. Pass `--image /path/to/input.jpg` to use a real image
instead of the built-in synthetic frame.
