# Checkpoint Verification

The production Rust backend loads SAM3 checkpoints through
`candle_transformers::models::sam3::Sam3CheckpointSource::upstream_pth`.

## Download status

The checkpoint service owns each model's download lifecycle and writes its
checkpoint through the existing `.part` file, flush, and rename path. A model
has one of these states:

| State | Meaning |
| --- | --- |
| `idle` | No installed checkpoint or active download. |
| `connecting` | Preparing the download and waiting for the source response. |
| `downloading` | Receiving and writing checkpoint bytes. |
| `finalizing` | Finishing the file and installing it at the checkpoint path. |
| `available` | The final checkpoint file is installed. |
| `failed` | The attempt failed; its error remains available until retry or installation. |

`GET /model-status` retains `checkpoint_dir` and the `models` availability
booleans. It adds `downloads` for the same `sam3` and `medical_sam3` keys:

```json
{
  "state": "downloading",
  "bytes_received": 1048576,
  "total_bytes": 4194304,
  "error": null
}
```

Each value in `downloads` has this shape. `bytes_received` counts bytes written
by the current attempt; an already installed file reports its size.
`total_bytes` is `null` when the response does not
provide a reliable total; clients must use indeterminate progress in that
case. A complete byte count still does not mean installation has finished.
The compatibility booleans derive from `state == "available"` in the same
snapshot; `.part` files never qualify. Existing installed files are recognized
without redownloading or content/checksum verification.

`POST /download-model` keeps its existing request and completion response:
`success` after installation, or `exists` for an installed checkpoint. A
second request for an active model receives HTTP 409 and can observe the
existing attempt through `/model-status`. Downloads for different models can
run concurrently. Disconnecting the POST caller or closing the panel does not
cancel a service-owned download; a lost client response must not be treated as
a backend download failure.

Snapshots and failures live for the backend process lifetime. After a backend
restart, final files are rediscovered and incomplete `.part` files remain
unavailable; the next attempt replaces the partial file rather than resuming
its bytes. Download authorization is held only by the active request/worker;
status responses do not retain tokens or arbitrary upstream response bodies.
The existing one-hour download timeout and model catalog remain in effect.

The Models panel polls the existing status endpoint independently of ML runtime
readiness, so downloading a first model does not require one to be loaded.
It polls serially, keeps its last
snapshot visibly stale on contact loss, and restores live progress after
reconnection. Its request feedback describes HTTP contact only; the service
snapshot determines whether a model is active, installed, or failed.

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
