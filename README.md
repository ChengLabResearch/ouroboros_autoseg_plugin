# Automatic segmentation plugin for Ouroboros

This template is designed to integrate easily with the main app in development.

A plugin consists of a React frontend and a Docker backend. It has a GitHub action that automatically runs `npm run build` and creates a release from the `dist` folder. 

The main production app has an option to download a plugin from GitHub in the plugin manager.

## Model compatibility

The production Rust backend presents two SAM3-compatible model choices. Both
use the Candle SAM3 runtime in-process; the model selector controls which
checkpoint is downloaded and loaded.

| Selector value | Display label | Checkpoint source | Stored checkpoint | Token required | Prompt support |
| --- | --- | --- | --- | --- | --- |
| `sam3` | SAM3 | Hugging Face `facebook/sam3`, file `sam3.pt` | `sam3.pt` | Yes | Image points, video points |
| `medical_sam3` | Medical SAM3 | Hugging Face `ChongCong/Medical-SAM3`, file `checkpoint_3D.pt` at revision `716c9e1fb70edbdbbe65526b781e3a60f382d6cc` | `medical_sam3.pt` | No | Image points, video points |

The production Rust model catalog pins Medical SAM3 to the previously tested
stage-1 checkpoint 5. Upstream removed `checkpoint_3D.pt` from `main` on
2026-08-16, causing new downloads to fail with an upstream 404 (reported by the
plugin as HTTP 502). The [pinned checkpoint](https://huggingface.co/ChongCong/Medical-SAM3/resolve/716c9e1fb70edbdbbe65526b781e3a60f382d6cc/checkpoint_3D.pt)
has SHA-256 `6e40bbaa739ac44e3e47dc6355ef6dedc560a30411377ad891f8af9e6df0dbd6`,
matching the recorded [point-prompted video baseline](backend/docs/candle2_old_pin_baseline.md).
Existing `medical_sam3.pt` files are reused without redownloading; the download
path does not revalidate their contents or enforce this checksum.

The newer `checkpoint_3D_checkpoint8.pt` uses a different training resolution,
and `checkpoint_3D_LoRA2D.pt` is an adapter requiring checkpoint 8. Neither is
selected by this fix. Image and video point prompts remain supported.

The `/model-status` endpoint reports `sam3` and `medical_sam3` only, matching
the current production SAM3 surface. The `/download-model` endpoint stores
checkpoints under the plugin checkpoint directory, and `/process` lazy-loads
the selected checkpoint through the Candle SAM3 path before staging input
frames.

The Models panel shows download stages and received bytes. When the source
provides a reliable total, it also shows a percentage; otherwise the progress
bar stays indeterminate. Reopening the panel reconnects to an active download.
Downloads become available after the checkpoint is finalized, and failed
downloads can be retried. See the [download status contract](backend/docs/checkpoints.md#download-status)
for API and restart behavior.

### Centre-line mask cleanup

The optional **Keep Centre-line Component** control (`keep_centerline_component`,
default `false`) applies to both predictors. Before saving, it keeps the
8-connected foreground region containing the interpolated annotation point in
each 2D slice. Coordinates are rounded and clamped to the image. If that pixel
is background, the nearest foreground pixel within a 10-pixel Euclidean radius
is used (row-major order breaks ties). If no seed is available, the slice is
left unchanged. Missing metadata uses the existing centre-point annotations.

Separate branches or folds without the seed are removed; connected leakage
into neighbouring tissue remains. The output and optional annotation overlay
both use the cleaned masks. The job log reports
changed frames, removed pixels, background/missing seed counts (including
recovered seeds), and frames left unchanged because no seed was found.

### Inconsistent-frame repair

**Repair Inconsistent Frames** is optional and off by default for both
predictors. It runs after centre-line cleanup, if that is enabled. It uses the
same interpolated trace, nearest-foreground search and 8-connected component
selector on every frame, including video frames without inference prompts.
Enabling repair removes off-trace components from every frame, including
accepted frames, even when **Keep Centre-line Component** is off. If a trace
point exists but no foreground lies within 10 pixels of it, centre-line
cleanup leaves the frame unchanged; repair empties it, flags it
`empty_component`, and may fill it from an accepted-frame estimate.

The selected component is flagged if it touches an image edge, its area changes
by more than a factor of **1.5** in either direction, or its IoU
falls below **0.5**. These thresholds are fixed. Area and IoU compare against the
previous selected component **before any repair**, including previously flagged
frames. The first frame has no previous-frame comparison. Missing trace points
and empty selections are also flagged. Empty/nonempty area transitions count
as an infinite ratio; two empty components have ratio 1 and IoU 0.

A flagged, nonempty component that does not touch an edge is recovered as an
anchor when its original area is within 0.7–1.4 times the median area of
initially accepted frames within ±150 frames, including both endpoints. The
median averages the middle two values for an even count. Recovery uses this
fixed initial set, so one recovered anchor cannot cause another recovery.
Without an accepted frame in the window, the frame remains flagged. Recovered
anchors keep their own selected masks unchanged.

Remaining flagged frames are grouped into runs. Each frame's expected mask
comes from the two surrounding accepted components. Their chessboard
(Chebyshev) signed distance fields are interpolated in place by frame position,
without alignment, and thresholded strictly above zero. A
leading or trailing run uses its only accepted neighbour as the estimate.

The frame keeps its own component intersected with the estimate dilated by a
fixed 21 × 21 Euclidean disc of radius 10 pixels. Only when the surviving area is **less than 30% of
the undilated estimate's area** is the estimate substituted. There is no extra
overlap/agreement gate. This preserves observed detail while filling dropouts.
Actions are `kept` when there is no pixel change beyond component selection,
`clipped` when repair removes foreground, and `replaced` when it substitutes
the estimate. Clipping retains original foreground values; substituted
estimates use 0/255.

Frames without any trace remain empty and flagged. If every frame is flagged,
there is no accepted anchor: selected components remain unchanged with action
`kept`, and remain in the review list. An empty estimate yields an empty
intersection; if both the selected component and estimate are empty, the frame
stays empty and flagged. Repair does not add inference prompts or rerun the
model. Estimated frames remain review
targets, particularly where the source image has lost contrast.

The API options are two booleans: `repair_inconsistent_frames` and
`keep_unrepaired`, both defaulting to `false`. Set the first to `true` to enable
repair, and optionally the second to retain its input stack. `keep_unrepaired`
has no effect when repair is off. Disabled repair leaves the existing output
unchanged.

For `segmented.tif`, the repaired primary output is accompanied by
`segmented_frame_repair.json` (schema version 2). Each record contains the
original `component_area`, detector `reasons`, `anchor_recovered`, and `repair`
action. A frame remains flagged when its reasons are nonempty and it was not
recovered as an anchor. Original reasons are retained for recovered anchors;
they are excluded from the final runs and review list. The summary counts
initial flags, recovered anchors, final flags, and kept/clipped/replaced
actions. Run boundaries are inclusive and frame indices start at zero.
The same records supply the job-log summary, final flag list, recovered-anchor
list, and overlay markers. **Keep Unrepaired Stack** additionally writes
`segmented_unrepaired.tif`, captured after optional centre-line cleanup and
before component selection and repair. With **Draw Prompts**, each actually
repaired frame gets a small X centred at pixel (2, 2), at intensity 192 (191 if
the positive prompt intensity is 192). Kept frames, including recovered anchors
and unchanged flagged frames, have no repair marker. All requested artifacts
are copied beside the primary output.

### Prompt overlay

With **Draw Prompts**, original metadata and fallback positives
use stars; positives added by **Max Prompt Gap** use hollow diamonds.

The diagnostic overlay uses the same prepared prompt list submitted to
inference. Video frames without prompts, including frames before the first and
after the last prompt, remain identical to the primary mask unless marked as
repaired by the optional repair step. Every positive
on a prompted frame is drawn, including multiple annotations whose fractional
z coordinates round to that frame.
Image prediction still marks every frame it prompts. Coordinates are rounded
to pixels and glyphs are clipped at image edges.
Drawing prompts changes neither inference nor centre-line cleanup, which keeps
its existing interpolated annotation seeds.

To get started with developing a plugin:

**Option 1:** Clone the main repository and make a folder/repository inside of the plugins folder. Push only that folder to GitHub. 

**Option 2:** If it is easier for you, copy all the contents of the `plugin-template` folder into a completely separate repository. All of the usage steps should still work.

### Usage

1. Follow the instructions in the [README](https://github.com/We-Gold/ouroboros/) to install the app in development mode.

2. Open a terminal and `cd` into your plugin's folder. Then run `npm install` to install your plugin's dependencies.

3. Start the main app in development mode: Run `npm run dev` in the main project folder. 

4. Start the plugin in development mode: Run `npm run dev` in your plugin's folder.

5. In the main app, go to the first menu dropdown and open the plugin manager. Click the plus, and paste the URL of your plugin (something like `http://localhost:5172`) in the development plugin option.

### Backend Development Docker

`npm run dev-backend` uses backend dev compose files with:

- Rust backend compose files selected by local hardware:
  - `backend/compose.dev.yml` when no NVIDIA GPU is detected
  - `backend/compose.gpu.dev.yml` when `nvidia-smi` is available
- conditional Docker rebuilds only when backend build inputs change:
  - `backend/Dockerfile`
  - `backend/Cargo.toml`
  - `backend/Cargo.lock`
  - `backend/src`
  - `backend/tests`
  - backend compose files

Frontend-only edits do not trigger a backend image rebuild. Rust backend edits do rebuild the image because the release-style container compiles the Rust server into the runtime image.

### Production Plugin Artifacts

Tagged releases publish two preinstallable plugin artifacts:

- `auto-segmentation-<tag>-cpu.zip`
- `auto-segmentation-<tag>-cuda.zip`

The beta.7 plugin release uses these artifact names and image tags:

- tag: `v0.4.0-beta.7`
- CPU asset: `auto-segmentation-v0.4.0-beta.7-cpu.zip`
- CUDA asset: `auto-segmentation-v0.4.0-beta.7-cuda.zip`
- CPU backend image: `ghcr.io/chenglabresearch/ouroboros-autoseg-backend:v0.4.0-beta.7`
- CUDA backend image: `ghcr.io/chenglabresearch/ouroboros-autoseg-backend:v0.4.0-beta.7-cuda`

The beta.7 backend is pinned to Candle SAM3 commit
`c0400c6513c21655828bb92633cc190a3501a6f6`.

Both archives unpack to the normal Ouroboros plugin folder layout, including
`package.json`, `index.html`, `icon.svg`, `compose.yml`, frontend assets, and
`plugin-release.json`. Release artifacts pin the CPU or CUDA backend by its
verified `ghcr.io/chenglabresearch/ouroboros-autoseg-backend@sha256:...`
digest. The CUDA artifact also includes the NVIDIA GPU device reservation.

For production package preinstalls, unpack the selected artifact under
`extra-resources/preinstalled-plugins/auto-segmentation/` before building the
Ouroboros package.

### GPU Backend Images

The GPU compose files use a CUDA-specific Docker target:

- `backend/compose.gpu.yml` for packaged GPU backends
- `backend/compose.gpu.dev.yml` for local GPU development

Those compose files select the `cuda-runtime` Docker target and pass the
canonical `CANDLE_FEATURES=cuda,cudnn` feature set to the Candle dependencies.
The target uses matching CUDA 12.4.1 cuDNN development and runtime bases and
reports `cuda=true cudnn=true` in its startup log. Building or running these
images requires an NVIDIA-capable Docker environment with the NVIDIA container
toolkit available.

### Registry Backend Images

The `Build and Verify Backend Images` workflow fingerprints the actual CPU and
CUDA build inputs. Same-repository pull requests publish
`candidate-pr-<number>-<fingerprint>` images and update the bounded canonical
CPU/CUDA registry caches; fork pull requests only read those caches. A matching
`main` build locates the merged pull request and reuses its candidates instead
of rerunning Cargo. Successful `main` builds publish commit tags:

- `ghcr.io/chenglabresearch/ouroboros-autoseg-backend:sha-<commit>`
- `ghcr.io/chenglabresearch/ouroboros-autoseg-backend:sha-<commit>-cuda`

Version-tag jobs verify those exact commit images, promote their manifests to
the release tags, and build plugin archives containing the verified digests;
they do not rebuild the Rust backend. The unsuffixed tags use the CPU runtime
target, and the `-cuda` tags use the CUDA runtime target. The existing
`backend/compose.yml` remains the local-build fallback.
`backend/compose.registry.yml` is an opt-in compose file that accepts a prebuilt
immutable image through `OUROBOROS_AUTOSEG_BACKEND_IMAGE`.

Hosted CI checks the declared `cuda,cudnn` capabilities and verifies that the
backend's ELF dependencies resolve the cuDNN runtime without initializing a
GPU. An optional checkpoint-backed pre-release diagnostic can exercise an
exact image digest on a GPU host without rebuilding it:

```bash
BACKEND_IMAGE=ghcr.io/chenglabresearch/ouroboros-autoseg-backend@sha256:<digest> \
INPUT_STACK=/path/to/straightened-stack.tif \
CHECKPOINT_PATH=/path/to/checkpoint_3D.pt \
  backend/scripts/certify_cuda_candidate_gpu.sh
```

The diagnostic is not a dependency of the `main` image workflow and does not
write registry state. Its `ARTIFACT_DIR` retains the exact image reference,
compiled features, revisions, telemetry, bounded backend log, and validated
output for release review.

### `package.json`

The first lines of the package.json are important to identifying your plugin.

```
"name": "plugin-template",
"pluginName": "Plugin Template",
"icon": "./icon.svg",
"index": "./index.html",
"dockerCompose": "./compose.yml",
```

- `name` is considered to be the plugin `id`
- `pluginName` is the display name of the plugin
- `icon` is the **`dist`-relative** path to the icon for the plugin
- `index` is the **`dist`-relative** path to the index HTML file generated by the build
- `dockerCompose` is an optional **`dist`-relative** path to a Docker Compose file to run the plugin backend.
