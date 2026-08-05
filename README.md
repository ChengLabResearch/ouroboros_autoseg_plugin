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
| `medical_sam3` | Medical SAM3 | Hugging Face `ChongCong/Medical-SAM3`, file `checkpoint_3D.pt` | `medical_sam3.pt` | No | Image points, video points |

The `/model-status` endpoint reports `sam3` and `medical_sam3` only, matching
the current production SAM3 surface. The `/download-model` endpoint stores
checkpoints under the plugin checkpoint directory, and `/process` lazy-loads
the selected checkpoint through the Candle SAM3 path before staging input
frames.

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

The current production beta pin for Ouroboros package builds is:

- tag: `v0.4.0-beta.3`
- CPU asset: `auto-segmentation-v0.4.0-beta.3-cpu.zip`
- CUDA asset: `auto-segmentation-v0.4.0-beta.3-cuda.zip`
- CPU backend image: `ghcr.io/chenglabresearch/ouroboros-autoseg-backend:v0.4.0-beta.3`
- CUDA backend image: `ghcr.io/chenglabresearch/ouroboros-autoseg-backend:v0.4.0-beta.3-cuda`

The beta.3 backend is pinned to Candle SAM3 commit
`c0400c6513c21655828bb92633cc190a3501a6f6`.

Both archives unpack to the normal Ouroboros plugin folder layout, including
`package.json`, `index.html`, `icon.svg`, `compose.yml`, frontend assets, and
`plugin-release.json`. Release artifacts pin the CPU or CUDA backend by its
certified `ghcr.io/chenglabresearch/ouroboros-autoseg-backend@sha256:...`
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

The `Build and Certify Backend Images` workflow fingerprints the actual CPU and
CUDA build inputs. Same-repository pull requests publish
`candidate-pr-<number>-<fingerprint>` images and update the bounded canonical
CPU/CUDA registry caches; fork pull requests only read those caches. A matching
`main` build locates the merged pull request and reuses its candidates instead
of rerunning Cargo. Successful `main` builds publish commit tags:

- `ghcr.io/chenglabresearch/ouroboros-autoseg-backend:sha-<commit>`
- `ghcr.io/chenglabresearch/ouroboros-autoseg-backend:sha-<commit>-cuda`

Version-tag jobs verify those exact commit images, promote their manifests to
the release tags, and build plugin archives containing the certified digests;
they do not rebuild the Rust backend. The unsuffixed tags use the CPU runtime
target, and the `-cuda` tags use the CUDA runtime target. The existing
`backend/compose.yml` remains the local-build fallback.
`backend/compose.registry.yml` is an opt-in compose file that accepts a prebuilt
immutable image through `OUROBOROS_AUTOSEG_BACKEND_IMAGE`.

CUDA certification has two gates against the same candidate digest. Hosted CI
checks the declared `cuda,cudnn` capabilities and verifies that the backend's
ELF dependencies resolve the cuDNN runtime without initializing a GPU. The
checkpoint-backed encoder gate runs on a GPU host and must not rebuild:

```bash
BACKEND_IMAGE=ghcr.io/chenglabresearch/ouroboros-autoseg-backend@sha256:<digest> \
INPUT_STACK=/path/to/straightened-stack.tif \
CHECKPOINT_PATH=/path/to/checkpoint_3D.pt \
  backend/scripts/certify_cuda_candidate_gpu.sh
```

After the smoke passes, the script records a digest-specific
`gpu-certified-<digest>` registry marker. The `main` workflow refuses to create
the immutable `sha-<commit>-cuda` tag unless that marker resolves to the exact
candidate digest. If the GPU gate happens after the merge build, rerun the
failed workflow; it reuses the existing candidate instead of compiling again.
Publishing the marker requires a prior `docker login ghcr.io` with package
write access. The smoke's `ARTIFACT_DIR` retains the exact image reference,
compiled features, revisions, telemetry, bounded backend log, and validated
output.

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
