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

- tag: `v0.4.0-beta.1`
- CPU asset: `auto-segmentation-v0.4.0-beta.1-cpu.zip`
- CUDA asset: `auto-segmentation-v0.4.0-beta.1-cuda.zip`
- CPU backend image: `ghcr.io/chenglabresearch/ouroboros-autoseg-backend:v0.4.0-beta.1`
- CUDA backend image: `ghcr.io/chenglabresearch/ouroboros-autoseg-backend:v0.4.0-beta.1-cuda`

Both archives unpack to the normal Ouroboros plugin folder layout, including
`package.json`, `index.html`, `icon.svg`, `compose.yml`, frontend assets, and
`plugin-release.json`. The CPU artifact `compose.yml` points at
`ghcr.io/chenglabresearch/ouroboros-autoseg-backend:<tag>`. The CUDA artifact
points at `ghcr.io/chenglabresearch/ouroboros-autoseg-backend:<tag>-cuda` and
includes the NVIDIA GPU device reservation.

For production package preinstalls, unpack the selected artifact under
`extra-resources/preinstalled-plugins/auto-segmentation/` before building the
Ouroboros package.

### GPU Backend Images

The GPU compose files use a CUDA-specific Docker target:

- `backend/compose.gpu.yml` for packaged GPU backends
- `backend/compose.gpu.dev.yml` for local GPU development

Those compose files select the `cuda-runtime` Docker target and pass `CANDLE_FEATURES=cuda`, which forwards the plugin crate's `cuda` feature to the Candle dependencies. Building these images requires an NVIDIA-capable Docker environment with the NVIDIA container toolkit available.

### Registry Backend Images

The `Publish Backend Image` workflow publishes the Rust backend image to GHCR for release tags and commit SHAs:

- `ghcr.io/chenglabresearch/ouroboros-autoseg-backend:<release-tag>`
- `ghcr.io/chenglabresearch/ouroboros-autoseg-backend:sha-<commit>`
- `ghcr.io/chenglabresearch/ouroboros-autoseg-backend:<release-tag>-cuda`
- `ghcr.io/chenglabresearch/ouroboros-autoseg-backend:sha-<commit>-cuda`

The unsuffixed tags use the CPU runtime target, and the `-cuda` tags use the CUDA runtime target. The existing `backend/compose.yml` remains the local-build fallback. `backend/compose.registry.yml` is an opt-in packaged compose file for release builds that want to use a prebuilt immutable image by setting `OUROBOROS_AUTOSEG_BACKEND_IMAGE`.

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
