import asyncio
import os
from pathlib import PureWindowsPath, PurePosixPath, Path, PurePath
from typing import Dict, Optional
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
import numpy as np
from pydantic import BaseModel
import requests
import tifffile
import torch


# --- Imports for SAM2/SAM3 ---
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor

except ImportError:
    print("SAM2 not installed.")
    build_sam2 = None
    SAM2ImagePredictor = None

try:
    from sam3.build_sam import build_sam3
    from sam3.predictor import SAM3Predictor
except ImportError:
    print("SAM3 not installed.")
    build_sam3 = None
    SAM3Predictor = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config & Globals ---
jobs: Dict[str, Dict] = {}
PLUGIN_NAME = "sam3-segmentation"
VOLUME_SERVER_URL = os.getenv("VOLUME_SERVER_URL", "http://host.docker.internal:3001")
INTERNAL_VOLUME_PATH = os.getenv("VOLUME_MOUNT_PATH", "/ouroboros-volume")
CHECKPOINT_DIR = os.path.join(INTERNAL_VOLUME_PATH, PLUGIN_NAME, "chkpts")

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Caching
loaded_model = None
loaded_model_name = None
loaded_predictor = None

# Docker/Service Initialization Status
startup_status = {
    "is_ready": False,
    "initialization_steps": [
        {"name": "Building Docker Image", "status": "pending"},
        {"name": "Connecting to Volume Server", "status": "pending"},
        {"name": "Initializing ML Models", "status": "pending"}
    ],
    "start_time": None,
    "ready_time": None
}


def mark_initialization_complete():
    """Call this when the service is fully initialized"""
    global startup_status
    startup_status["is_ready"] = True
    startup_status["ready_time"] = time.time()
    for step in startup_status["initialization_steps"]:
        if step["status"] == "pending":
            step["status"] = "completed"
    print("Service initialization complete!")


def update_initialization_step(step_name: str, status: str):
    """Update the status of a specific initialization step"""
    global startup_status
    if startup_status["start_time"] is None:
        startup_status["start_time"] = time.time()
    for step in startup_status["initialization_steps"]:
        if step["name"] == step_name:
            step["status"] = status
            print(f"Initialization: {step_name} -> {status}")


# --- Model Definitions ---
SAM2_CONFIGS = {
    "sam2_hiera_tiny": "sam2_hiera_t.yaml",
    "sam2_hiera_small": "sam2_hiera_s.yaml",
    "sam2_hiera_base_plus": "sam2_hiera_b+.yaml",
    "sam2_hiera_large": "sam2_hiera_l.yaml"
}

SAM2_URLS = {
    "sam2_hiera_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
    "sam2_hiera_small": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
    "sam2_hiera_base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
    "sam2_hiera_large": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
}

# --- Helpers ---


def get_pure_path(path_str: str) -> PurePath:
    # Robustly get filename from path string, handling Windows/Posix mixed scenarios.
    stripped = path_str.strip('"').strip("'")
    if '\\' in stripped or (len(stripped) > 1 and stripped[1] == ':'):
        return PureWindowsPath(stripped)
    return PurePosixPath(stripped)


async def request_volume_server(path, data):
    url = f"{VOLUME_SERVER_URL}/{path}"
    try:
        result = requests.post(url, headers={"Content-Type": "application/json"}, json=data)
        print(f"Data: {data}")
        print(f"Result: {result.text}")
        return (True, "") if result.ok else (False, result.text)
    except Exception as error:
        return False, str(error)


async def copy_to_volume(files):
    return await request_volume_server("copy-to-volume", {
        "volumeName": "ouroboros-volume",
        "pluginFolderName": PLUGIN_NAME,
        "files": files
    })


async def copy_to_host(files):
    return await request_volume_server("copy-to-host", {
        "volumeName": "ouroboros-volume",
        "pluginFolderName": PLUGIN_NAME, "files": files
    })


def download_file(url, dest):
    print(f"Downloading {url} to {dest}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
    else:
        raise Exception(f"Failed to download checkpoint: {response.status_code}")

# --- Model Loading Logic ---


def get_predictor(model_name: str, predictor_type: str):
    global loaded_model, loaded_model_name, loaded_predictor

    cache_key = f"{model_name}_{predictor_type}"

    if loaded_model_name == cache_key and loaded_predictor is not None:
        return loaded_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model: {model_name} (Predictor: {predictor_type}) on {device}...")

    # Define path in the volume checkpoint directory
    checkpoint_name = f"{model_name}.pt"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)

    if model_name in SAM2_CONFIGS:
        if SAM2ImagePredictor is None:
            raise ImportError("SAM2 library not found/installed")

        # Fallback: Auto-download SAM2 if missing (convenience)
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found in {checkpoint_path}, attempting auto-download...")
            try:
                download_file(SAM2_URLS[model_name], checkpoint_path)
            except Exception as e:
                raise RuntimeError(f"Model not found and download failed: {e}")

        config_file = SAM2_CONFIGS[model_name]
        model = build_sam2(config_file, checkpoint_path, device=device)

        if predictor_type == "ImagePredictor":
            loaded_predictor = SAM2ImagePredictor(model)
        elif predictor_type == "VideoPredictor":
            if SAM2VideoPredictor is None:
                raise ImportError("SAM2 VideoPredictor not available")
            loaded_predictor = SAM2VideoPredictor(model)
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")

    elif model_name.startswith("sam3"):
        if SAM3Predictor is None:
            raise ImportError("SAM3 library not found/installed")

        # Strict Check: SAM3 must be downloaded via the UI first
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"SAM3 checkpoint not found at {checkpoint_path}. "
                "Please use the 'Models' section to download it with your authentication token."
            )

        model = build_sam3(checkpoint=checkpoint_path, device=device)
        loaded_predictor = SAM3Predictor(model)

    else:
        raise ValueError(f"Unknown model type: {model_name}")

    loaded_model = model
    loaded_model_name = cache_key
    return loaded_predictor

# --- Endpoints ---


class DownloadRequest(BaseModel):
    model_type: str
    hf_token: Optional[str] = None


class ProcessRequest(BaseModel):
    file_path: str
    output_file: str
    model_type: str
    predictor_type: str


@app.post("/download-model")
async def download_model(req: DownloadRequest):
    # Downloads the requested model to the /ouroboros-volume/sam3-segmentation/chkpts directory.
    try:
        model_name = req.model_type
        target_path = os.path.join(CHECKPOINT_DIR, f"{model_name}.pt")

        if os.path.exists(target_path):
            return {"status": "exists", "message": f"Model {model_name} already exists."}

        print(f"Initiating download for {model_name}...")

        # SAM 2 Download
        if model_name in SAM2_URLS:
            download_file(SAM2_URLS[model_name], target_path)
            return {"status": "success", "message": f"Downloaded {model_name}"}

        # SAM 3 Download
        elif model_name.startswith("sam3"):
            if not req.hf_token:
                raise HTTPException(400, "Authentication Token required for SAM 3")

            repo_id = "facebook/sam3"
            filename = f"{model_name}.pt"

            print(f"Fetching {filename} from Hugging Face...")
            try:
                # hf_hub_download downloads to cache, we move/copy it to our persistent dir
                cached_path = hf_hub_download(repo_id=repo_id, filename=filename, token=req.hf_token)

                # Move/Copy to our volume structure
                import shutil
                shutil.copy(cached_path, target_path)

                return {"status": "success", "message": f"Downloaded {model_name}"}
            except Exception as e:
                raise HTTPException(500, f"Hugging Face Download Failed: {str(e)}")

        else:
            raise HTTPException(400, "Unknown model type")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/process-stack")
async def process_stack(req: ProcessRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "running",
        "steps": [
            {"name": "Transferring", "progress": 0},
            {"name": "Inference", "progress": 0},
            {"name": "Saving", "progress": 0}
        ]
    }
    background_tasks.add_task(
        run_pipeline,
        job_id,
        req.file_path,
        req.output_file,
        req.model_type,
        req.predictor_type
    )
    return {"job_id": job_id, "status": "started"}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


def update_step(job_id, step_index, progress):
    if job_id in jobs:
        jobs[job_id]["steps"][step_index]["progress"] = progress


async def run_pipeline(job_id: str, host_path: str, output_path: str, model_type: str, predictor_type: str):
    try:
        host_source = get_pure_path(host_path)
        host_result = get_pure_path(output_path)
        volume_source = Path(INTERNAL_VOLUME_PATH, PLUGIN_NAME, host_source.name)
        volume_result = Path(INTERNAL_VOLUME_PATH, PLUGIN_NAME, "Segmentation", host_result.name)

        # 1. Transfer
        update_step(job_id, 0, 10)
        success, msg = await copy_to_volume([{"sourcePath": str(host_source), "targetPath": ""}])
        if not success:
            raise Exception(f"Copy failed: {msg}")
        update_step(job_id, 0, 100)

        # 2. Inference
        if not os.path.exists(volume_source):
            raise FileNotFoundError(f"File missing: {volume_source}")

        stack = tifffile.imread(volume_source)

        if stack.dtype != np.uint8:
            img_min, img_max = stack.min(), stack.max()
            if img_max > img_min:
                stack = ((stack.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                stack = np.zeros_like(stack, dtype=np.uint8)

        update_step(job_id, 1, 0)
        # Load Model (will look in chkpts dir)
        try:
            predictor = get_predictor(model_type, predictor_type)
        except Exception as e:
            print(f"Model Init Error: {e}")
            raise e

        if stack.ndim == 2:
            stack = stack[np.newaxis, ...]

        num_slices = stack.shape[0]
        # FIX: Result Stack Dimensions (Z, Y, X)
        result_shape = stack.shape[:3]
        result_stack = np.zeros(result_shape, dtype=np.uint8)

        # Determine if we're using VideoPredictor or ImagePredictor
        is_video_predictor = predictor_type == "VideoPredictor"

        if is_video_predictor:
            # Initialize video predictor state
            predictor.init_state(None)

        for i in range(num_slices):
            img_slice = stack[i]

            # --- Tensor Dimension Fix (Ensure H,W,3) ---
            if img_slice.ndim == 2:
                img_3ch = np.expand_dims(img_slice, axis=-1)
            else:
                img_3ch = img_slice

            if img_3ch.shape[-1] == 1:
                img_3ch = np.repeat(img_3ch, 3, axis=-1)
            elif img_3ch.shape[-1] > 3:
                img_3ch = img_3ch[..., :3]

            img_3ch = np.ascontiguousarray(img_3ch)

            h, w = img_3ch.shape[:2]
            input_point = np.array([[w // 2, h // 2]])
            input_label = np.array([1])

            if is_video_predictor:
                # For VideoPredictor, process frames sequentially
                if i == 0:
                    # Add the first frame with prompts
                    _, out_obj_ids, out_mask_logits = predictor.add_new_frame_with_prompt(
                        img_3ch,
                        predictor_masks=None,
                        point_coords=input_point,
                        point_labels=input_label
                    )
                    masks = out_mask_logits[0] > 0  # Convert logits to binary mask
                else:
                    # Propagate through subsequent frames
                    _, out_obj_ids, out_mask_logits = predictor.track_step(img_3ch)
                    if out_obj_ids is not None and len(out_obj_ids) > 0:
                        masks = out_mask_logits[0] > 0  # Convert logits to binary mask
                    else:
                        # No objects tracked, use empty mask
                        masks = np.zeros((h, w), dtype=bool)
            else:
                # For ImagePredictor, process each frame independently
                predictor.set_image(img_3ch)
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False,
                )

                if masks.ndim == 3:
                    masks = masks[0]

            result_stack[i] = (masks > 0).astype(np.uint8) * 255

            pct = int(((i + 1) / num_slices) * 100)
            update_step(job_id, 1, pct)
            await asyncio.sleep(0.001)

        update_step(job_id, 1, 100)

        # 3. Saving
        volume_result.parent.mkdir(exist_ok=True, parents=True)
        tifffile.imwrite(volume_result, result_stack)
        update_step(job_id, 2, 50)

        # Robust copy back
        print(f"Copying {volume_result} back to {host_result}")
        success, msg = await copy_to_host([{"sourcePath": str(host_result), "targetPath": "Segmentation"}])
        if not success:
            raise Exception(f"Copy back failed: {msg}")
        update_step(job_id, 2, 100)

        jobs[job_id]["status"] = "completed"

    except Exception as e:
        print(f"Pipeline Error: {e}")
        import traceback
        traceback.print_exc()
        jobs[job_id]["status"] = "error"
