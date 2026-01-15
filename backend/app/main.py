import asyncio
from multiprocessing import Pool
import os
from pathlib import PureWindowsPath, PurePosixPath, Path, PurePath
from PIL import Image
import psutil
import shutil
import threading
import time
import traceback
from typing import Dict, Optional
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
import numpy as np
from pydantic import BaseModel
import requests
import tifffile as tf
import torch


# --- Imports for SAM2/SAM3 ---
try:
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

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

# Frame loading mode for SAM2 VideoPredictor
# Set to "sync" for synchronous loading (safer, slower) or "async" for asynchronous (faster, may have threading issues)
FRAME_LOADING_MODE = os.getenv("SAM2_FRAME_LOADING_MODE", "sync").lower()
print(f"SAM2 frame loading mode: {FRAME_LOADING_MODE}")

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


def num_digits_for_n_files(n: int) -> int:
    return len(str(n - 1))


def get_system_memory_info():
    cpu_mem = psutil.virtual_memory()
    gpu_mem = None

    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated(0) / 1e9
        gpu_reserved = torch.cuda.memory_reserved(0) / 1e9
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_mem = {
            "allocated_gb": gpu_allocated,
            "reserved_gb": gpu_reserved,
            "total_gb": gpu_total,
            "free_gb": gpu_total - gpu_allocated
        }

    return {
        "cpu_used_gb": cpu_mem.used / 1e9,
        "cpu_available_gb": cpu_mem.available / 1e9,
        "cpu_total_gb": cpu_mem.total / 1e9,
        "cpu_percent": cpu_mem.percent,
        "gpu": gpu_mem
    }


def check_async_frame_loader_exception(inference_state):
    images = inference_state.get("images")
    if images is None:
        return None

    # Check if it's an AsyncVideoFrameLoader instance
    if hasattr(images, 'exception'):
        return images.exception

    return None


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

        if predictor_type == "ImagePredictor":
            loaded_predictor = SAM2ImagePredictor(build_sam2(config_file, checkpoint_path, device=device))
        elif predictor_type == "VideoPredictor":
            loaded_predictor = build_sam2_video_predictor(config_file, checkpoint_path,
                                                          device=device, vos_optimized=False)
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

    with tf.TiffFile(img_path) as tif:
        im = Image.fromarray(tif.pages[img_page].asarray() // 255).convert("RGB")
    im.save(target_path.with_suffix(".jpg"), format="JPEG", quality=90)


def downsample(img_path: Path, target_path: Path, img_page: int = 0):
    with tf.TiffFile(img_path) as tif:
        im = tif.pages[img_page].asarray()

    if im.ndim == 2:
        im = np.expand_dims(im, axis=-1)

    if im.shape[-1] == 1:
        im = np.repeat(im, 3, axis=-1)
    elif im.shape[-1] > 3:
        im = im[..., :3]

    # TODO: Be less stupid conversion.
    tf.imwrite(target_path.with_suffix(".tif"), (im // 255).astype(np.uint8))


async def run_pipeline(job_id: str, host_path: str, output_path: str, model_type: str, predictor_type: str):
    try:
        host_source = get_pure_path(host_path)
        host_result = get_pure_path(output_path)
        volume_source = Path(INTERNAL_VOLUME_PATH, PLUGIN_NAME, host_source.name)
        volume_folder = Path(INTERNAL_VOLUME_PATH, PLUGIN_NAME, f"{host_source.stem}_temp")
        volume_result = Path(INTERNAL_VOLUME_PATH, PLUGIN_NAME, "Segmentation", host_result.name)

        # Clear out and create temporary folder.
        if volume_folder.exists():
            shutil.rmtree(volume_folder)
        volume_folder.mkdir(parents=True, exist_ok=True)

        # 1. Transfer
        update_step(job_id, 0, 10)
        success, msg = await copy_to_volume([{"sourcePath": str(host_source), "targetPath": ""}])
        if not success:
            raise Exception(f"Copy failed: {msg}")
        update_step(job_id, 0, 50)
        await asyncio.sleep(0.001)

        # 2. Inference
        if not os.path.exists(volume_source):
            raise FileNotFoundError(f"File missing: {volume_source}")

        if volume_source.is_dir():
            zero_count = num_digits_for_n_files(len(list(volume_source.iterdir())))
            convert_args = [(path, volume_folder.joinpath(f"{str(i).zfill(zero_count)}"))
                            for i, path in enumerate(volume_source.iterdir())]
            with tf.TiffFile(next(volume_folder.iterdir())) as img:
                input_shape = (len(list(volume_folder.iterdir())), img.pages[0].shape[0], img.pages[0].shape[1])
        else:
            with tf.TiffFile(volume_source) as img:
                zero_count = num_digits_for_n_files(len(img.pages))
                convert_args = [(volume_source, volume_folder.joinpath(f"{str(i).zfill(zero_count)}"), i)
                                for i in range(len(img.pages))]
                input_shape = (len(img.pages), img.pages[0].shape[0], img.pages[0].shape[1])

        input_point = np.array([[input_shape[1] // 2, input_shape[2] // 2]]).astype(np.float32)
        input_label = np.array([1]).astype(np.int32)

        with Pool(8) as pool:
            if predictor_type == "ImagePredictor":
                pool.starmap(downsample, convert_args)
            elif predictor_type == "VideoPredictor":
                pool.starmap(jpeg_convert, convert_args)
        update_step(job_id, 0, 100)
        await asyncio.sleep(0.001)
        print("Predictor Created.")

        update_step(job_id, 1, 0)
        # Load Model (will look in chkpts dir)
        try:
            predictor = get_predictor(model_type, predictor_type)
            update_step(job_id, 1, 1)
            await asyncio.sleep(0.001)
            print("Predictor Initialized.")
        except Exception as e:
            print(f"Model Init Error: {e}")
            raise e
        await asyncio.sleep(0.001)

        result_stack = np.zeros(input_shape, dtype=np.uint8)

        if predictor_type == "VideoPredictor":
            # Initialize video predictor state
            # Use CPU offloading to handle large video volumes without OOM
            use_async_loading = FRAME_LOADING_MODE == "async"
            print(f"Initializing VideoPredictor state with {input_shape[0]} frames...")
            print(f"Frame loading mode: {'async' if use_async_loading else 'sync'}")
            mem_before = get_system_memory_info()
            print(f"Memory before init_state: {mem_before}")

            try:
                inference_state = predictor.init_state(
                    str(volume_folder),
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=False,
                    async_loading_frames=use_async_loading
                )
                mem_after = get_system_memory_info()
                print(f"Memory after init_state: {mem_after}")
            except Exception as e:
                print(f"Unexpected error during init_state: {type(e).__name__}: {e}\n"
                      f"Memory at error: {get_system_memory_info()}\n{traceback.print_exc()}")
                raise e

            update_step(job_id, 1, 10)
            await asyncio.sleep(0.001)
            print("Video Predictor Initial State Set.")

            annotation_frames = list(range(0, input_shape[0], 200))
            print(f"Annotation frames to process: {annotation_frames} from {input_shape}")

            if not annotation_frames:
                raise RuntimeError("No annotation frames generated - video may be too short")

            print(f"Point set shape: {input_point.shape}, Labels shape: {input_label.shape}")
            print(f"Adding points to {len(annotation_frames)} frames...")

            added_count = 0

            for i, frame in enumerate(annotation_frames):
                try:
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame,
                        obj_id=1,
                        points=input_point,
                        labels=input_label,
                    )
                    added_count += 1
                    print(f"Added points to frame {frame}")
                    update_step(job_id, 1, 10 + i / (len(annotation_frames) * 10))
                    await asyncio.sleep(0.001)
                except Exception as e:
                    print(f"Error adding points to frame {frame}: {type(e).__name__}: {e}\n{traceback.print_exc()}")
                    raise

            print(f"Video Predictor Annotations Added: {added_count} frames annotated.")

            try:
                for frame, _obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
                    # Check for background thread exceptions every frame
                    async_exception = check_async_frame_loader_exception(inference_state)
                    if async_exception is not None:
                        raise RuntimeError(
                            f"Background frame loader thread crashed at frame {frame}: "
                            f"{type(async_exception).__name__}: {async_exception}"
                        )

                    try:
                        result_stack[frame] = (mask_logits[0] > 0.0).cpu().numpy()
                        pct = int(((frame + 1) / input_shape[0]) * 100)
                        update_step(job_id, 1, 20 + 0.8 * pct)
                        await asyncio.sleep(0.001)
                        if frame % 10 == 0:
                            print(f"Memory:{get_system_memory_info()}|\nThreading at watch: {threading.active_count()}")
                    except Exception as e:
                        print(f"Error propogating video at frame {frame}: {type(e).__name__}: {e}\n"
                              f"Memory at error: {get_system_memory_info()}\n{traceback.print_exc()}\n"
                              f"Threading at error: {threading.active_count()}")
                        raise
            except Exception as e:
                print(f"Error adding points to frame {frame}: {type(e).__name__}: {e}\n"
                      f"Memory at error: {get_system_memory_info()}\n{traceback.print_exc()}\n"
                      f"Threading at error: {threading.active_count()}")
                raise
        else:
            for i, img_path in enumerate(list(volume_folder.iterdir())):
                predictor.set_image(tf.imread(img_path))
                masks, _scores, _logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False,
                )

                if masks.ndim == 3:
                    masks = masks[0]

                result_stack[i] = (masks > 0).astype(np.uint8) * 255
                pct = int(((i + 1) / input_shape[0]) * 100)
                update_step(job_id, 1, pct)
                await asyncio.sleep(0.001)

        update_step(job_id, 1, 100)

        # 3. Saving
        volume_result.parent.mkdir(exist_ok=True, parents=True)
        tf.imwrite(volume_result, result_stack)
        update_step(job_id, 2, 50)

        # Robust copy back
        print(f"Copying {volume_result} back to {host_result}")
        success, msg = await copy_to_host([{"sourcePath": str(host_result), "targetPath": "Segmentation"}])
        if not success:
            raise Exception(f"Copy back failed: {msg}")
        update_step(job_id, 2, 100)

        print(f"Cleaning Up Temporary Dir {volume_folder}")
        shutil.rmtree(volume_folder)

        jobs[job_id]["status"] = "completed"

    except Exception as e:
        print(f"Pipeline Error: {e}")
        import traceback
        traceback.print_exc()
        jobs[job_id]["status"] = "error"
