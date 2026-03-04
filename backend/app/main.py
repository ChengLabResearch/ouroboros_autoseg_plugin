import asyncio
import json
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
from fastapi.responses import JSONResponse
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
    from sam3.model_builder import build_sam3_video_predictor, build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as ie:
    print("SAM3 not installed.")
    print(f"{ie}:{ie.name}|{ie.path}")
    build_sam3 = None
    Sam3Processor = None

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


def refresh_startup_status():
    """
    Refresh startup status based on currently reachable dependencies.

    Returns
    -------
    dict
        Updated startup status object
    """
    global startup_status

    # If this API is reachable, the container image has already started.
    update_initialization_step("Building Docker Image", "completed")

    try:
        response = requests.get(f"{VOLUME_SERVER_URL}/", timeout=1)
        if response.ok or response.status_code in (404, 405):
            update_initialization_step("Connecting to Volume Server", "completed")
        else:
            update_initialization_step("Connecting to Volume Server", "warning")
    except requests.RequestException:
        update_initialization_step("Connecting to Volume Server", "warning")

    models_available = (
        (build_sam2 is not None and SAM2ImagePredictor is not None)
        or (build_sam3 is not None and Sam3Processor is not None)
    )
    update_initialization_step("Initializing ML Models", "completed" if models_available else "warning")

    all_steps_completed = all(step["status"] == "completed" for step in startup_status["initialization_steps"])
    startup_status["is_ready"] = all_steps_completed
    if all_steps_completed and startup_status["ready_time"] is None:
        startup_status["ready_time"] = time.time()
    if not all_steps_completed:
        startup_status["ready_time"] = None

    return startup_status


def get_shared_memory_info():
    """
    Get usage statistics for /dev/shm (Shared Memory).
    """
    try:
        total, used, free = shutil.disk_usage("/dev/shm")
        return {
            "shm_total_gb": total / 1e9,
            "shm_used_gb": used / 1e9,
            "shm_free_gb": free / 1e9,
            "shm_percent": (used / total) * 100
        }
    except FileNotFoundError:
        return {"error": "/dev/shm not found"}


def num_digits_for_n_files(n: int) -> int:
    """
    Return zero-padding width required to index `n` files from zero.

    Parameters
    ----------
    n : int
        Total number of files.

    Returns
    -------
    int
        Number of digits required to represent the largest index (`n - 1`).
    """
    return len(str(n - 1))


def _sorted_tif_paths(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in {".tif", ".tiff"})


def _read_annotation_points_from_tiff(tiff_path: Path) -> Optional[np.ndarray]:
    """
    Read annotation points from TIFF ImageDescription metadata.

    The expected metadata schema (from ouroboros) stores points as:
    {"annotation_points": [[x, y, z], ...]}
    """
    try:
        with tf.TiffFile(tiff_path) as tif:
            metadata = json.loads(tif.pages[0].description)
    except Exception as e:
        print(f"Unable to parse TIFF metadata from {tiff_path}: {type(e).__name__}: {e}")
        return None

    try:
        arr = np.asarray(metadata["annotation_points"], dtype=np.float32)
    except (AttributeError, TypeError, ValueError):
        return None

    if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
        return None
    return arr[:, :3]


def load_annotation_points(volume_source: Path) -> Optional[np.ndarray]:
    """
    Load annotation points from a straightened-volume input.

    For multi-file TIFF outputs, points are expected on the first TIFF file metadata.

    Parameters
    ----------
    volume_source : pathlib.Path
        Path to a single TIFF file or a directory containing TIFF slices.

    Returns
    -------
    numpy.ndarray or None
        Array with shape ``(N, 3)`` containing ``[x, y, z]`` annotation points,
        or ``None`` when metadata is missing or invalid.
    """
    if volume_source.is_dir():
        tif_paths = _sorted_tif_paths(volume_source)
        if not tif_paths:
            return None
        return _read_annotation_points_from_tiff(tif_paths[0])
    return _read_annotation_points_from_tiff(volume_source)


def _annotation_samples_for_video(annotation_points: np.ndarray,
                                  num_frames: int) -> list[tuple[int, np.ndarray]]:
    """
    Convert XYZ annotation points into per-frame XY prompts for video predictor.
    """
    frame_to_points = {}
    for x, y, z in annotation_points:
        frame_idx = int(np.rint(z))
        if frame_idx < 0 or frame_idx >= num_frames:
            continue
        frame_to_points.setdefault(frame_idx, []).append([x, y])

    return [(frame_idx, np.asarray(points, dtype=np.float32))
            for frame_idx, points in sorted(frame_to_points.items(), key=lambda item: item[0])]


def _annotation_point_for_frame(annotation_points: np.ndarray, frame_idx: int) -> np.ndarray:
    """
    Interpolate an XY point for a frame from nearest annotation points in Z.
    """
    z = annotation_points[:, 2]
    x = annotation_points[:, 0]
    y = annotation_points[:, 1]

    order = np.argsort(z)
    z = z[order]
    x = x[order]
    y = y[order]

    if len(z) == 1:
        return np.array([[x[0], y[0]]], dtype=np.float32)

    right = int(np.searchsorted(z, frame_idx, side="left"))
    if right <= 0:
        return np.array([[x[0], y[0]]], dtype=np.float32)
    if right >= len(z):
        return np.array([[x[-1], y[-1]]], dtype=np.float32)

    left = right - 1
    z0, z1 = z[left], z[right]
    if z1 == z0:
        return np.array([[x[right], y[right]]], dtype=np.float32)

    t = (frame_idx - z0) / (z1 - z0)
    px = x[left] + t * (x[right] - x[left])
    py = y[left] + t * (y[right] - y[left])
    return np.array([[px, py]], dtype=np.float32)


def get_system_memory_info():
    """
    Get current system memory usage statistics.

    Returns
    -------
    dict
        Dictionary with CPU and GPU memory stats in GB
    """
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
    """
    Check if the async frame loader thread encountered an exception.

    SAM2's AsyncVideoFrameLoader stores background thread exceptions in self.exception.
    This function directly accesses that to catch thread crashes that won't propagate
    to the main thread via normal exception handling.

    Parameters
    ----------
    inference_state : dict
        The inference state dict returned by predictor.init_state()

    Returns
    -------
    Exception or None
        The exception from the async loader thread, if any

    """
    images = inference_state.get("images")
    if images is None:
        return None

    # Check if it's an AsyncVideoFrameLoader instance
    if hasattr(images, 'exception'):
        return images.exception

    return None


def mark_initialization_complete():
    """
    Mark the service as fully initialized and complete all pending initialization steps.

    Updates the global startup_status to indicate the service is ready for processing.
    Sets the ready_time timestamp and marks all pending initialization steps as completed.
    """
    global startup_status
    startup_status["is_ready"] = True
    startup_status["ready_time"] = time.time()
    for step in startup_status["initialization_steps"]:
        if step["status"] == "pending":
            step["status"] = "completed"
    print("Service initialization complete!")


def update_initialization_step(step_name: str, status: str):
    """
    Update the status of a specific initialization step.

    Parameters
    ----------
    step_name : str
        Name of the initialization step to update
    status : str
        New status for the step (e.g., 'pending', 'in_progress', 'completed', 'warning')
    """
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
    """
    Robustly parse a file path string and return the appropriate PurePath object.

    Handles Windows and POSIX path formats, including mixed scenarios.
    Strips surrounding quotes before parsing.

    Parameters
    ----------
    path_str : str
        Path string to parse (may contain quotes)

    Returns
    -------
    PurePath
        PureWindowsPath if Windows path detected, otherwise PurePosixPath
    """
    stripped = path_str.strip('"').strip("'")
    if '\\' in stripped or (len(stripped) > 1 and stripped[1] == ':'):
        return PureWindowsPath(stripped)
    return PurePosixPath(stripped)


async def request_volume_server(path, data):
    """
    Send a POST request to the volume server to perform file operations.

    Parameters
    ----------
    path : str
        API endpoint path on the volume server
    data : dict
        JSON data to send in the request body

    Returns
    -------
    tuple
        (success: bool, message: str) - Success status and response/error message
    """
    url = f"{VOLUME_SERVER_URL}/{path}"
    try:
        result = requests.post(url, headers={"Content-Type": "application/json"}, json=data)
        print(f"Data: {data}")
        print(f"Result: {result.text}")
        return (True, "") if result.ok else (False, result.text)
    except Exception as error:
        return False, str(error)


async def copy_to_volume(files):
    """
    Copy files from the host to the ouroboros volume.

    Parameters
    ----------
    files : list
        List of file path dictionaries with 'sourcePath' and 'targetPath' keys

    Returns
    -------
    tuple
        (success: bool, message: str) - Result of the copy operation
    """
    return await request_volume_server("copy-to-volume", {
        "volumeName": "ouroboros-volume",
        "pluginFolderName": PLUGIN_NAME,
        "files": files
    })


async def copy_to_host(files):
    """
    Copy files from the ouroboros volume back to the host.

    Parameters
    ----------
    files : list
        List of file path dictionaries with 'sourcePath' and 'targetPath' keys

    Returns
    -------
    tuple
        (success: bool, message: str) - Result of the copy operation
    """
    return await request_volume_server("copy-to-host", {
        "volumeName": "ouroboros-volume",
        "pluginFolderName": PLUGIN_NAME, "files": files
    })


def download_file(url, dest):
    """
    Download a file from a URL and save it to the specified destination.

    Parameters
    ----------
    url : str
        URL of the file to download
    dest : str
        Destination file path

    Raises
    ------
    Exception
        If the download fails with a non-200 status code
    """
    print(f"Downloading {url} to {dest}...")
    target_path = Path(dest)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Recover from older bug where checkpoint path was created as a directory.
    if target_path.exists() and target_path.is_dir():
        print(f"Found directory at checkpoint path {target_path}, removing it before download.")
        shutil.rmtree(target_path)

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
    else:
        raise Exception(f"Failed to download checkpoint: {response.status_code}")

# --- Model Loading Logic ---


def get_predictor(model_name: str, predictor_type: str):
    """
    Load and cache a segmentation predictor model.

    Supports SAM2 (with ImagePredictor or VideoPredictor) and SAM3 models.
    Implements model caching to avoid reloading the same model.
    Auto-downloads SAM2 models if not found locally.

    Parameters
    ----------
    model_name : str
        Name of the model to load (e.g., 'sam2_hiera_base_plus', 'sam3')
    predictor_type : str
        Type of predictor - 'ImagePredictor' or 'VideoPredictor'

    Returns
    -------
    Predictor
        Initialized SAM2ImagePredictor, SAM2VideoPredictor, or Sam3Processor

    Raises
    ------
    ImportError
        If required library not installed
    RuntimeError
        If model download fails
    FileNotFoundError
        If SAM3 checkpoint not found
    ValueError
        If model or predictor type is unknown
    """
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
        if not os.path.isfile(checkpoint_path):
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
        if Sam3Processor is None:
            raise ImportError("SAM3 library not found/installed")

        # Strict Check: SAM3 must be downloaded via the UI first
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"SAM3 checkpoint not found at {checkpoint_path}. "
                "Please use the 'Models' section to download it with your authentication token."
            )

        if predictor_type == "ImagePredictor":
            loaded_predictor = Sam3Processor(build_sam3_image_model(checkpoint_path=checkpoint_path, device=device))
        elif predictor_type == "VideoPredictor":
            loaded_predictor = build_sam3_video_predictor(checkpoint_path=checkpoint_path, device=device)

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


@app.get("/")
async def server_active():
    return JSONResponse("Segmentation server is active")


@app.get("/startup-status")
async def get_startup_status():
    return refresh_startup_status()


@app.get("/model-status")
async def get_model_status():
    """
    Report whether key model checkpoints are present on the mounted volume.

    Returns
    -------
    dict
        Availability flags per model and checkpoint directory path.
    """
    tracked_models = ["sam2_hiera_base_plus", "sam3"]
    statuses = {}
    for model_name in tracked_models:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}.pt")
        statuses[model_name] = os.path.isfile(checkpoint_path)

    return {
        "checkpoint_dir": CHECKPOINT_DIR,
        "models": statuses
    }


@app.post("/download-model")
async def download_model(req: DownloadRequest):
    """
    Download a segmentation model checkpoint.

    Downloads SAM2 models from Meta's public URLs or SAM3 models from Hugging Face.
    Saves the model to the persistent checkpoint directory.

    Parameters
    ----------
    req : DownloadRequest
        Request containing model_type and optional hf_token

    Returns
    -------
    dict
        Status dict with 'status' and 'message' keys

    Raises
    ------
    HTTPException
        If token required but not provided, or download fails
    """
    try:
        model_name = req.model_type
        target_path = os.path.join(CHECKPOINT_DIR, f"{model_name}.pt")

        if os.path.isfile(target_path):
            print(f"Model {model_name} already exists.")
            await asyncio.sleep(0.0001)
            return {"status": "exists", "message": f"Model {model_name} already exists."}

        # Recover from old bug: remove accidental directory at checkpoint file path.
        if os.path.isdir(target_path):
            print(f"Removing directory at checkpoint path: {target_path}")
            shutil.rmtree(target_path)

        print(f"Initiating download for {model_name}...")

        # SAM 2 Download
        if model_name in SAM2_URLS:
            download_file(SAM2_URLS[model_name], target_path)
            print(f"Downloaded {model_name}")
            await asyncio.sleep(0.0001)
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

                await asyncio.sleep(0.0001)
                return {"status": "success", "message": f"Downloaded {model_name}"}
            except Exception as e:
                raise HTTPException(500, f"Hugging Face Download Failed: {str(e)}")

        else:
            raise HTTPException(400, "Unknown model type")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/process-stack")
async def process_stack(req: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Submit a segmentation processing job.

    Creates a new background job to process an image stack with the specified model and predictor.
    Returns immediately with a job ID for status tracking.

    Parameters
    ----------
    req : ProcessRequest
        Request containing file paths, model type, and predictor type
    background_tasks : BackgroundTasks
        FastAPI background tasks manager

    Returns
    -------
    dict
        Job information with 'job_id' and 'status' keys
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "running",
        "steps": [
            {"name": "Transferring", "progress": 0},
            {"name": "Inference", "progress": 0},
            {"name": "Saving", "progress": 0}
        ],
        "created_at": time.time(),
        "updated_at": time.time()
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
    """
    Get the current status of a processing job.

    Parameters
    ----------
    job_id : str
        Unique identifier of the job

    Returns
    -------
    dict
        Job status including overall status and progress for each step

    Raises
    ------
    HTTPException
        404 if job not found
    """
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/latest-job")
async def get_latest_job():
    """
    Return the most recently created running job, if any.
    """
    running_jobs = [
        (job_id, data)
        for job_id, data in jobs.items()
        if data.get("status") == "running"
    ]
    if not running_jobs:
        return {"job_id": None, "status": "none"}
    latest_job_id, latest_job = max(
        running_jobs, key=lambda item: item[1].get("created_at", 0)
    )
    return {"job_id": latest_job_id, "status": latest_job.get("status", "running")}


def update_step(job_id, step_index, progress):
    """
    Update the progress of a specific step in a processing job.

    Parameters
    ----------
    job_id : str
        Unique identifier of the job
    step_index : int
        Index of the step to update (0=Transferring, 1=Inference, 2=Saving)
    progress : int
        Progress percentage (0-100)
    """
    if job_id in jobs:
        jobs[job_id]["steps"][step_index]["progress"] = progress
        jobs[job_id]["updated_at"] = time.time()


def jpeg_convert(img_path: Path, target_path: Path, img_page: int = 0):
    """
    Convert an image file to JPEG format for video predictor processing.

    Used in multiprocessing to prepare image frames for the SAM2 VideoPredictor.

    Parameters
    ----------
    img_path : Path
        Path to the image file to convert.
    target_path : Path
        Path to write converted image to.
    img_page : Path
        Page of the input to read from, for use with single-file, multipage TIFFs.
    """
    with tf.TiffFile(img_path) as tif:
        im = Image.fromarray(tif.pages[img_page].asarray() // 255).convert("RGB")
    im.save(target_path.with_suffix(".jpg"), format="JPEG", quality=90)


def downsample(img_path: Path, target_path: Path, img_page: int = 0):
    """
    Prepare an image for the SAM2 ImagePredictor by ensuring RGB format.

    Used in multiprocessing to convert grayscale or multi-channel images to 8-bit RGB format.

    Parameters
    ----------
    img_path : Path
        Path to the image file to process
    target_path : Path
        Path to write converted image to.
    img_page : Path
        Page of the input to read from, for use with single-file, multipage TIFFs.
    """
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
    """
    Execute the full segmentation pipeline for an image stack.

    Orchestrates the complete workflow:
    1. Transfer files from host to volume
    2. Preprocess image data and run segmentation inference
    3. Save results and transfer back to host

    Supports both SAM2 ImagePredictor (per-frame segmentation) and VideoPredictor (temporal coherence).
    Updates job status throughout execution for UI progress tracking.

    Parameters
    ----------
    job_id : str
        Unique job identifier for status tracking
    host_path : str
        Path to input image stack on host
    output_path : str
        Desired output path for segmentation results
    model_type : str
        Name of the model to use (e.g., 'sam2_hiera_base_plus')
    predictor_type : str
        Type of predictor - 'ImagePredictor' or 'VideoPredictor'

    Notes
    -----
    Updates global jobs dict with progress and final status (completed or error).
    """
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
            source_frame_paths = _sorted_tif_paths(volume_source)
            zero_count = num_digits_for_n_files(len(source_frame_paths))
            convert_args = [(path, volume_folder.joinpath(f"{str(i).zfill(zero_count)}"))
                            for i, path in enumerate(source_frame_paths)]
            with tf.TiffFile(source_frame_paths[0]) as img:
                input_shape = (len(source_frame_paths), img.pages[0].shape[0], img.pages[0].shape[1])
        else:
            with tf.TiffFile(volume_source) as img:
                zero_count = num_digits_for_n_files(len(img.pages))
                convert_args = [(volume_source, volume_folder.joinpath(f"{str(i).zfill(zero_count)}"), i)
                                for i in range(len(img.pages))]
                input_shape = (len(img.pages), img.pages[0].shape[0], img.pages[0].shape[1])

        input_point = np.array([[input_shape[1] // 2, input_shape[2] // 2]], dtype=np.float32)
        input_label = np.array([1]).astype(np.int32)
        annotation_points = load_annotation_points(volume_source)
        if annotation_points is None:
            print("No annotation points in TIFF metadata, using center-point fallback.")
        else:
            print(f"Loaded {len(annotation_points)} annotation points from TIFF metadata.")

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

            if annotation_points is not None:
                video_annotation_samples = _annotation_samples_for_video(annotation_points, input_shape[0])
            else:
                video_annotation_samples = [(frame, input_point) for frame in range(0, input_shape[0], 200)]

            annotation_frames = [frame for frame, _ in video_annotation_samples]
            print(f"Annotation frames to process: {annotation_frames} from {input_shape}")

            if not video_annotation_samples:
                if annotation_points is not None:
                    print("All metadata annotation z-values were out of bounds, using center-point fallback.")
                    video_annotation_samples = [(frame, input_point) for frame in range(0, input_shape[0], 200)]
                if not video_annotation_samples:
                    raise RuntimeError("No annotation frames generated - video may be too short")

            print(f"Labels shape: {input_label.shape}")
            print(f"Adding points to {len(video_annotation_samples)} frames...")

            added_count = 0

            for i, (frame, frame_points) in enumerate(video_annotation_samples):
                try:
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame,
                        obj_id=1,
                        points=frame_points,
                        labels=np.ones(len(frame_points), dtype=np.int32),
                    )
                    added_count += 1
                    print(f"Added points to frame {frame}")
                    update_step(job_id, 1, 10 + i / (len(video_annotation_samples) * 10))
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
                            print(f"Memory:{get_system_memory_info()}\n{get_shared_memory_info()}\n"
                                  f"Threading at watch: {threading.active_count()}")
                    except Exception as e:
                        print(f"Error propogating video at frame {frame}: {type(e).__name__}: {e}\n"
                              f"Memory at error: {get_system_memory_info()}\n{traceback.print_exc()}\n"
                              f"Threading at error: {threading.active_count()}")
                        raise
            except Exception as e:
                print(f"Error propogating video at frame {frame}: {type(e).__name__}: {e}\n"
                      f"Memory at error: {get_system_memory_info()}\n{traceback.print_exc()}\n"
                      f"Threading at error: {threading.active_count()}")
                raise
        else:
            for i, img_path in enumerate(sorted(volume_folder.iterdir())):
                if annotation_points is not None:
                    frame_point = _annotation_point_for_frame(annotation_points, i)
                else:
                    frame_point = input_point
                predictor.set_image(tf.imread(img_path))
                masks, _scores, _logits = predictor.predict(
                    point_coords=frame_point,
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
        jobs[job_id]["updated_at"] = time.time()

    except Exception as e:
        print(f"Pipeline Error: {e}")
        traceback.print_exc()
        jobs[job_id]["status"] = "error"
        jobs[job_id]["updated_at"] = time.time()
