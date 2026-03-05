import json
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
import shutil
import time
from typing import Optional

import numpy as np
from PIL import Image
import psutil
from pydantic import BaseModel
import tifffile as tf
import torch

from . import config


class DownloadRequest(BaseModel):
    model_type: str
    hf_token: Optional[str] = None


class ProcessRequest(BaseModel):
    file_path: str
    output_file: str
    model_type: str
    predictor_type: str


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
            "shm_percent": (used / total) * 100,
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
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        print(f"Invalid annotation_points in TIFF metadata from {tiff_path}: {type(e).__name__}: {e}")
        return None

    if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
        return None
    return arr[:, :3]


def default_annotations(input_shape: tuple[int, int, int]) -> np.ndarray:
    """
    Create fallback annotations based on periodic center points when specific annotation points are missing.
    As annotations are at the start and end point, iteration starts and 0 and always adds on an endcap.
    (This can create a double point in rare situations but that should not be an issue).

    Parameters
    ----------
    input_shape : tuple
        X,Y,Z shape of straightened volume.

    Returns
    -------
    np.ndarray
        Array with shape ``(N, 3)`` containing ``[x, y, z]`` annotation points at center points,
        with interval based on config.ANNOTATION_INTERVAL
    """
    if input_shape[0]:
        return np.array([(input_shape[2] // 2, input_shape[1] // 2, z)
                        for z in range(0, input_shape[0], config.FALLBACK_ANNOTATION_INTERVAL)]
                        + [(input_shape[2] // 2, input_shape[1] // 2, input_shape[0] - 1)],
                        dtype=np.float32)
    else:
        return np.empty((0, 3), dtype=np.float32)


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
            "free_gb": gpu_total - gpu_allocated,
        }

    return {
        "cpu_used_gb": cpu_mem.used / 1e9,
        "cpu_available_gb": cpu_mem.available / 1e9,
        "cpu_total_gb": cpu_mem.total / 1e9,
        "cpu_percent": cpu_mem.percent,
        "gpu": gpu_mem,
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
    if hasattr(images, "exception"):
        return images.exception

    return None


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
    if "\\" in stripped or (len(stripped) > 1 and stripped[1] == ":"):
        return PureWindowsPath(stripped)
    return PurePosixPath(stripped)


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
    if job_id in config.jobs:
        config.jobs[job_id]["steps"][step_index]["progress"] = progress
        config.jobs[job_id]["updated_at"] = time.time()


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
