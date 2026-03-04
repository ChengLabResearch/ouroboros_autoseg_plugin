import os
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download
import requests
import torch

from . import config


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
    build_sam3_video_predictor = None
    build_sam3_image_model = None
    Sam3Processor = None


def models_available() -> bool:
    return (
        (build_sam2 is not None and SAM2ImagePredictor is not None)
        or (build_sam3_video_predictor is not None and Sam3Processor is not None)
    )


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
    url = f"{config.VOLUME_SERVER_URL}/{path}"
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
        "pluginFolderName": config.PLUGIN_NAME,
        "files": files,
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
        "pluginFolderName": config.PLUGIN_NAME,
        "files": files,
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
    config.ensure_checkpoint_dir()
    target_path = Path(dest)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Recover from older bug where checkpoint path was created as a directory.
    if target_path.exists() and target_path.is_dir():
        print(f"Found directory at checkpoint path {target_path}, removing it before download.")
        shutil.rmtree(target_path)

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
    else:
        raise Exception(f"Failed to download checkpoint: {response.status_code}")


def download_sam3_checkpoint(model_name: str, hf_token: str, target_path: str):
    repo_id = "facebook/sam3"
    filename = f"{model_name}.pt"
    print(f"Fetching {filename} from Hugging Face...")
    cached_path = hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token)
    shutil.copy(cached_path, target_path)


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
    cache_key = f"{model_name}_{predictor_type}"

    if config.loaded_model_name == cache_key and config.loaded_predictor is not None:
        return config.loaded_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model: {model_name} (Predictor: {predictor_type}) on {device}...")

    config.ensure_checkpoint_dir()
    checkpoint_name = f"{model_name}.pt"
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, checkpoint_name)

    if model_name in config.SAM2_CONFIGS:
        if SAM2ImagePredictor is None:
            raise ImportError("SAM2 library not found/installed")

        if not os.path.isfile(checkpoint_path):
            print(f"Checkpoint not found in {checkpoint_path}, attempting auto-download...")
            try:
                download_file(config.SAM2_URLS[model_name], checkpoint_path)
            except Exception as e:
                raise RuntimeError(f"Model not found and download failed: {e}")

        config_file = config.SAM2_CONFIGS[model_name]

        if predictor_type == "ImagePredictor":
            config.loaded_predictor = SAM2ImagePredictor(build_sam2(config_file, checkpoint_path, device=device))
        elif predictor_type == "VideoPredictor":
            config.loaded_predictor = build_sam2_video_predictor(
                config_file,
                checkpoint_path,
                device=device,
                vos_optimized=False,
            )
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")

    elif model_name.startswith("sam3"):
        if Sam3Processor is None:
            raise ImportError("SAM3 library not found/installed")

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"SAM3 checkpoint not found at {checkpoint_path}. "
                "Please use the 'Models' section to download it with your authentication token."
            )

        if predictor_type == "ImagePredictor":
            config.loaded_predictor = Sam3Processor(
                build_sam3_image_model(checkpoint_path=checkpoint_path, device=device)
            )
        elif predictor_type == "VideoPredictor":
            config.loaded_predictor = build_sam3_video_predictor(checkpoint_path=checkpoint_path, device=device)
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")

    else:
        raise ValueError(f"Unknown model type: {model_name}")

    config.loaded_model_name = cache_key
    return config.loaded_predictor
