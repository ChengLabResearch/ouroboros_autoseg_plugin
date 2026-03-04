import logging
import os
from pathlib import Path
import time
from typing import Dict


jobs: Dict[str, Dict] = {}
PLUGIN_NAME = "sam3-segmentation"
VOLUME_SERVER_URL = os.getenv("VOLUME_SERVER_URL", "http://host.docker.internal:3001")
logger = logging.getLogger(__name__)

SAM2_CONFIGS = {
    "sam2_hiera_tiny": "sam2_hiera_t.yaml",
    "sam2_hiera_small": "sam2_hiera_s.yaml",
    "sam2_hiera_base_plus": "sam2_hiera_b+.yaml",
    "sam2_hiera_large": "sam2_hiera_l.yaml",
}

SAM2_URLS = {
    "sam2_hiera_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
    "sam2_hiera_small": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
    "sam2_hiera_base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
    "sam2_hiera_large": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
}


def _running_in_docker() -> bool:
    return Path("/.dockerenv").exists() or os.getenv("RUNNING_IN_DOCKER") == "1"


INTERNAL_VOLUME_PATH = "/ouroboros-volume" if _running_in_docker() else os.getenv(
    "VOLUME_MOUNT_PATH", "/tmp/ouroboros-volume"
)
CHECKPOINT_DIR = os.path.join(INTERNAL_VOLUME_PATH, PLUGIN_NAME, "chkpts")

# Frame loading mode for SAM2 VideoPredictor
# Set to "sync" for synchronous loading (safer, slower) or "async" for asynchronous (faster, may have threading issues)
FRAME_LOADING_MODE = os.getenv("SAM2_FRAME_LOADING_MODE", "sync").lower()
print(f"SAM2 frame loading mode: {FRAME_LOADING_MODE}")

if _running_in_docker() and not os.path.isdir(INTERNAL_VOLUME_PATH):
    logger.error(
        "Expected mounted volume path %s not found. Ensure ouroboros-volume is mounted.",
        INTERNAL_VOLUME_PATH,
    )


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
        {"name": "Initializing ML Models", "status": "pending"},
    ],
    "start_time": None,
    "ready_time": None,
}


def ensure_checkpoint_dir():
    """
    Ensure the checkpoint directory exists.

    Directory creation is deferred until checkpoint download/use-time.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def mark_initialization_complete():
    """
    Mark the service as fully initialized and complete all pending initialization steps.

    Updates startup_status to indicate the service is ready for processing.
    Sets the ready_time timestamp and marks all pending initialization steps as completed.
    """
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
    if startup_status["start_time"] is None:
        startup_status["start_time"] = time.time()
    for step in startup_status["initialization_steps"]:
        if step["name"] == step_name:
            step["status"] = status
            print(f"Initialization: {step_name} -> {status}")
