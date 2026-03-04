import asyncio
import os
import shutil
import time
import traceback
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .pipeline.pipeline import run_pipeline
from .util import config, network
from .util.util import DownloadRequest, ProcessRequest

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def refresh_startup_status():
    """
    Refresh startup status based on currently reachable dependencies.

    Returns
    -------
    dict
        Updated startup status object
    """
    # If this API is reachable, the container image has already started.
    config.update_initialization_step("Building Docker Image", "completed")

    try:
        response = network.requests.get(f"{config.VOLUME_SERVER_URL}/", timeout=1)
        if response.ok or response.status_code in (404, 405):
            config.update_initialization_step("Connecting to Volume Server", "completed")
        else:
            config.update_initialization_step("Connecting to Volume Server", "warning")
    except network.requests.RequestException:
        config.update_initialization_step("Connecting to Volume Server", "warning")

    config.update_initialization_step(
        "Initializing ML Models",
        "completed" if network.models_available() else "warning",
    )

    all_steps_completed = all(
        step["status"] == "completed" for step in config.startup_status["initialization_steps"]
    )
    config.startup_status["is_ready"] = all_steps_completed
    if all_steps_completed and config.startup_status["ready_time"] is None:
        config.startup_status["ready_time"] = time.time()
    if not all_steps_completed:
        config.startup_status["ready_time"] = None

    return config.startup_status


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
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}.pt")
        statuses[model_name] = os.path.isfile(checkpoint_path)

    return {
        "checkpoint_dir": config.CHECKPOINT_DIR,
        "models": statuses,
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
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        target_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}.pt")

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
        if model_name in config.SAM2_URLS:
            network.download_file(config.SAM2_URLS[model_name], target_path)
            print(f"Downloaded {model_name}")
            await asyncio.sleep(0.0001)
            return {"status": "success", "message": f"Downloaded {model_name}"}

        # SAM 3 Download
        elif model_name.startswith("sam3"):
            if not req.hf_token:
                raise HTTPException(400, "Authentication Token required for SAM 3")

            try:
                network.download_sam3_checkpoint(model_name, req.hf_token, target_path)
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
    config.jobs[job_id] = {
        "status": "running",
        "steps": [
            {"name": "Transferring", "progress": 0},
            {"name": "Inference", "progress": 0},
            {"name": "Saving", "progress": 0},
        ],
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    background_tasks.add_task(
        run_pipeline,
        job_id,
        req.file_path,
        req.output_file,
        req.model_type,
        req.predictor_type,
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
    if job_id not in config.jobs:
        raise HTTPException(404, "Job not found")
    return config.jobs[job_id]


@app.get("/latest-job")
async def get_latest_job():
    """
    Return the most recently created running job, if any.
    """
    running_jobs = [
        (job_id, data)
        for job_id, data in config.jobs.items()
        if data.get("status") == "running"
    ]
    if not running_jobs:
        return {"job_id": None, "status": "none"}
    latest_job_id, latest_job = max(
        running_jobs, key=lambda item: item[1].get("created_at", 0)
    )
    return {"job_id": latest_job_id, "status": latest_job.get("status", "running")}
