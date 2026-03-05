import asyncio
from multiprocessing import Pool
import os
from pathlib import Path
import shutil
import threading
import time
import traceback

import numpy as np
import tifffile as tf

from ..util import config
from ..util.network import copy_to_host, copy_to_volume, get_predictor
from ..util.util import (
    _annotation_point_for_frame,
    _annotation_samples_for_video,
    _sorted_tif_paths,
    check_async_frame_loader_exception,
    downsample,
    get_pure_path,
    get_shared_memory_info,
    get_system_memory_info,
    jpeg_convert,
    load_annotation_points,
    num_digits_for_n_files,
    update_step,
    default_annotations
)


async def run_video_predictor(
    predictor,
    volume_folder: Path,
    input_shape: tuple[int, int, int],
    annotation_points: np.ndarray,
    input_label: np.ndarray,
    result_stack: np.ndarray,
    job_id: str,
):
    # Initialize video predictor state
    # Use CPU offloading to handle large video volumes without OOM
    use_async_loading = config.FRAME_LOADING_MODE == "async"
    print(f"Initializing VideoPredictor state with {input_shape[0]} frames...")
    print(f"Frame loading mode: {'async' if use_async_loading else 'sync'}")
    mem_before = get_system_memory_info()
    print(f"Memory before init_state: {mem_before}")

    try:
        inference_state = predictor.init_state(
            str(volume_folder),
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
            async_loading_frames=use_async_loading,
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

    video_annotation_samples = _annotation_samples_for_video(annotation_points, input_shape[0])

    print(f"Annotation frames to process: {len(video_annotation_samples)} from {input_shape}")

    if not video_annotation_samples:
        if annotation_points is not None:
            print("All metadata annotation z-values were out of bounds, using center-point fallback.")
            video_annotation_samples = _annotation_samples_for_video(default_annotations(input_shape))
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


async def run_image_predictor(
    predictor,
    volume_folder: Path,
    annotation_points: np.ndarray,
    input_label: np.ndarray,
    input_shape: tuple[int, int, int],
    result_stack: np.ndarray,
    job_id: str,
):
    for i, img_path in enumerate(sorted(volume_folder.iterdir())):
        predictor.set_image(tf.imread(img_path))
        masks, _scores, _logits = predictor.predict(
            point_coords=_annotation_point_for_frame(annotation_points, i),
            point_labels=input_label,
            multimask_output=False,
        )

        if masks.ndim == 3:
            masks = masks[0]

        result_stack[i] = (masks > 0).astype(np.uint8) * 255
        pct = int(((i + 1) / input_shape[0]) * 100)
        update_step(job_id, 1, pct)
        await asyncio.sleep(0.001)


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
        volume_source = Path(config.INTERNAL_VOLUME_PATH, config.PLUGIN_NAME, host_source.name)
        volume_folder = Path(config.INTERNAL_VOLUME_PATH, config.PLUGIN_NAME, f"{host_source.stem}_temp")
        volume_result = Path(config.INTERNAL_VOLUME_PATH, config.PLUGIN_NAME, "Segmentation", host_result.name)

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

        input_label = np.array([1]).astype(np.int32)
        annotation_points = load_annotation_points(volume_source)
        if annotation_points is None:
            print("No annotation points in TIFF metadata, using center-point fallback.")
            annotation_points = default_annotations(input_shape)
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
            await run_video_predictor(
                predictor=predictor,
                volume_folder=volume_folder,
                input_shape=input_shape,
                annotation_points=annotation_points,
                input_label=input_label,
                result_stack=result_stack,
                job_id=job_id,
            )
        else:
            await run_image_predictor(
                predictor=predictor,
                volume_folder=volume_folder,
                annotation_points=annotation_points,
                input_label=input_label,
                input_shape=input_shape,
                result_stack=result_stack,
                job_id=job_id,
            )

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

        config.jobs[job_id]["status"] = "completed"
        config.jobs[job_id]["updated_at"] = time.time()

    except Exception as e:
        print(f"Pipeline Error: {e}")
        traceback.print_exc()
        config.jobs[job_id]["status"] = "error"
        config.jobs[job_id]["updated_at"] = time.time()
