import asyncio
import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import tifffile as tf

_TEST_VOLUME_ROOT = tempfile.mkdtemp(prefix="ouroboros-test-volume-")
os.environ["VOLUME_MOUNT_PATH"] = _TEST_VOLUME_ROOT

from backend.app.pipeline import pipeline as app_pipeline  # noqa: E402
from backend.app.util import config as app_config  # noqa: E402


class _FakeImagePredictor:
    def __init__(self):
        self.images = []

    def set_image(self, image):
        self.images.append(image.shape)

    def predict(self, point_coords, point_labels, multimask_output):
        mask = np.array([[[1, 0], [0, 1]]], dtype=np.uint8)
        return mask, None, None


class _FakeSam3ImageProcessor:
    def __init__(self):
        self.images = []
        self.boxes = []
        self.labels = []

    def set_image(self, image):
        shape = getattr(image, "shape", None)
        if shape is None and hasattr(image, "size"):
            shape = (image.size[1], image.size[0])
        self.images.append(shape)
        return {"image_shape": shape}

    def add_geometric_prompt(self, box, label, state):
        self.boxes.append(box)
        self.labels.append(label)
        state["masks"] = np.array([[[[True, False], [False, True]]]], dtype=bool)
        return state


class _FakeMaskTensor:
    def __init__(self, arr):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeMaskLogit:
    def __init__(self, arr):
        self._arr = arr

    def __gt__(self, value):
        return _FakeMaskTensor((self._arr > value).astype(np.uint8))


class _FakeVideoPredictorAsyncError:
    def init_state(self, *args, **kwargs):
        return {"images": type("Images", (), {"exception": RuntimeError("boom")})()}

    def add_new_points_or_box(self, **kwargs):
        return None

    def propagate_in_video(self, _inference_state):
        yield 0, [1], [np.array([[1]], dtype=np.float32)]


class _FakeVideoPredictorSuccess:
    def __init__(self):
        self.added = []

    def init_state(self, *args, **kwargs):
        return {}

    def add_new_points_or_box(self, **kwargs):
        self.added.append(kwargs)

    def propagate_in_video(self, _inference_state):
        yield 0, [1], [_FakeMaskLogit(np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32))]


class _FakeVideoPredictorInitError:
    def init_state(self, *args, **kwargs):
        raise RuntimeError("init failed")


class _FakeVideoPredictorAddPointsError:
    def init_state(self, *args, **kwargs):
        return {}

    def add_new_points_or_box(self, **kwargs):
        raise ValueError("add failed")

    def propagate_in_video(self, _inference_state):
        return iter(())


class _RecordingPool:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def starmap(self, func, args):
        self.calls.append((func, args))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.jobs_original = copy.deepcopy(app_config.jobs)

    def tearDown(self):
        app_config.jobs = self.jobs_original

    def _seed_job(self, job_id="job-1"):
        app_config.jobs[job_id] = {
            "status": "running",
            "steps": [
                {"name": "Transferring", "progress": 0},
                {"name": "Inference", "progress": 0},
                {"name": "Saving", "progress": 0},
            ],
            "updated_at": 0,
        }

    def test_run_image_predictor_writes_result_stack(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            volume_folder = Path(tmpdir)
            tf.imwrite(volume_folder / "000.tif", np.zeros((2, 2), dtype=np.uint8))
            tf.imwrite(volume_folder / "001.tif", np.zeros((2, 2), dtype=np.uint8))

            self._seed_job("job-1")

            result_stack = np.zeros((2, 2, 2), dtype=np.uint8)
            annotation_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
            input_label = np.array([1], dtype=np.int32)

            asyncio.run(
                app_pipeline.run_image_predictor(
                    predictor=_FakeImagePredictor(),
                    volume_folder=volume_folder,
                    annotation_points=annotation_points,
                    input_label=input_label,
                    input_shape=(2, 2, 2),
                    result_stack=result_stack,
                    job_id="job-1",
                )
            )

            expected = np.array([[255, 0], [0, 255]], dtype=np.uint8)
            np.testing.assert_array_equal(result_stack[0], expected)
            np.testing.assert_array_equal(result_stack[1], expected)
            self.assertEqual(app_config.jobs["job-1"]["steps"][1]["progress"], 100)

    def test_run_image_predictor_supports_sam3_processor_interface(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            volume_folder = Path(tmpdir)
            tf.imwrite(volume_folder / "000.tif", np.zeros((2, 2), dtype=np.uint8))
            tf.imwrite(volume_folder / "001.tif", np.zeros((2, 2), dtype=np.uint8))

            self._seed_job("job-1")

            result_stack = np.zeros((2, 2, 2), dtype=np.uint8)
            annotation_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
            input_label = np.array([1], dtype=np.int32)
            predictor = _FakeSam3ImageProcessor()

            asyncio.run(
                app_pipeline.run_image_predictor(
                    predictor=predictor,
                    volume_folder=volume_folder,
                    annotation_points=annotation_points,
                    input_label=input_label,
                    input_shape=(2, 2, 2),
                    result_stack=result_stack,
                    job_id="job-1",
                )
            )

            expected = np.array([[255, 0], [0, 255]], dtype=np.uint8)
            np.testing.assert_array_equal(result_stack[0], expected)
            np.testing.assert_array_equal(result_stack[1], expected)
            self.assertEqual(app_config.jobs["job-1"]["steps"][1]["progress"], 100)
            self.assertEqual(len(predictor.boxes), 2)
            self.assertEqual(predictor.labels, [True, True])

    def test_run_video_predictor_success_updates_result_stack(self):
        predictor = _FakeVideoPredictorSuccess()
        result_stack = np.zeros((1, 2, 2), dtype=np.uint8)

        with patch("backend.app.pipeline.pipeline.get_system_memory_info", return_value={"cpu": 1}), patch(
            "backend.app.pipeline.pipeline.get_shared_memory_info",
            return_value={"shm": 1},
        ), patch("backend.app.pipeline.pipeline.update_step"), patch(
            "backend.app.pipeline.pipeline.asyncio.sleep",
            new=AsyncMock(),
        ):
            asyncio.run(
                app_pipeline.run_video_predictor(
                    predictor=predictor,
                    volume_folder=Path("/tmp"),
                    input_shape=(1, 2, 2),
                    annotation_points=np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
                    input_label=np.array([1], dtype=np.int32),
                    result_stack=result_stack,
                    job_id="job-1",
                )
            )

        expected = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(result_stack[0], expected)
        self.assertEqual(len(predictor.added), 1)

    def test_run_video_predictor_raises_on_async_loader_exception(self):
        with patch("backend.app.pipeline.pipeline.get_system_memory_info", return_value={"cpu": 1}), patch(
            "backend.app.pipeline.pipeline.get_shared_memory_info",
            return_value={"shm": 1},
        ), patch("backend.app.pipeline.pipeline.update_step"), patch(
            "backend.app.pipeline.pipeline.asyncio.sleep",
            new=AsyncMock(),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                asyncio.run(
                    app_pipeline.run_video_predictor(
                        predictor=_FakeVideoPredictorAsyncError(),
                        volume_folder=Path("/tmp"),
                        input_shape=(1, 2, 2),
                        annotation_points=np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
                        input_label=np.array([1], dtype=np.int32),
                        result_stack=np.zeros((1, 2, 2), dtype=np.uint8),
                        job_id="job-1",
                    )
                )
        self.assertIn("Background frame loader thread crashed", str(ctx.exception))

    def test_run_video_predictor_raises_on_init_state_error(self):
        with patch("backend.app.pipeline.pipeline.get_system_memory_info", return_value={"cpu": 1}), patch(
            "backend.app.pipeline.pipeline.update_step"
        ), patch("backend.app.pipeline.pipeline.asyncio.sleep", new=AsyncMock()):
            with self.assertRaises(RuntimeError) as ctx:
                asyncio.run(
                    app_pipeline.run_video_predictor(
                        predictor=_FakeVideoPredictorInitError(),
                        volume_folder=Path("/tmp"),
                        input_shape=(1, 2, 2),
                        annotation_points=np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
                        input_label=np.array([1], dtype=np.int32),
                        result_stack=np.zeros((1, 2, 2), dtype=np.uint8),
                        job_id="job-1",
                    )
                )
        self.assertIn("init failed", str(ctx.exception))

    def test_run_video_predictor_raises_when_no_annotation_frames_available(self):
        with patch("backend.app.pipeline.pipeline.get_system_memory_info", return_value={"cpu": 1}), patch(
            "backend.app.pipeline.pipeline.update_step"
        ), patch("backend.app.pipeline.pipeline.asyncio.sleep", new=AsyncMock()):
            with self.assertRaises(RuntimeError) as ctx:
                asyncio.run(
                    app_pipeline.run_video_predictor(
                        predictor=_FakeVideoPredictorSuccess(),
                        volume_folder=Path("/tmp"),
                        input_shape=(0, 2, 2),
                        annotation_points=np.empty((0, 3), dtype=np.float32),
                        input_label=np.array([1], dtype=np.int32),
                        result_stack=np.zeros((0, 2, 2), dtype=np.uint8),
                        job_id="job-1",
                    )
                )
        self.assertIn("No annotation frames generated", str(ctx.exception))

    def test_run_video_predictor_raises_on_add_points_error(self):
        with patch("backend.app.pipeline.pipeline.get_system_memory_info", return_value={"cpu": 1}), patch(
            "backend.app.pipeline.pipeline.update_step"
        ), patch("backend.app.pipeline.pipeline.asyncio.sleep", new=AsyncMock()):
            with self.assertRaises(ValueError) as ctx:
                asyncio.run(
                    app_pipeline.run_video_predictor(
                        predictor=_FakeVideoPredictorAddPointsError(),
                        volume_folder=Path("/tmp"),
                        input_shape=(1, 2, 2),
                        annotation_points=np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
                        input_label=np.array([1], dtype=np.int32),
                        result_stack=np.zeros((1, 2, 2), dtype=np.uint8),
                        job_id="job-1",
                    )
                )
        self.assertIn("add failed", str(ctx.exception))

    def test_run_pipeline_image_success_marks_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.app.pipeline.pipeline.config.INTERNAL_VOLUME_PATH", tmpdir):
                plugin_root = Path(tmpdir, app_config.PLUGIN_NAME)
                plugin_root.mkdir(parents=True, exist_ok=True)
                source = plugin_root / "input.tif"
                tf.imwrite(source, np.zeros((2, 3, 4), dtype=np.uint8))

                self._seed_job("job-1")
                pool = _RecordingPool()

                with patch("backend.app.pipeline.pipeline.Pool", return_value=pool), patch(
                    "backend.app.pipeline.pipeline.copy_to_volume",
                    new=AsyncMock(return_value=(True, "")),
                ), patch(
                    "backend.app.pipeline.pipeline.copy_to_host",
                    new=AsyncMock(return_value=(True, "")),
                ), patch(
                    "backend.app.pipeline.pipeline.get_predictor",
                    return_value=object(),
                ), patch(
                    "backend.app.pipeline.pipeline.load_annotation_points",
                    return_value=np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
                ), patch(
                    "backend.app.pipeline.pipeline.run_image_predictor",
                    new=AsyncMock(),
                ) as mock_run_image, patch(
                    "backend.app.pipeline.pipeline.asyncio.sleep",
                    new=AsyncMock(),
                ), patch("backend.app.pipeline.pipeline.shutil.rmtree") as mock_rmtree, patch(
                    "backend.app.pipeline.pipeline.time.time",
                    return_value=123.0,
                ):
                    asyncio.run(
                        app_pipeline.run_pipeline(
                            "job-1",
                            "/host/input.tif",
                            "/host/output.tif",
                            "sam2_hiera_tiny",
                            "ImagePredictor",
                        )
                    )

                self.assertEqual(app_config.jobs["job-1"]["status"], "completed")
                self.assertEqual(app_config.jobs["job-1"]["updated_at"], 123.0)
                self.assertEqual(pool.calls[0][0], app_pipeline.downsample)
                mock_run_image.assert_awaited_once()
                mock_rmtree.assert_called()

    def test_run_pipeline_video_success_uses_video_branches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.app.pipeline.pipeline.config.INTERNAL_VOLUME_PATH", tmpdir):
                plugin_root = Path(tmpdir, app_config.PLUGIN_NAME)
                source_dir = plugin_root / "input"
                source_dir.mkdir(parents=True, exist_ok=True)
                tf.imwrite(source_dir / "000.tif", np.zeros((3, 4), dtype=np.uint8))
                tf.imwrite(source_dir / "001.tif", np.zeros((3, 4), dtype=np.uint8))

                self._seed_job("job-1")
                pool = _RecordingPool()

                with patch("backend.app.pipeline.pipeline.Pool", return_value=pool), patch(
                    "backend.app.pipeline.pipeline.copy_to_volume",
                    new=AsyncMock(return_value=(True, "")),
                ), patch(
                    "backend.app.pipeline.pipeline.copy_to_host",
                    new=AsyncMock(return_value=(True, "")),
                ), patch(
                    "backend.app.pipeline.pipeline.get_predictor",
                    return_value=object(),
                ), patch(
                    "backend.app.pipeline.pipeline.load_annotation_points",
                    return_value=np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
                ), patch(
                    "backend.app.pipeline.pipeline.run_video_predictor",
                    new=AsyncMock(),
                ) as mock_run_video, patch(
                    "backend.app.pipeline.pipeline.asyncio.sleep",
                    new=AsyncMock(),
                ), patch("backend.app.pipeline.pipeline.shutil.rmtree"), patch(
                    "backend.app.pipeline.pipeline.time.time",
                    return_value=456.0,
                ):
                    asyncio.run(
                        app_pipeline.run_pipeline(
                            "job-1",
                            "/host/input",
                            "/host/output.tif",
                            "sam2_hiera_tiny",
                            "VideoPredictor",
                        )
                    )

                self.assertEqual(app_config.jobs["job-1"]["status"], "completed")
                self.assertEqual(app_config.jobs["job-1"]["updated_at"], 456.0)
                self.assertEqual(pool.calls[0][0], app_pipeline.jpeg_convert)
                mock_run_video.assert_awaited_once()

    def test_run_pipeline_uses_default_annotations_when_metadata_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.app.pipeline.pipeline.config.INTERNAL_VOLUME_PATH", tmpdir):
                plugin_root = Path(tmpdir, app_config.PLUGIN_NAME)
                plugin_root.mkdir(parents=True, exist_ok=True)
                source = plugin_root / "input.tif"
                tf.imwrite(source, np.zeros((2, 3, 4), dtype=np.uint8))

                self._seed_job("job-1")
                pool = _RecordingPool()

                with patch("backend.app.pipeline.pipeline.Pool", return_value=pool), patch(
                    "backend.app.pipeline.pipeline.copy_to_volume",
                    new=AsyncMock(return_value=(True, "")),
                ), patch(
                    "backend.app.pipeline.pipeline.copy_to_host",
                    new=AsyncMock(return_value=(True, "")),
                ), patch(
                    "backend.app.pipeline.pipeline.get_predictor",
                    return_value=object(),
                ), patch(
                    "backend.app.pipeline.pipeline.load_annotation_points",
                    return_value=None,
                ), patch(
                    "backend.app.pipeline.pipeline.default_annotations",
                    return_value=np.array([[2.0, 1.0, 0.0]], dtype=np.float32),
                ) as mock_default, patch(
                    "backend.app.pipeline.pipeline.run_image_predictor",
                    new=AsyncMock(),
                ), patch(
                    "backend.app.pipeline.pipeline.asyncio.sleep",
                    new=AsyncMock(),
                ), patch("backend.app.pipeline.pipeline.shutil.rmtree"), patch(
                    "backend.app.pipeline.pipeline.time.time",
                    return_value=457.0,
                ):
                    asyncio.run(
                        app_pipeline.run_pipeline(
                            "job-1",
                            "/host/input.tif",
                            "/host/output.tif",
                            "sam2_hiera_tiny",
                            "ImagePredictor",
                        )
                    )

                self.assertEqual(app_config.jobs["job-1"]["status"], "completed")
                mock_default.assert_called_once()

    def test_run_pipeline_marks_error_when_copy_to_volume_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.app.pipeline.pipeline.config.INTERNAL_VOLUME_PATH", tmpdir):
                self._seed_job("job-1")

                with patch(
                    "backend.app.pipeline.pipeline.copy_to_volume",
                    new=AsyncMock(return_value=(False, "cannot copy")),
                ), patch("backend.app.pipeline.pipeline.asyncio.sleep", new=AsyncMock()), patch(
                    "backend.app.pipeline.pipeline.time.time",
                    return_value=200.0,
                ):
                    asyncio.run(
                        app_pipeline.run_pipeline(
                            "job-1",
                            "/host/input.tif",
                            "/host/output.tif",
                            "sam2_hiera_tiny",
                            "ImagePredictor",
                        )
                    )

                self.assertEqual(app_config.jobs["job-1"]["status"], "error")
                self.assertEqual(app_config.jobs["job-1"]["updated_at"], 200.0)

    def test_run_pipeline_marks_error_when_source_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.app.pipeline.pipeline.config.INTERNAL_VOLUME_PATH", tmpdir):
                self._seed_job("job-1")

                with patch(
                    "backend.app.pipeline.pipeline.copy_to_volume",
                    new=AsyncMock(return_value=(True, "")),
                ), patch("backend.app.pipeline.pipeline.asyncio.sleep", new=AsyncMock()), patch(
                    "backend.app.pipeline.pipeline.time.time",
                    return_value=201.0,
                ):
                    asyncio.run(
                        app_pipeline.run_pipeline(
                            "job-1",
                            "/host/input.tif",
                            "/host/output.tif",
                            "sam2_hiera_tiny",
                            "ImagePredictor",
                        )
                    )

                self.assertEqual(app_config.jobs["job-1"]["status"], "error")
                self.assertEqual(app_config.jobs["job-1"]["updated_at"], 201.0)

    def test_run_pipeline_marks_error_when_predictor_init_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.app.pipeline.pipeline.config.INTERNAL_VOLUME_PATH", tmpdir):
                plugin_root = Path(tmpdir, app_config.PLUGIN_NAME)
                plugin_root.mkdir(parents=True, exist_ok=True)
                source = plugin_root / "input.tif"
                tf.imwrite(source, np.zeros((2, 3, 4), dtype=np.uint8))

                self._seed_job("job-1")
                pool = _RecordingPool()

                with patch("backend.app.pipeline.pipeline.Pool", return_value=pool), patch(
                    "backend.app.pipeline.pipeline.copy_to_volume",
                    new=AsyncMock(return_value=(True, "")),
                ), patch(
                    "backend.app.pipeline.pipeline.get_predictor",
                    side_effect=RuntimeError("model init failed"),
                ), patch(
                    "backend.app.pipeline.pipeline.load_annotation_points",
                    return_value=np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
                ), patch(
                    "backend.app.pipeline.pipeline.asyncio.sleep",
                    new=AsyncMock(),
                ), patch(
                    "backend.app.pipeline.pipeline.time.time",
                    return_value=202.0,
                ):
                    asyncio.run(
                        app_pipeline.run_pipeline(
                            "job-1",
                            "/host/input.tif",
                            "/host/output.tif",
                            "sam2_hiera_tiny",
                            "ImagePredictor",
                        )
                    )

                self.assertEqual(app_config.jobs["job-1"]["status"], "error")
                self.assertEqual(app_config.jobs["job-1"]["updated_at"], 202.0)

    def test_run_pipeline_marks_error_when_predictor_execution_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.app.pipeline.pipeline.config.INTERNAL_VOLUME_PATH", tmpdir):
                plugin_root = Path(tmpdir, app_config.PLUGIN_NAME)
                plugin_root.mkdir(parents=True, exist_ok=True)
                source = plugin_root / "input.tif"
                tf.imwrite(source, np.zeros((2, 3, 4), dtype=np.uint8))

                self._seed_job("job-1")
                pool = _RecordingPool()

                with patch("backend.app.pipeline.pipeline.Pool", return_value=pool), patch(
                    "backend.app.pipeline.pipeline.copy_to_volume",
                    new=AsyncMock(return_value=(True, "")),
                ), patch(
                    "backend.app.pipeline.pipeline.get_predictor",
                    return_value=object(),
                ), patch(
                    "backend.app.pipeline.pipeline.load_annotation_points",
                    return_value=np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
                ), patch(
                    "backend.app.pipeline.pipeline.run_image_predictor",
                    new=AsyncMock(side_effect=RuntimeError("predict failed")),
                ), patch(
                    "backend.app.pipeline.pipeline.asyncio.sleep",
                    new=AsyncMock(),
                ), patch(
                    "backend.app.pipeline.pipeline.time.time",
                    return_value=202.5,
                ):
                    asyncio.run(
                        app_pipeline.run_pipeline(
                            "job-1",
                            "/host/input.tif",
                            "/host/output.tif",
                            "sam2_hiera_tiny",
                            "ImagePredictor",
                        )
                    )

                self.assertEqual(app_config.jobs["job-1"]["status"], "error")
                self.assertEqual(app_config.jobs["job-1"]["updated_at"], 202.5)

    def test_run_pipeline_marks_error_when_copy_to_host_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.app.pipeline.pipeline.config.INTERNAL_VOLUME_PATH", tmpdir):
                plugin_root = Path(tmpdir, app_config.PLUGIN_NAME)
                plugin_root.mkdir(parents=True, exist_ok=True)
                source = plugin_root / "input.tif"
                tf.imwrite(source, np.zeros((2, 3, 4), dtype=np.uint8))

                self._seed_job("job-1")
                pool = _RecordingPool()

                with patch("backend.app.pipeline.pipeline.Pool", return_value=pool), patch(
                    "backend.app.pipeline.pipeline.copy_to_volume",
                    new=AsyncMock(return_value=(True, "")),
                ), patch(
                    "backend.app.pipeline.pipeline.copy_to_host",
                    new=AsyncMock(return_value=(False, "cannot copy back")),
                ), patch(
                    "backend.app.pipeline.pipeline.get_predictor",
                    return_value=object(),
                ), patch(
                    "backend.app.pipeline.pipeline.load_annotation_points",
                    return_value=np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
                ), patch(
                    "backend.app.pipeline.pipeline.run_image_predictor",
                    new=AsyncMock(),
                ), patch(
                    "backend.app.pipeline.pipeline.asyncio.sleep",
                    new=AsyncMock(),
                ), patch(
                    "backend.app.pipeline.pipeline.time.time",
                    return_value=203.0,
                ):
                    asyncio.run(
                        app_pipeline.run_pipeline(
                            "job-1",
                            "/host/input.tif",
                            "/host/output.tif",
                            "sam2_hiera_tiny",
                            "ImagePredictor",
                        )
                    )

                self.assertEqual(app_config.jobs["job-1"]["status"], "error")
                self.assertEqual(app_config.jobs["job-1"]["updated_at"], 203.0)


if __name__ == "__main__":
    unittest.main()
