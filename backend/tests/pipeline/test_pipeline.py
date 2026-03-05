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


class _FakeVideoPredictor:
    def init_state(self, *args, **kwargs):
        return {"images": type("Images", (), {"exception": RuntimeError("boom")})()}

    def add_new_points_or_box(self, **kwargs):
        return None

    def propagate_in_video(self, _inference_state):
        yield 0, [1], [np.array([[1]], dtype=np.float32)]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.jobs_original = copy.deepcopy(app_config.jobs)

    def tearDown(self):
        app_config.jobs = self.jobs_original

    def test_run_image_predictor_writes_result_stack(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            volume_folder = Path(tmpdir)
            tf.imwrite(volume_folder / "000.tif", np.zeros((2, 2), dtype=np.uint8))
            tf.imwrite(volume_folder / "001.tif", np.zeros((2, 2), dtype=np.uint8))

            app_config.jobs = {
                "job-1": {
                    "steps": [
                        {"progress": 0},
                        {"progress": 0},
                        {"progress": 0},
                    ],
                    "updated_at": 0,
                }
            }

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
                        predictor=_FakeVideoPredictor(),
                        volume_folder=Path("/tmp"),
                        input_shape=(1, 2, 2),
                        annotation_points=np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
                        input_label=np.array([1], dtype=np.int32),
                        result_stack=np.zeros((1, 2, 2), dtype=np.uint8),
                        job_id="job-1",
                    )
                )
        self.assertIn("Background frame loader thread crashed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
