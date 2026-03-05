import json
import os
import tempfile
import types
import unittest
from pathlib import Path, PurePosixPath, PureWindowsPath
from unittest.mock import patch

import numpy as np
from PIL import Image
import tifffile as tf

_TEST_VOLUME_ROOT = tempfile.mkdtemp(prefix="ouroboros-test-volume-")
os.environ["VOLUME_MOUNT_PATH"] = _TEST_VOLUME_ROOT

from backend.app.util import config as app_config  # noqa: E402
from backend.app.util import util as app_util  # noqa: E402


class UtilTests(unittest.TestCase):
    def setUp(self):
        self.jobs_original = app_config.jobs.copy()

    def tearDown(self):
        app_config.jobs = self.jobs_original

    def test_num_digits_for_n_files(self):
        self.assertEqual(app_util.num_digits_for_n_files(1), 1)
        self.assertEqual(app_util.num_digits_for_n_files(10), 1)
        self.assertEqual(app_util.num_digits_for_n_files(100), 2)

    def test_sorted_tif_paths_filters_tiff_extensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.tif").write_bytes(b"")
            (root / "b.tiff").write_bytes(b"")
            (root / "c.txt").write_bytes(b"")

            out = app_util._sorted_tif_paths(root)
            self.assertEqual([p.name for p in out], ["a.tif", "b.tiff"])

    def test_get_pure_path_detects_windows_and_posix(self):
        self.assertIsInstance(app_util.get_pure_path(r"C:\data\file.tif"), PureWindowsPath)
        self.assertIsInstance(app_util.get_pure_path("/tmp/file.tif"), PurePosixPath)

    def test_default_annotations_returns_center_points_at_interval(self):
        with patch("backend.app.util.util.config.FALLBACK_ANNOTATION_INTERVAL", 2):
            points = app_util.default_annotations((5, 8, 10))

        expected = np.array(
            [
                [5, 4, 0],
                [5, 4, 2],
                [5, 4, 4],
                [5, 4, 4],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(points, expected)

    def test_default_annotations_empty_when_no_frames(self):
        with patch("backend.app.util.util.config.FALLBACK_ANNOTATION_INTERVAL", 3):
            points = app_util.default_annotations((0, 8, 10))
        print(points.shape)
        print(points)
        self.assertEqual(points.shape, (0, 3))

    def test_load_annotation_points_from_single_tiff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tif_path = Path(tmpdir, "volume.tif")
            metadata = {
                "annotation_points": [
                    [10.0, 20.0, 0.0],
                    [11.5, 22.5, 5.0],
                ]
            }
            tf.imwrite(tif_path, np.zeros((8, 8), dtype=np.uint8), description=json.dumps(metadata))

            points = app_util.load_annotation_points(tif_path)
            self.assertIsNotNone(points)
            self.assertEqual(points.shape, (2, 3))
            np.testing.assert_allclose(points, np.array(metadata["annotation_points"], dtype=np.float32))

    def test_load_annotation_points_from_directory_first_tiff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            metadata = {"annotation_points": [[1, 2, 3], [4, 5, 6]]}
            tf.imwrite(folder / "000.tif", np.zeros((4, 4), dtype=np.uint8), description=json.dumps(metadata))
            tf.imwrite(folder / "001.tif", np.zeros((4, 4), dtype=np.uint8), description=json.dumps({}))

            points = app_util.load_annotation_points(folder)
            self.assertIsNotNone(points)
            np.testing.assert_allclose(points, np.array(metadata["annotation_points"], dtype=np.float32))

    def test_load_annotation_points_missing_metadata_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tif_path = Path(tmpdir, "volume.tif")
            tf.imwrite(tif_path, np.zeros((8, 8), dtype=np.uint8), description=json.dumps({}))
            points = app_util.load_annotation_points(tif_path)
            self.assertIsNone(points)

    def test_load_annotation_points_invalid_json_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tif_path = Path(tmpdir, "volume.tif")
            tf.imwrite(tif_path, np.zeros((8, 8), dtype=np.uint8), description="not-json")
            points = app_util.load_annotation_points(tif_path)
            self.assertIsNone(points)

    def test_load_annotation_points_invalid_shape_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tif_path = Path(tmpdir, "volume.tif")
            metadata = {"annotation_points": [1, 2, 3]}
            tf.imwrite(tif_path, np.zeros((8, 8), dtype=np.uint8), description=json.dumps(metadata))
            points = app_util.load_annotation_points(tif_path)
            self.assertIsNone(points)

    def test_load_annotation_points_empty_directory_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            points = app_util.load_annotation_points(Path(tmpdir))
            self.assertIsNone(points)

    def test_annotation_samples_for_video_rounds_and_groups_z(self):
        annotation_points = np.array(
            [
                [5.0, 6.0, 0.49],
                [7.0, 8.0, 0.51],
                [9.0, 10.0, 1.4],
                [3.0, 4.0, 99.0],
            ],
            dtype=np.float32,
        )

        samples = app_util._annotation_samples_for_video(annotation_points, num_frames=5)

        self.assertEqual([frame for frame, _ in samples], [0, 1])
        np.testing.assert_allclose(samples[0][1], np.array([[5.0, 6.0]], dtype=np.float32))
        np.testing.assert_allclose(samples[1][1], np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32))

    def test_annotation_point_for_frame_interpolates_between_neighbors(self):
        annotation_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 20.0, 10.0],
            ],
            dtype=np.float32,
        )

        point = app_util._annotation_point_for_frame(annotation_points, frame_idx=5)
        np.testing.assert_allclose(point, np.array([[5.0, 10.0]], dtype=np.float32))

    def test_annotation_point_for_frame_clamps_outside_range(self):
        annotation_points = np.array(
            [
                [2.0, 4.0, 3.0],
                [10.0, 12.0, 8.0],
            ],
            dtype=np.float32,
        )

        before = app_util._annotation_point_for_frame(annotation_points, frame_idx=0)
        after = app_util._annotation_point_for_frame(annotation_points, frame_idx=50)

        np.testing.assert_allclose(before, np.array([[2.0, 4.0]], dtype=np.float32))
        np.testing.assert_allclose(after, np.array([[10.0, 12.0]], dtype=np.float32))

    def test_annotation_point_for_frame_with_single_point_returns_that_point(self):
        annotation_points = np.array([[3.0, 7.0, 5.0]], dtype=np.float32)
        point = app_util._annotation_point_for_frame(annotation_points, frame_idx=2)
        np.testing.assert_allclose(point, np.array([[3.0, 7.0]], dtype=np.float32))

    def test_check_async_frame_loader_exception(self):
        class Loader:
            exception = RuntimeError("boom")

        self.assertIsNone(app_util.check_async_frame_loader_exception({}))
        self.assertIsNone(app_util.check_async_frame_loader_exception({"images": object()}))
        err = app_util.check_async_frame_loader_exception({"images": Loader()})
        self.assertIsInstance(err, RuntimeError)

    def test_update_step_updates_job(self):
        app_config.jobs = {"job-1": {"steps": [{"progress": 0}], "updated_at": 0}}
        with patch("backend.app.util.util.time.time", return_value=77.0):
            app_util.update_step("job-1", 0, 42)
        self.assertEqual(app_config.jobs["job-1"]["steps"][0]["progress"], 42)
        self.assertEqual(app_config.jobs["job-1"]["updated_at"], 77.0)

    def test_update_step_missing_job_is_noop(self):
        app_config.jobs = {}
        app_util.update_step("missing", 0, 10)
        self.assertEqual(app_config.jobs, {})

    def test_get_shared_memory_info_and_file_not_found(self):
        with patch("backend.app.util.util.shutil.disk_usage", return_value=(1000, 400, 600)):
            info = app_util.get_shared_memory_info()
            self.assertAlmostEqual(info["shm_percent"], 40.0)
        with patch("backend.app.util.util.shutil.disk_usage", side_effect=FileNotFoundError):
            info = app_util.get_shared_memory_info()
            self.assertIn("error", info)

    def test_get_system_memory_info_cpu_only(self):
        cpu_mem = types.SimpleNamespace(used=10, available=20, total=30, percent=33.3)
        with patch("backend.app.util.util.psutil.virtual_memory", return_value=cpu_mem), patch(
            "backend.app.util.util.torch.cuda.is_available",
            return_value=False,
        ):
            info = app_util.get_system_memory_info()
        self.assertIsNone(info["gpu"])
        self.assertEqual(info["cpu_percent"], 33.3)

    def test_get_system_memory_info_with_gpu(self):
        cpu_mem = types.SimpleNamespace(used=10, available=20, total=30, percent=33.3)
        gpu_props = types.SimpleNamespace(total_memory=8e9)
        with patch("backend.app.util.util.psutil.virtual_memory", return_value=cpu_mem), patch(
            "backend.app.util.util.torch.cuda.is_available",
            return_value=True,
        ), patch("backend.app.util.util.torch.cuda.memory_allocated", return_value=2e9), patch(
            "backend.app.util.util.torch.cuda.memory_reserved",
            return_value=3e9,
        ), patch("backend.app.util.util.torch.cuda.get_device_properties", return_value=gpu_props):
            info = app_util.get_system_memory_info()
        self.assertIsNotNone(info["gpu"])
        self.assertEqual(info["gpu"]["total_gb"], 8.0)

    def test_jpeg_convert_writes_rgb_jpg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir, "input.tif")
            out_path = Path(tmpdir, "out")
            tf.imwrite(img_path, np.arange(16, dtype=np.uint16).reshape(4, 4))

            app_util.jpeg_convert(img_path, out_path)
            jpg_path = out_path.with_suffix(".jpg")
            self.assertTrue(jpg_path.exists())
            with Image.open(jpg_path) as image:
                self.assertEqual(image.mode, "RGB")

    def test_downsample_grayscale_expands_to_three_channels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir, "input.tif")
            out_path = Path(tmpdir, "out")
            tf.imwrite(img_path, np.arange(16, dtype=np.uint16).reshape(4, 4))

            app_util.downsample(img_path, out_path)
            written = tf.imread(out_path.with_suffix(".tif"))
            self.assertEqual(written.shape, (4, 4, 3))

    def test_downsample_trims_extra_channels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir, "input.tif")
            out_path = Path(tmpdir, "out")
            data = np.random.randint(0, 255, size=(4, 4, 5), dtype=np.uint8)
            tf.imwrite(img_path, data)

            app_util.downsample(img_path, out_path)
            written = tf.imread(out_path.with_suffix(".tif"))
            self.assertEqual(written.shape, (4, 4, 3))


if __name__ == "__main__":
    unittest.main()
