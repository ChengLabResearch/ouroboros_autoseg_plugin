import asyncio
import copy
import os
import tempfile
import types
import unittest
from pathlib import Path, PurePosixPath, PureWindowsPath
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import BackgroundTasks, HTTPException
import numpy as np
from PIL import Image
import tifffile as tf


_TEST_VOLUME_ROOT = tempfile.mkdtemp(prefix="ouroboros-test-volume-")
os.environ["VOLUME_MOUNT_PATH"] = _TEST_VOLUME_ROOT

from backend.app import main


class PublicFunctionTests(unittest.TestCase):
    def setUp(self):
        self.startup_status_original = copy.deepcopy(main.startup_status)
        self.jobs_original = copy.deepcopy(main.jobs)
        self.loaded_model_name_original = main.loaded_model_name
        self.loaded_predictor_original = main.loaded_predictor

    def tearDown(self):
        main.startup_status = self.startup_status_original
        main.jobs = self.jobs_original
        main.loaded_model_name = self.loaded_model_name_original
        main.loaded_predictor = self.loaded_predictor_original

    def test_num_digits_for_n_files(self):
        self.assertEqual(main.num_digits_for_n_files(1), 1)
        self.assertEqual(main.num_digits_for_n_files(10), 1)
        self.assertEqual(main.num_digits_for_n_files(100), 2)

    def test_sorted_tif_paths_filters_tiff_extensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.tif").write_bytes(b"")
            (root / "b.tiff").write_bytes(b"")
            (root / "c.txt").write_bytes(b"")

            out = main._sorted_tif_paths(root)
            self.assertEqual([p.name for p in out], ["a.tif", "b.tiff"])

    def test_get_pure_path_detects_windows_and_posix(self):
        self.assertIsInstance(main.get_pure_path(r"C:\data\file.tif"), PureWindowsPath)
        self.assertIsInstance(main.get_pure_path("/tmp/file.tif"), PurePosixPath)

    def test_server_active_returns_json_response(self):
        response = asyncio.run(main.server_active())
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Segmentation server is active", response.body)

    def test_get_startup_status_delegates_to_refresh(self):
        with patch("backend.app.main.refresh_startup_status", return_value={"ok": True}):
            result = asyncio.run(main.get_startup_status())
        self.assertEqual(result, {"ok": True})

    def test_get_model_status_reports_files(self):
        def fake_isfile(path):
            return str(path).endswith("sam3.pt")

        with patch("backend.app.main.os.path.isfile", side_effect=fake_isfile):
            status = asyncio.run(main.get_model_status())
        self.assertIn("checkpoint_dir", status)
        self.assertEqual(status["models"]["sam2_hiera_base_plus"], False)
        self.assertEqual(status["models"]["sam3"], True)

    def test_check_async_frame_loader_exception(self):
        class Loader:
            exception = RuntimeError("boom")

        self.assertIsNone(main.check_async_frame_loader_exception({}))
        self.assertIsNone(main.check_async_frame_loader_exception({"images": object()}))
        err = main.check_async_frame_loader_exception({"images": Loader()})
        self.assertIsInstance(err, RuntimeError)

    def test_update_initialization_step_sets_status_and_start_time(self):
        main.startup_status = {
            "is_ready": False,
            "initialization_steps": [{"name": "Step", "status": "pending"}],
            "start_time": None,
            "ready_time": None,
        }
        with patch("backend.app.main.time.time", return_value=123.0):
            main.update_initialization_step("Step", "completed")
        self.assertEqual(main.startup_status["initialization_steps"][0]["status"], "completed")
        self.assertEqual(main.startup_status["start_time"], 123.0)

    def test_mark_initialization_complete_marks_pending_steps(self):
        main.startup_status = {
            "is_ready": False,
            "initialization_steps": [
                {"name": "A", "status": "pending"},
                {"name": "B", "status": "warning"},
            ],
            "start_time": None,
            "ready_time": None,
        }
        with patch("backend.app.main.time.time", return_value=555.0):
            main.mark_initialization_complete()
        self.assertTrue(main.startup_status["is_ready"])
        self.assertEqual(main.startup_status["ready_time"], 555.0)
        self.assertEqual(main.startup_status["initialization_steps"][0]["status"], "completed")
        self.assertEqual(main.startup_status["initialization_steps"][1]["status"], "warning")

    def test_update_step_updates_job(self):
        main.jobs = {"job-1": {"steps": [{"progress": 0}], "updated_at": 0}}
        with patch("backend.app.main.time.time", return_value=77.0):
            main.update_step("job-1", 0, 42)
        self.assertEqual(main.jobs["job-1"]["steps"][0]["progress"], 42)
        self.assertEqual(main.jobs["job-1"]["updated_at"], 77.0)

    def test_get_shared_memory_info_and_file_not_found(self):
        with patch("backend.app.main.shutil.disk_usage", return_value=(1000, 400, 600)):
            info = main.get_shared_memory_info()
            self.assertAlmostEqual(info["shm_percent"], 40.0)
        with patch("backend.app.main.shutil.disk_usage", side_effect=FileNotFoundError):
            info = main.get_shared_memory_info()
            self.assertIn("error", info)

    def test_get_system_memory_info_cpu_only(self):
        cpu_mem = types.SimpleNamespace(used=10, available=20, total=30, percent=33.3)
        with patch("backend.app.main.psutil.virtual_memory", return_value=cpu_mem), patch(
            "backend.app.main.torch.cuda.is_available",
            return_value=False,
        ):
            info = main.get_system_memory_info()
        self.assertIsNone(info["gpu"])
        self.assertEqual(info["cpu_percent"], 33.3)

    def test_get_system_memory_info_with_gpu(self):
        cpu_mem = types.SimpleNamespace(used=10, available=20, total=30, percent=33.3)
        gpu_props = types.SimpleNamespace(total_memory=8e9)
        with patch("backend.app.main.psutil.virtual_memory", return_value=cpu_mem), patch(
            "backend.app.main.torch.cuda.is_available",
            return_value=True,
        ), patch("backend.app.main.torch.cuda.memory_allocated", return_value=2e9), patch(
            "backend.app.main.torch.cuda.memory_reserved",
            return_value=3e9,
        ), patch("backend.app.main.torch.cuda.get_device_properties", return_value=gpu_props):
            info = main.get_system_memory_info()
        self.assertIsNotNone(info["gpu"])
        self.assertEqual(info["gpu"]["total_gb"], 8.0)

    def test_refresh_startup_status_completed(self):
        main.startup_status = {
            "is_ready": False,
            "initialization_steps": [
                {"name": "Building Docker Image", "status": "pending"},
                {"name": "Connecting to Volume Server", "status": "pending"},
                {"name": "Initializing ML Models", "status": "pending"},
            ],
            "start_time": None,
            "ready_time": None,
        }
        response = types.SimpleNamespace(ok=True, status_code=200)
        with patch("backend.app.main.requests.get", return_value=response), patch(
            "backend.app.main.build_sam2",
            object(),
        ), patch("backend.app.main.SAM2ImagePredictor", object()), patch(
            "backend.app.main.time.time",
            return_value=321.0,
        ):
            status = main.refresh_startup_status()
        self.assertTrue(status["is_ready"])
        self.assertEqual(status["ready_time"], 321.0)

    def test_refresh_startup_status_warning_path(self):
        main.startup_status = {
            "is_ready": False,
            "initialization_steps": [
                {"name": "Building Docker Image", "status": "pending"},
                {"name": "Connecting to Volume Server", "status": "pending"},
                {"name": "Initializing ML Models", "status": "pending"},
            ],
            "start_time": None,
            "ready_time": None,
        }
        with patch("backend.app.main.requests.get", side_effect=main.requests.RequestException), patch(
            "backend.app.main.build_sam2",
            None,
        ), patch("backend.app.main.SAM2ImagePredictor", None), patch(
            "backend.app.main.build_sam3",
            None,
        ), patch("backend.app.main.Sam3Processor", None):
            status = main.refresh_startup_status()
        self.assertFalse(status["is_ready"])
        self.assertIsNone(status["ready_time"])

    def test_request_volume_server_success_and_failure(self):
        ok_response = types.SimpleNamespace(ok=True, text="ok")
        fail_response = types.SimpleNamespace(ok=False, text="bad")

        with patch("backend.app.main.requests.post", return_value=ok_response):
            success, msg = asyncio.run(main.request_volume_server("path", {"a": 1}))
        self.assertTrue(success)
        self.assertEqual(msg, "")

        with patch("backend.app.main.requests.post", return_value=fail_response):
            success, msg = asyncio.run(main.request_volume_server("path", {"a": 1}))
        self.assertFalse(success)
        self.assertEqual(msg, "bad")

    def test_request_volume_server_handles_exception(self):
        with patch("backend.app.main.requests.post", side_effect=RuntimeError("network down")):
            success, msg = asyncio.run(main.request_volume_server("path", {"a": 1}))
        self.assertFalse(success)
        self.assertIn("network down", msg)

    def test_download_model_exists_path(self):
        req = main.DownloadRequest(model_type="sam2_hiera_tiny")
        with patch("backend.app.main.os.makedirs"), patch(
            "backend.app.main.os.path.isfile",
            return_value=True,
        ), patch(
            "backend.app.main.os.path.isdir",
            return_value=False,
        ):
            result = asyncio.run(main.download_model(req))
        self.assertEqual(result["status"], "exists")

    def test_download_model_sam2_success(self):
        req = main.DownloadRequest(model_type="sam2_hiera_tiny")
        with patch("backend.app.main.os.makedirs"), patch(
            "backend.app.main.os.path.isfile",
            return_value=False,
        ), patch(
            "backend.app.main.os.path.isdir",
            return_value=False,
        ), patch("backend.app.main.download_file") as mocked_download:
            result = asyncio.run(main.download_model(req))
        self.assertEqual(result["status"], "success")
        mocked_download.assert_called_once()
        args, _kwargs = mocked_download.call_args
        self.assertEqual(args[0], main.SAM2_URLS["sam2_hiera_tiny"])
        self.assertTrue(str(args[1]).endswith("sam2_hiera_tiny.pt"))

    def test_download_model_unknown_type_raises_http_500(self):
        req = main.DownloadRequest(model_type="unknown-model")
        with patch("backend.app.main.os.makedirs"), patch(
            "backend.app.main.os.path.isfile",
            return_value=False,
        ), patch(
            "backend.app.main.os.path.isdir",
            return_value=False,
        ):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(main.download_model(req))
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("Unknown model type", str(ctx.exception.detail))

    def test_download_model_removes_existing_directory_target(self):
        req = main.DownloadRequest(model_type="sam2_hiera_tiny")
        target_path = os.path.join(main.CHECKPOINT_DIR, f"{req.model_type}.pt")
        with patch("backend.app.main.os.makedirs"), patch(
            "backend.app.main.os.path.isfile",
            return_value=False,
        ), patch(
            "backend.app.main.os.path.isdir",
            side_effect=lambda path: str(path) == str(target_path),
        ), patch("backend.app.main.shutil.rmtree") as mocked_rmtree, patch(
            "backend.app.main.download_file"
        ):
            result = asyncio.run(main.download_model(req))
        self.assertEqual(result["status"], "success")
        mocked_rmtree.assert_called_once()

    def test_copy_helpers_delegate_to_request_volume_server(self):
        with patch("backend.app.main.request_volume_server", new=AsyncMock(return_value=(True, ""))) as mocked:
            result = asyncio.run(main.copy_to_volume([{"sourcePath": "a", "targetPath": "b"}]))
            self.assertEqual(result, (True, ""))
            mocked.assert_awaited_once()
            args, _kwargs = mocked.await_args
            self.assertEqual(args[0], "copy-to-volume")

        with patch("backend.app.main.request_volume_server", new=AsyncMock(return_value=(True, ""))) as mocked:
            result = asyncio.run(main.copy_to_host([{"sourcePath": "a", "targetPath": "b"}]))
            self.assertEqual(result, (True, ""))
            args, _kwargs = mocked.await_args
            self.assertEqual(args[0], "copy-to-host")

    def test_download_file_success_and_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = f"{tmpdir}/model.pt"
            response = MagicMock()
            response.status_code = 200
            response.iter_content.return_value = [b"abc", b"def"]

            with patch("backend.app.main.requests.get", return_value=response):
                main.download_file("https://example/model.pt", out_path)

            with open(out_path, "rb") as handle:
                self.assertEqual(handle.read(), b"abcdef")

        bad_response = MagicMock()
        bad_response.status_code = 500
        with patch("backend.app.main.requests.get", return_value=bad_response):
            with self.assertRaises(Exception):
                main.download_file("https://example/model.pt", "/tmp/unused.pt")

    def test_download_file_replaces_directory_at_target_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "model.pt"
            target.mkdir()

            response = MagicMock()
            response.status_code = 200
            response.iter_content.return_value = [b"abc"]
            with patch("backend.app.main.requests.get", return_value=response):
                main.download_file("https://example/model.pt", str(target))

            self.assertTrue(target.is_file())
            self.assertEqual(target.read_bytes(), b"abc")

    def test_process_stack_creates_job_and_background_task(self):
        req = main.ProcessRequest(
            file_path="/host/input.tif",
            output_file="/host/output.tif",
            model_type="sam2_hiera_tiny",
            predictor_type="ImagePredictor",
        )
        bg = BackgroundTasks()
        with patch("backend.app.main.uuid.uuid4", return_value="job-abc"), patch(
            "backend.app.main.time.time",
            return_value=100.0,
        ):
            result = asyncio.run(main.process_stack(req, bg))

        self.assertEqual(result, {"job_id": "job-abc", "status": "started"})
        self.assertIn("job-abc", main.jobs)
        self.assertEqual(len(bg.tasks), 1)
        task = bg.tasks[0]
        self.assertIs(task.func, main.run_pipeline)
        self.assertEqual(task.args, ("job-abc", req.file_path, req.output_file, req.model_type, req.predictor_type))

    def test_get_status_success_and_not_found(self):
        main.jobs = {"job1": {"status": "running"}}
        self.assertEqual(asyncio.run(main.get_status("job1")), {"status": "running"})
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(main.get_status("missing"))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_get_latest_job_none_when_no_running(self):
        main.jobs = {"done": {"status": "completed", "created_at": 10}}
        result = asyncio.run(main.get_latest_job())
        self.assertEqual(result, {"job_id": None, "status": "none"})

    def test_get_latest_job_selects_latest_running(self):
        main.jobs = {
            "a": {"status": "running", "created_at": 1},
            "b": {"status": "running", "created_at": 3},
            "c": {"status": "completed", "created_at": 99},
        }
        result = asyncio.run(main.get_latest_job())
        self.assertEqual(result, {"job_id": "b", "status": "running"})

    def test_get_predictor_cache_hit(self):
        sentinel = object()
        main.loaded_model_name = "sam2_hiera_tiny_ImagePredictor"
        main.loaded_predictor = sentinel
        predictor = main.get_predictor("sam2_hiera_tiny", "ImagePredictor")
        self.assertIs(predictor, sentinel)

    def test_get_predictor_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            main.get_predictor("nope", "ImagePredictor")

    def test_get_predictor_unknown_predictor_type_for_sam2_raises(self):
        main.loaded_model_name = None
        main.loaded_predictor = None

        with patch("backend.app.main.SAM2ImagePredictor", object()), patch(
            "backend.app.main.build_sam2",
            lambda *args, **kwargs: object(),
        ), patch("backend.app.main.os.path.isfile", return_value=True):
            with self.assertRaises(ValueError):
                main.get_predictor("sam2_hiera_tiny", "BadPredictor")

    def test_get_predictor_sam2_import_error_when_library_missing(self):
        with patch("backend.app.main.SAM2ImagePredictor", None):
            with self.assertRaises(ImportError):
                main.get_predictor("sam2_hiera_tiny", "ImagePredictor")

    def test_get_predictor_sam2_image_predictor_success(self):
        main.loaded_model_name = None
        main.loaded_predictor = None

        with patch("backend.app.main.SAM2ImagePredictor", side_effect=lambda model: ("wrapped", model)), patch(
            "backend.app.main.build_sam2",
            return_value="built-model",
        ), patch("backend.app.main.os.path.isfile", return_value=True):
            predictor = main.get_predictor("sam2_hiera_tiny", "ImagePredictor")
        self.assertEqual(predictor, ("wrapped", "built-model"))

    def test_get_predictor_sam2_video_predictor_success(self):
        main.loaded_model_name = None
        main.loaded_predictor = None

        with patch("backend.app.main.SAM2ImagePredictor", object()), patch(
            "backend.app.main.build_sam2_video_predictor",
            return_value="video-predictor",
        ), patch("backend.app.main.os.path.isfile", return_value=True):
            predictor = main.get_predictor("sam2_hiera_tiny", "VideoPredictor")
        self.assertEqual(predictor, "video-predictor")

    def test_jpeg_convert_writes_rgb_jpg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir, "input.tif")
            out_path = Path(tmpdir, "out")
            tf.imwrite(img_path, np.arange(16, dtype=np.uint16).reshape(4, 4))

            main.jpeg_convert(img_path, out_path)
            jpg_path = out_path.with_suffix(".jpg")
            self.assertTrue(jpg_path.exists())
            with Image.open(jpg_path) as image:
                self.assertEqual(image.mode, "RGB")

    def test_downsample_grayscale_expands_to_three_channels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir, "input.tif")
            out_path = Path(tmpdir, "out")
            tf.imwrite(img_path, np.arange(16, dtype=np.uint16).reshape(4, 4))

            main.downsample(img_path, out_path)
            written = tf.imread(out_path.with_suffix(".tif"))
            self.assertEqual(written.shape, (4, 4, 3))

    def test_downsample_trims_extra_channels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir, "input.tif")
            out_path = Path(tmpdir, "out")
            data = np.random.randint(0, 255, size=(4, 4, 5), dtype=np.uint8)
            tf.imwrite(img_path, data)

            main.downsample(img_path, out_path)
            written = tf.imread(out_path.with_suffix(".tif"))
            self.assertEqual(written.shape, (4, 4, 3))


if __name__ == "__main__":
    unittest.main()
