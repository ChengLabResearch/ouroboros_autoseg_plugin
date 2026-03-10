import asyncio
import copy
import os
import tempfile
import types
import unittest
from unittest.mock import patch

from fastapi import BackgroundTasks, HTTPException

_TEST_VOLUME_ROOT = tempfile.mkdtemp(prefix="ouroboros-test-volume-")
os.environ["VOLUME_MOUNT_PATH"] = _TEST_VOLUME_ROOT

from backend.app import main  # noqa: E402
from backend.app.util import config as app_config  # noqa: E402
from backend.app.util import network as app_network  # noqa: E402
from backend.app.util.util import DownloadRequest, ProcessRequest  # noqa: E402


class MainTests(unittest.TestCase):
    def setUp(self):
        self.startup_status_original = copy.deepcopy(app_config.startup_status)
        self.jobs_original = copy.deepcopy(app_config.jobs)

    def tearDown(self):
        app_config.startup_status = self.startup_status_original
        app_config.jobs = self.jobs_original

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

    def test_refresh_startup_status_completed(self):
        app_config.startup_status = {
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
        with patch("backend.app.main.network.requests.get", return_value=response), patch(
            "backend.app.main.network.models_available",
            return_value=True,
        ), patch("backend.app.main.time.time", return_value=321.0):
            status = main.refresh_startup_status()
        self.assertTrue(status["is_ready"])
        self.assertEqual(status["ready_time"], 321.0)

    def test_refresh_startup_status_warning_path(self):
        app_config.startup_status = {
            "is_ready": False,
            "initialization_steps": [
                {"name": "Building Docker Image", "status": "pending"},
                {"name": "Connecting to Volume Server", "status": "pending"},
                {"name": "Initializing ML Models", "status": "pending"},
            ],
            "start_time": None,
            "ready_time": None,
        }
        with patch("backend.app.main.network.requests.get", side_effect=app_network.requests.RequestException), patch(
            "backend.app.main.network.models_available",
            return_value=False,
        ):
            status = main.refresh_startup_status()
        self.assertFalse(status["is_ready"])
        self.assertIsNone(status["ready_time"])

    def test_refresh_startup_status_non_ok_response_sets_warning(self):
        app_config.startup_status = {
            "is_ready": False,
            "initialization_steps": [
                {"name": "Building Docker Image", "status": "pending"},
                {"name": "Connecting to Volume Server", "status": "pending"},
                {"name": "Initializing ML Models", "status": "pending"},
            ],
            "start_time": None,
            "ready_time": None,
        }
        response = types.SimpleNamespace(ok=False, status_code=500)
        with patch("backend.app.main.network.requests.get", return_value=response), patch(
            "backend.app.main.network.models_available",
            return_value=True,
        ):
            status = main.refresh_startup_status()
        self.assertFalse(status["is_ready"])
        connect = next(step for step in status["initialization_steps"] if step["name"] == "Connecting to Volume Server")
        self.assertEqual(connect["status"], "warning")

    def test_download_model_exists_path(self):
        req = DownloadRequest(model_type="sam2_hiera_tiny")
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
        req = DownloadRequest(model_type="sam2_hiera_tiny")
        with patch("backend.app.main.os.makedirs"), patch(
            "backend.app.main.os.path.isfile",
            return_value=False,
        ), patch(
            "backend.app.main.os.path.isdir",
            return_value=False,
        ), patch("backend.app.main.network.download_file") as mocked_download:
            result = asyncio.run(main.download_model(req))
        self.assertEqual(result["status"], "success")
        mocked_download.assert_called_once()

    def test_download_model_sam3_success(self):
        req = DownloadRequest(model_type="sam3", hf_token="token")
        with patch("backend.app.main.os.makedirs"), patch(
            "backend.app.main.os.path.isfile",
            return_value=False,
        ), patch(
            "backend.app.main.os.path.isdir",
            return_value=False,
        ), patch("backend.app.main.network.download_sam3_checkpoint") as mocked_download:
            result = asyncio.run(main.download_model(req))
        self.assertEqual(result["status"], "success")
        mocked_download.assert_called_once()

    def test_download_model_sam3_without_token_raises_http_500(self):
        req = DownloadRequest(model_type="sam3")
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
        self.assertIn("Authentication Token required for SAM 3", str(ctx.exception.detail))

    def test_download_model_sam3_download_failure_raises_http_500(self):
        req = DownloadRequest(model_type="sam3", hf_token="token")
        with patch("backend.app.main.os.makedirs"), patch(
            "backend.app.main.os.path.isfile",
            return_value=False,
        ), patch(
            "backend.app.main.os.path.isdir",
            return_value=False,
        ), patch(
            "backend.app.main.network.download_sam3_checkpoint",
            side_effect=RuntimeError("hf down"),
        ):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(main.download_model(req))
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("Hugging Face Download Failed", str(ctx.exception.detail))

    def test_download_model_unknown_type_raises_http_500(self):
        req = DownloadRequest(model_type="unknown-model")
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
        req = DownloadRequest(model_type="sam2_hiera_tiny")
        target_path = os.path.join(app_config.CHECKPOINT_DIR, f"{req.model_type}.pt")
        with patch("backend.app.main.os.makedirs"), patch(
            "backend.app.main.os.path.isfile",
            return_value=False,
        ), patch(
            "backend.app.main.os.path.isdir",
            side_effect=lambda path: str(path) == str(target_path),
        ), patch("backend.app.main.shutil.rmtree") as mocked_rmtree, patch(
            "backend.app.main.network.download_file"
        ):
            result = asyncio.run(main.download_model(req))
        self.assertEqual(result["status"], "success")
        mocked_rmtree.assert_called_once()

    def test_process_stack_creates_job_and_background_task(self):
        req = ProcessRequest(
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
        self.assertIn("job-abc", app_config.jobs)
        self.assertEqual(len(bg.tasks), 1)
        task = bg.tasks[0]
        self.assertIs(task.func, main.run_pipeline)
        self.assertEqual(
            task.args,
            (
                "job-abc",
                req.file_path,
                req.output_file,
                req.model_type,
                req.predictor_type,
                req.overlay_annotation_points,
                req.annotation_overlay_intensity,
            ),
        )

    def test_get_status_success_and_not_found(self):
        app_config.jobs = {"job1": {"status": "running"}}
        self.assertEqual(asyncio.run(main.get_status("job1")), {"status": "running"})
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(main.get_status("missing"))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_get_latest_job_none_when_no_running(self):
        app_config.jobs = {"done": {"status": "completed", "created_at": 10}}
        result = asyncio.run(main.get_latest_job())
        self.assertEqual(result, {"job_id": None, "status": "none"})

    def test_get_latest_job_selects_latest_running(self):
        app_config.jobs = {
            "a": {"status": "running", "created_at": 1},
            "b": {"status": "running", "created_at": 3},
            "c": {"status": "completed", "created_at": 99},
        }
        result = asyncio.run(main.get_latest_job())
        self.assertEqual(result, {"job_id": "b", "status": "running"})


if __name__ == "__main__":
    unittest.main()
