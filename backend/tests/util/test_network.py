import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

_TEST_VOLUME_ROOT = tempfile.mkdtemp(prefix="ouroboros-test-volume-")
os.environ["VOLUME_MOUNT_PATH"] = _TEST_VOLUME_ROOT

from backend.app.util import config as app_config  # noqa: E402
from backend.app.util import network as app_network  # noqa: E402


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.loaded_model_name_original = app_config.loaded_model_name
        self.loaded_predictor_original = app_config.loaded_predictor

    def tearDown(self):
        app_config.loaded_model_name = self.loaded_model_name_original
        app_config.loaded_predictor = self.loaded_predictor_original

    def test_models_available_true_and_false(self):
        with patch("backend.app.util.network.build_sam2", object()), patch(
            "backend.app.util.network.SAM2ImagePredictor", object()
        ), patch("backend.app.util.network.build_sam3_video_predictor", None), patch(
            "backend.app.util.network.Sam3Processor", None
        ):
            self.assertTrue(app_network.models_available())

        with patch("backend.app.util.network.build_sam2", None), patch(
            "backend.app.util.network.SAM2ImagePredictor", None
        ), patch("backend.app.util.network.build_sam3_video_predictor", None), patch(
            "backend.app.util.network.Sam3Processor", None
        ):
            self.assertFalse(app_network.models_available())

    def test_request_volume_server_success_and_failure(self):
        ok_response = MagicMock(ok=True, text="ok")
        fail_response = MagicMock(ok=False, text="bad")

        with patch("backend.app.util.network.requests.post", return_value=ok_response):
            success, msg = asyncio.run(app_network.request_volume_server("path", {"a": 1}))
        self.assertTrue(success)
        self.assertEqual(msg, "")

        with patch("backend.app.util.network.requests.post", return_value=fail_response):
            success, msg = asyncio.run(app_network.request_volume_server("path", {"a": 1}))
        self.assertFalse(success)
        self.assertEqual(msg, "bad")

    def test_request_volume_server_handles_exception(self):
        with patch("backend.app.util.network.requests.post", side_effect=RuntimeError("network down")):
            success, msg = asyncio.run(app_network.request_volume_server("path", {"a": 1}))
        self.assertFalse(success)
        self.assertIn("network down", msg)

    def test_copy_helpers_delegate_to_request_volume_server(self):
        with patch("backend.app.util.network.request_volume_server", new=AsyncMock(return_value=(True, ""))) as mocked:
            result = asyncio.run(app_network.copy_to_volume([{"sourcePath": "a", "targetPath": "b"}]))
            self.assertEqual(result, (True, ""))
            mocked.assert_awaited_once()
            args, _kwargs = mocked.await_args
            self.assertEqual(args[0], "copy-to-volume")

        with patch("backend.app.util.network.request_volume_server", new=AsyncMock(return_value=(True, ""))) as mocked:
            result = asyncio.run(app_network.copy_to_host([{"sourcePath": "a", "targetPath": "b"}]))
            self.assertEqual(result, (True, ""))
            args, _kwargs = mocked.await_args
            self.assertEqual(args[0], "copy-to-host")

    def test_download_file_success_and_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = f"{tmpdir}/model.pt"
            response = MagicMock()
            response.status_code = 200
            response.iter_content.return_value = [b"abc", b"def"]

            with patch("backend.app.util.network.requests.get", return_value=response):
                app_network.download_file("https://example/model.pt", out_path)

            with open(out_path, "rb") as handle:
                self.assertEqual(handle.read(), b"abcdef")

        bad_response = MagicMock()
        bad_response.status_code = 500
        with patch("backend.app.util.network.requests.get", return_value=bad_response):
            with self.assertRaises(Exception):
                app_network.download_file("https://example/model.pt", "/tmp/unused.pt")

    def test_download_file_replaces_directory_at_target_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "model.pt"
            target.mkdir()

            response = MagicMock()
            response.status_code = 200
            response.iter_content.return_value = [b"abc"]
            with patch("backend.app.util.network.requests.get", return_value=response):
                app_network.download_file("https://example/model.pt", str(target))

            self.assertTrue(target.is_file())
            self.assertEqual(target.read_bytes(), b"abc")

    def test_get_predictor_cache_hit(self):
        sentinel = object()
        app_config.loaded_model_name = "sam2_hiera_tiny_ImagePredictor"
        app_config.loaded_predictor = sentinel

        predictor = app_network.get_predictor("sam2_hiera_tiny", "ImagePredictor")
        self.assertIs(predictor, sentinel)

    def test_get_predictor_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            app_network.get_predictor("nope", "ImagePredictor")

    def test_get_predictor_unknown_predictor_type_for_sam2_raises(self):
        app_config.loaded_model_name = None
        app_config.loaded_predictor = None

        with patch("backend.app.util.network.SAM2ImagePredictor", object()), patch(
            "backend.app.util.network.build_sam2",
            lambda *args, **kwargs: object(),
        ), patch("backend.app.util.network.os.path.isfile", return_value=True):
            with self.assertRaises(ValueError):
                app_network.get_predictor("sam2_hiera_tiny", "BadPredictor")

    def test_get_predictor_sam2_import_error_when_library_missing(self):
        with patch("backend.app.util.network.SAM2ImagePredictor", None):
            with self.assertRaises(ImportError):
                app_network.get_predictor("sam2_hiera_tiny", "ImagePredictor")

    def test_get_predictor_sam2_download_failure_raises_runtime_error(self):
        with patch("backend.app.util.network.SAM2ImagePredictor", object()), patch(
            "backend.app.util.network.os.path.isfile",
            return_value=False,
        ), patch("backend.app.util.network.download_file", side_effect=Exception("download failed")):
            with self.assertRaises(RuntimeError) as ctx:
                app_network.get_predictor("sam2_hiera_tiny", "ImagePredictor")
        self.assertIn("download failed", str(ctx.exception))

    def test_get_predictor_sam2_image_predictor_success(self):
        app_config.loaded_model_name = None
        app_config.loaded_predictor = None

        with patch("backend.app.util.network.SAM2ImagePredictor", side_effect=lambda model: ("wrapped", model)), patch(
            "backend.app.util.network.build_sam2",
            return_value="built-model",
        ), patch("backend.app.util.network.os.path.isfile", return_value=True):
            predictor = app_network.get_predictor("sam2_hiera_tiny", "ImagePredictor")
        self.assertEqual(predictor, ("wrapped", "built-model"))

    def test_get_predictor_sam2_video_predictor_success(self):
        app_config.loaded_model_name = None
        app_config.loaded_predictor = None

        with patch("backend.app.util.network.SAM2ImagePredictor", object()), patch(
            "backend.app.util.network.build_sam2_video_predictor",
            return_value="video-predictor",
        ), patch("backend.app.util.network.os.path.isfile", return_value=True):
            predictor = app_network.get_predictor("sam2_hiera_tiny", "VideoPredictor")
        self.assertEqual(predictor, "video-predictor")

    def test_get_predictor_sam3_import_error_when_library_missing(self):
        app_config.loaded_model_name = None
        app_config.loaded_predictor = None

        with patch("backend.app.util.network.Sam3Processor", None):
            with self.assertRaises(ImportError):
                app_network.get_predictor("sam3", "ImagePredictor")

    def test_get_predictor_sam3_checkpoint_missing_raises(self):
        app_config.loaded_model_name = None
        app_config.loaded_predictor = None

        with patch("backend.app.util.network.Sam3Processor", object()), patch(
            "backend.app.util.network.os.path.isfile", return_value=False
        ):
            with self.assertRaises(FileNotFoundError):
                app_network.get_predictor("sam3", "ImagePredictor")

    def test_get_predictor_unknown_predictor_type_for_sam3_raises(self):
        app_config.loaded_model_name = None
        app_config.loaded_predictor = None

        with patch("backend.app.util.network.Sam3Processor", object()), patch(
            "backend.app.util.network.os.path.isfile", return_value=True
        ):
            with self.assertRaises(ValueError):
                app_network.get_predictor("sam3", "BadPredictor")

    def test_get_predictor_sam3_image_predictor_success(self):
        app_config.loaded_model_name = None
        app_config.loaded_predictor = None

        with patch("backend.app.util.network.Sam3Processor", side_effect=lambda model: ("sam3", model)), patch(
            "backend.app.util.network.build_sam3_image_model",
            return_value="sam3-image-model",
        ), patch("backend.app.util.network.os.path.isfile", return_value=True):
            predictor = app_network.get_predictor("sam3", "ImagePredictor")
        self.assertEqual(predictor, ("sam3", "sam3-image-model"))


if __name__ == "__main__":
    unittest.main()
