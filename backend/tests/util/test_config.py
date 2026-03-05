import copy
import os
import tempfile
import unittest
from unittest.mock import patch

_TEST_VOLUME_ROOT = tempfile.mkdtemp(prefix="ouroboros-test-volume-")
os.environ["VOLUME_MOUNT_PATH"] = _TEST_VOLUME_ROOT

from backend.app.util import config as app_config  # noqa: E402


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.startup_status_original = copy.deepcopy(app_config.startup_status)

    def tearDown(self):
        app_config.startup_status = self.startup_status_original

    def test_ensure_checkpoint_dir_creates_directory(self):
        with patch("backend.app.util.config.os.makedirs") as mocked_makedirs:
            app_config.ensure_checkpoint_dir()
        mocked_makedirs.assert_called_once_with(app_config.CHECKPOINT_DIR, exist_ok=True)

    def test_update_initialization_step_sets_status_and_start_time(self):
        app_config.startup_status = {
            "is_ready": False,
            "initialization_steps": [{"name": "Step", "status": "pending"}],
            "start_time": None,
            "ready_time": None,
        }
        with patch("backend.app.util.config.time.time", return_value=123.0):
            app_config.update_initialization_step("Step", "completed")
        self.assertEqual(app_config.startup_status["initialization_steps"][0]["status"], "completed")
        self.assertEqual(app_config.startup_status["start_time"], 123.0)

    def test_update_initialization_step_unknown_step_only_sets_start_time(self):
        app_config.startup_status = {
            "is_ready": False,
            "initialization_steps": [{"name": "Known", "status": "pending"}],
            "start_time": None,
            "ready_time": None,
        }
        with patch("backend.app.util.config.time.time", return_value=111.0):
            app_config.update_initialization_step("Missing", "warning")
        self.assertEqual(app_config.startup_status["initialization_steps"][0]["status"], "pending")
        self.assertEqual(app_config.startup_status["start_time"], 111.0)

    def test_mark_initialization_complete_marks_pending_steps(self):
        app_config.startup_status = {
            "is_ready": False,
            "initialization_steps": [
                {"name": "A", "status": "pending"},
                {"name": "B", "status": "warning"},
            ],
            "start_time": None,
            "ready_time": None,
        }
        with patch("backend.app.util.config.time.time", return_value=555.0):
            app_config.mark_initialization_complete()
        self.assertTrue(app_config.startup_status["is_ready"])
        self.assertEqual(app_config.startup_status["ready_time"], 555.0)
        self.assertEqual(app_config.startup_status["initialization_steps"][0]["status"], "completed")
        self.assertEqual(app_config.startup_status["initialization_steps"][1]["status"], "warning")


if __name__ == "__main__":
    unittest.main()
