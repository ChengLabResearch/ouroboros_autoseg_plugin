import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen


MODULE_PATH = Path(__file__).resolve().parents[1] / "smoke_volume_server.py"
SPEC = importlib.util.spec_from_file_location("smoke_volume_server", MODULE_PATH)
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)


class SmokeVolumeServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input = self.root / "host input" / "stack 'with spaces'.tif"
        self.output = self.root / "host output" / "mask.tif"
        self.overlay = self.output.with_name("mask_annotation_overlay.tif")
        self.input.parent.mkdir()
        self.output.parent.mkdir()
        self.input.write_bytes(b"host TIFF input")
        self.volume = self.root / "physical volume"
        self.volume.mkdir()
        self.events = self.root / "events.jsonl"
        self.transfers = fixture.SmokeTransfers(
            self.input, [self.output, self.overlay], "isolated-benchmark-volume",
            "backend@sha256:fixture", self.events,
        )
        self.server = fixture.make_server(self.transfers, ("127.0.0.1", 0))
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temporary.cleanup()

    def post(self, operation, paths, target, **overrides):
        payload = {
            "volumeName": "ouroboros-volume",
            "pluginFolderName": "sam3-segmentation",
            "files": [{"sourcePath": str(path), "targetPath": target} for path in paths],
            **overrides,
        }
        request = Request(
            f"{self.base_url}/{operation}", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, response.read().decode()
        except HTTPError as error:
            return error.code, error.read().decode()

    def test_http_roundtrip_returns_primary_and_overlay_and_records_transfers(self):
        def copy_in_test(operation, host_path, target):
            volume_path = self.volume / "sam3-segmentation" / target / host_path.name
            if operation == "copy-to-volume":
                volume_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(host_path, volume_path)
            else:
                shutil.copyfile(volume_path, host_path)

        with patch.object(self.transfers, "copy", side_effect=copy_in_test):
            with urlopen(self.base_url, timeout=5) as response:
                self.assertEqual(response.status, 200)
            self.assertEqual(self.post("copy-to-volume", [self.input], "")[0], 200)
            self.assertEqual(
                (self.volume / "sam3-segmentation" / self.input.name).read_bytes(),
                self.input.read_bytes(),
            )
            segmentation = self.volume / "sam3-segmentation" / "Segmentation"
            segmentation.mkdir()
            (segmentation / self.output.name).write_bytes(b"binary mask")
            (segmentation / self.overlay.name).write_bytes(b"annotation overlay")
            self.assertFalse(self.output.exists())
            self.assertEqual(
                self.post("copy-to-host", [self.output, self.overlay], "Segmentation")[0], 200,
            )
            self.assertEqual(self.output.read_bytes(), b"binary mask")
            self.assertEqual(self.overlay.read_bytes(), b"annotation overlay")
        records = [json.loads(line) for line in self.events.read_text().splitlines()]
        self.assertEqual([record["operation"] for record in records], ["copy-to-volume", "copy-to-host"])
        self.assertEqual(records[1]["files"][1]["sourcePath"], str(self.overlay))

    def test_rejects_unrelated_paths_and_volumes_before_any_copy(self):
        with patch.object(self.transfers, "copy") as copy:
            self.assertEqual(self.post("copy-to-volume", [self.input], "", volumeName="other")[0], 400)
            self.assertEqual(self.post("copy-to-volume", [self.input], "../other")[0], 400)
            self.assertEqual(self.post("copy-to-host", [self.output, self.root / "other"], "Segmentation")[0], 400)
            self.assertEqual(self.post("copy-to-host", [self.input], "Segmentation")[0], 400)
            self.assertEqual(self.post("clear-volume", [self.input], "")[0], 404)
            copy.assert_not_called()
        self.assertFalse(self.events.exists())

    def test_copy_failure_returns_error_without_recording_success(self):
        error = subprocess.CalledProcessError(1, ["docker"], stderr="source missing")
        with patch.object(self.transfers, "copy", side_effect=error):
            status, body = self.post("copy-to-volume", [self.input], "")
        self.assertEqual(status, 500)
        self.assertIn("source missing", body)
        self.assertFalse(self.events.exists())

    def test_docker_boundary_uses_physical_volume_and_preserves_host_filename(self):
        with patch.object(fixture.subprocess, "run") as run:
            self.transfers.copy("copy-to-volume", self.input, "")
            self.transfers.copy("copy-to-host", self.output, "Segmentation")
        input_argv = run.call_args_list[0].args[0]
        output_argv = run.call_args_list[1].args[0]
        for argv in (input_argv, output_argv):
            self.assertIn("isolated-benchmark-volume:/volume", argv)
            self.assertNotIn("ouroboros-volume:/volume", argv)
            self.assertIn("backend@sha256:fixture", argv)
        self.assertIn(f"{self.input.parent}:/host:ro", input_argv)
        self.assertIn(f"FILE_NAME={self.input.name}", input_argv)
        self.assertNotIn(self.input.name, input_argv[-1])
        self.assertIn(f"{self.output.parent}:/host", output_argv)
        self.assertIn("VOLUME_FOLDER=/volume/sam3-segmentation/Segmentation", output_argv)


if __name__ == "__main__":
    unittest.main()
