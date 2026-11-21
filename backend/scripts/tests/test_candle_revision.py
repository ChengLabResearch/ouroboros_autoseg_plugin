from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


BACKEND = Path(__file__).resolve().parents[2]
HELPER = BACKEND / "scripts" / "candle_revision.py"
LOCKFILE = BACKEND / "Cargo.lock"
with LOCKFILE.open("rb") as handle:
    PACKAGE = next(package for package in tomllib.load(handle)["package"] if package["name"] == "candle-sam3")
REVISION = PACKAGE["source"].rsplit("#", 1)[1]


class CandleRevisionTests(unittest.TestCase):
    def run_helper(self, log=None):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, str(HELPER), str(LOCKFILE)]
            if log is not None:
                logfile = Path(directory) / "backend.log"
                logfile.write_text(log)
                command.append(str(logfile))
            return subprocess.run(command, capture_output=True, text=True)

    def test_reads_actual_lockfile(self):
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), REVISION)

    def test_accepts_matching_startup_and_model_load_fields(self):
        log = (
            f"2026-11-20T12:00:00Z INFO backend revision metadata plugin_sha=plugin candle_sha={REVISION}\n"
            f"2026-11-20T12:01:00Z INFO loading SAM3 model and tracker device=cuda candle_sha={REVISION}\n"
        )
        result = self.run_helper(log)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_mismatch_even_with_a_matching_field(self):
        for other in ("0" * 40, "unknown", ""):
            with self.subTest(other=other):
                result = self.run_helper(f"INFO startup candle_sha={REVISION}\nINFO loading candle_sha={other}\n")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("expected candle_sha=", result.stderr)

    def test_rejects_missing_field(self):
        result = self.run_helper("INFO backend revision metadata plugin_sha=plugin\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no candle_sha field", result.stderr)


if __name__ == "__main__":
    unittest.main()
