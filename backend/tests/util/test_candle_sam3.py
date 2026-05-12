"""Unit tests for :mod:`backend.app.util.candle_sam3`."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from backend.app.util import candle_sam3 as adapter_mod
from backend.app.util.candle_sam3 import (
    CandleSam3Adapter,
    CandleSam3AdapterError,
    CandleSam3Box,
    CandleSam3Job,
    CandleSam3Point,
    DEFAULT_SAM3_BINARY,
    SAM3_BINARY_ENV,
)


def _make_mask_png(path: Path, *, height: int = 4, width: int = 6, fill: int = 255) -> None:
    arr = np.full((height, width), fill, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


class IsAvailableTests(unittest.TestCase):
    def test_returns_false_for_missing_binary(self):
        adapter = CandleSam3Adapter(binary_path="/nonexistent/path/to/sam3")
        self.assertFalse(adapter.is_available())

    def test_returns_false_when_path_is_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = CandleSam3Adapter(binary_path=tmp)
            self.assertFalse(adapter.is_available())

    def test_returns_true_when_path_is_executable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = Path(tmp) / "sam3"
            bin_path.write_text("#!/bin/sh\nexit 0\n")
            bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            adapter = CandleSam3Adapter(binary_path=str(bin_path))
            self.assertTrue(adapter.is_available())

    def test_resolves_binary_from_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = Path(tmp) / "sam3"
            bin_path.write_text("")
            bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR)
            original = os.environ.get(SAM3_BINARY_ENV)
            try:
                os.environ[SAM3_BINARY_ENV] = str(bin_path)
                adapter = CandleSam3Adapter()
                self.assertEqual(adapter.binary_path, bin_path)
            finally:
                if original is None:
                    os.environ.pop(SAM3_BINARY_ENV, None)
                else:
                    os.environ[SAM3_BINARY_ENV] = original

    def test_falls_back_to_default_binary_path(self):
        original = os.environ.pop(SAM3_BINARY_ENV, None)
        try:
            adapter = CandleSam3Adapter()
            self.assertEqual(adapter.binary_path, Path(DEFAULT_SAM3_BINARY))
        finally:
            if original is not None:
                os.environ[SAM3_BINARY_ENV] = original


class NormalizeXyTests(unittest.TestCase):
    def test_basic_division(self):
        self.assertEqual(
            CandleSam3Adapter.normalize_xy(50, 25, width=100, height=100),
            (0.5, 0.25),
        )

    def test_clamps_below_zero(self):
        x, y = CandleSam3Adapter.normalize_xy(-10, -5, width=100, height=100)
        self.assertEqual((x, y), (0.0, 0.0))

    def test_clamps_above_one(self):
        x, y = CandleSam3Adapter.normalize_xy(200, 300, width=100, height=100)
        self.assertEqual((x, y), (1.0, 1.0))

    def test_rejects_non_positive_dimensions(self):
        with self.assertRaises(ValueError):
            CandleSam3Adapter.normalize_xy(1, 1, width=0, height=10)
        with self.assertRaises(ValueError):
            CandleSam3Adapter.normalize_xy(1, 1, width=10, height=-1)


class BuildBatchManifestTests(unittest.TestCase):
    def test_minimal_prompt_only_job(self):
        job = CandleSam3Job(name="alpha", image=Path("/tmp/a.png"), prompt="cat")
        manifest = CandleSam3Adapter.build_batch_manifest([job])
        self.assertEqual(
            manifest,
            {
                "jobs": [
                    {
                        "name": "alpha",
                        "image": "/tmp/a.png",
                        "prompt": "cat",
                        "points": [],
                        "boxes": [],
                    }
                ]
            },
        )

    def test_omits_prompt_when_none(self):
        job = CandleSam3Job(name="bare", image=Path("/tmp/b.png"))
        manifest = CandleSam3Adapter.build_batch_manifest([job])
        self.assertNotIn("prompt", manifest["jobs"][0])

    def test_geometry_serialization_order_and_defaults(self):
        job = CandleSam3Job(
            name="geo",
            image=Path("/tmp/c.png"),
            points=[CandleSam3Point(x=0.1, y=0.2), CandleSam3Point(x=0.3, y=0.4, label=0)],
            boxes=[CandleSam3Box(cx=0.5, cy=0.5, w=0.2, h=0.2)],
        )
        manifest = CandleSam3Adapter.build_batch_manifest([job])
        self.assertEqual(
            manifest["jobs"][0]["points"],
            [
                {"x": 0.1, "y": 0.2, "label": 1},
                {"x": 0.3, "y": 0.4, "label": 0},
            ],
        )
        self.assertEqual(
            manifest["jobs"][0]["boxes"],
            [{"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "label": 1}],
        )


class LoadMaskPngTests(unittest.TestCase):
    def test_binarizes_grayscale_at_127(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mask.png"
            arr = np.array([[0, 64, 127], [128, 200, 255]], dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(path)
            out = adapter_mod.CandleSam3Adapter.load_mask_png(path)
            np.testing.assert_array_equal(
                out,
                np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8),
            )

    def test_collapses_rgba_to_luminance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mask.png"
            arr = np.zeros((3, 3, 4), dtype=np.uint8)
            arr[..., :3] = 255
            arr[..., 3] = 255
            Image.fromarray(arr, mode="RGBA").save(path)
            out = adapter_mod.CandleSam3Adapter.load_mask_png(path)
            np.testing.assert_array_equal(out, np.full((3, 3), 255, dtype=np.uint8))


class PredictImageBatchTests(unittest.TestCase):
    def test_empty_jobs_returns_empty_list_without_running_binary(self):
        captured: list = []
        adapter = CandleSam3Adapter(binary_path="/fake/sam3")
        adapter._run_binary = lambda cmd: captured.append(cmd)  # type: ignore[assignment]
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(adapter.predict_image_batch([], Path(tmp)), [])
        self.assertEqual(captured, [])

    def test_runs_binary_with_expected_args_and_returns_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            jobs = [
                CandleSam3Job(name="job1", image=Path("/img/1.png"), prompt="x"),
                CandleSam3Job(name="job2", image=Path("/img/2.png"), prompt="y"),
            ]

            captured: list[Sequence[str]] = []

            def fake_run(cmd: Sequence[str]):
                captured.append(list(cmd))
                for job in jobs:
                    (output_dir / job.name).mkdir(parents=True, exist_ok=True)
                    _make_mask_png(output_dir / job.name / "mask.png")

            adapter = CandleSam3Adapter(
                binary_path="/fake/sam3",
                checkpoint_path="/fake/sam3.pt",
                tokenizer_path="/fake/tokenizer.json",
                cpu=True,
            )
            adapter._run_binary = fake_run  # type: ignore[assignment]
            masks = adapter.predict_image_batch(jobs, output_dir)

            self.assertEqual(len(masks), 2)
            for mask in masks:
                self.assertEqual(mask.dtype, np.uint8)
                self.assertEqual(set(np.unique(mask).tolist()), {255})

            self.assertEqual(len(captured), 1)
            cmd = captured[0]
            self.assertEqual(cmd[0], "/fake/sam3")
            self.assertIn("--checkpoint", cmd)
            self.assertIn("/fake/sam3.pt", cmd)
            self.assertIn("--tokenizer", cmd)
            self.assertIn("/fake/tokenizer.json", cmd)
            self.assertIn("--cpu", cmd)
            self.assertIn("--batch-manifest", cmd)
            self.assertIn("--output-dir", cmd)
            self.assertEqual(cmd[cmd.index("--output-dir") + 1], str(output_dir))

            manifest_path = output_dir / "manifest.json"
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(
                json.loads(manifest_path.read_text())["jobs"][0]["name"], "job1"
            )

    def test_missing_mask_raises_adapter_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            jobs = [CandleSam3Job(name="solo", image=Path("/img.png"), prompt="z")]
            output_dir.mkdir(parents=True)
            (output_dir / "solo").mkdir()
            adapter = CandleSam3Adapter(binary_path="/fake/sam3")
            adapter._run_binary = lambda cmd: None  # type: ignore[assignment]
            with self.assertRaises(CandleSam3AdapterError):
                adapter.predict_image_batch(jobs, output_dir)

    def test_omits_optional_flags_when_unset(self):
        captured: list[Sequence[str]] = []

        def fake_run(cmd: Sequence[str]):
            captured.append(list(cmd))
            (output_dir / "only").mkdir(parents=True, exist_ok=True)
            _make_mask_png(output_dir / "only" / "mask.png")

        adapter = CandleSam3Adapter(binary_path="/fake/sam3")
        adapter._run_binary = fake_run  # type: ignore[assignment]

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            adapter.predict_image_batch(
                [CandleSam3Job(name="only", image=Path("/x.png"), prompt="z")],
                output_dir,
            )

        cmd = captured[0]
        self.assertNotIn("--checkpoint", cmd)
        self.assertNotIn("--tokenizer", cmd)
        self.assertNotIn("--cpu", cmd)


class RunBinaryTests(unittest.TestCase):
    def test_called_process_error_raises_adapter_error(self):
        adapter = CandleSam3Adapter(binary_path="/bin/false")
        with self.assertRaises(CandleSam3AdapterError) as cm:
            adapter._run_binary(["/bin/false"])
        self.assertIn("candle-sam3 binary failed", str(cm.exception))

    def test_successful_invocation_returns_completed_process(self):
        adapter = CandleSam3Adapter(binary_path="/bin/true")
        result = adapter._run_binary(["/bin/echo", "hello"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("hello", result.stdout)


if __name__ == "__main__":
    unittest.main()
