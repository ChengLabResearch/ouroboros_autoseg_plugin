"""
Subprocess adapter for the candle-sam3 Rust example binary.

The candle-sam3 project (https://github.com/den-sq/candle_sam3) ships a CLI
example named ``sam3`` that runs SAM3 inference against a single image, a
batch manifest of images, or a video clip. This module wraps the image-batch
flow as a Python adapter so the plugin pipeline can invoke it without taking
a direct dependency on the upstream Python ``sam3`` package.

The adapter is intentionally pure: it only constructs CLI arguments, writes a
JSON manifest, runs the binary via :func:`subprocess.run`, and parses the
output PNG masks back into ``numpy`` arrays. All subprocess invocations go
through :meth:`CandleSam3Adapter._run_binary` so unit tests can monkeypatch a
single seam.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image


SAM3_BINARY_ENV = "SAM3_BINARY_PATH"
DEFAULT_SAM3_BINARY = "/opt/candle-sam3/sam3"
DEFAULT_MASK_FILENAME = "mask.png"


@dataclasses.dataclass(frozen=True)
class CandleSam3Point:
    """Normalized point prompt for the candle-sam3 batch manifest."""

    x: float
    y: float
    label: int = 1


@dataclasses.dataclass(frozen=True)
class CandleSam3Box:
    """Normalized box prompt (center + extent) for the candle-sam3 batch manifest."""

    cx: float
    cy: float
    w: float
    h: float
    label: int = 1


@dataclasses.dataclass(frozen=True)
class CandleSam3Job:
    """One image-prediction job in a candle-sam3 batch manifest."""

    name: str
    image: Path
    prompt: Optional[str] = None
    points: Sequence[CandleSam3Point] = ()
    boxes: Sequence[CandleSam3Box] = ()


class CandleSam3AdapterError(RuntimeError):
    """Raised when the candle-sam3 binary fails or produces unexpected output."""


class CandleSam3Adapter:
    """
    Wrap the candle-sam3 ``sam3`` example binary for image-batch inference.

    Parameters
    ----------
    binary_path : str or None
        Path to the compiled ``sam3`` binary. When ``None``, falls back to the
        ``SAM3_BINARY_PATH`` environment variable and finally to
        :data:`DEFAULT_SAM3_BINARY`.
    checkpoint_path : str or None
        Path to ``sam3.pt`` weights. Passed to ``--checkpoint``.
    tokenizer_path : str or None
        Path to ``tokenizer.json``. Required by candle-sam3 only when any job
        in a batch carries a text prompt.
    cpu : bool
        When True, force ``--cpu`` on the underlying binary.
    extra_args : sequence of str or None
        Extra CLI arguments appended verbatim to every invocation.
    """

    def __init__(
        self,
        binary_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        cpu: bool = False,
        extra_args: Optional[Sequence[str]] = None,
    ):
        resolved = binary_path or os.getenv(SAM3_BINARY_ENV) or DEFAULT_SAM3_BINARY
        self.binary_path = Path(resolved)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else None
        self.cpu = bool(cpu)
        self.extra_args = list(extra_args) if extra_args else []

    def is_available(self) -> bool:
        """Return True when the configured binary exists and is executable."""
        try:
            return self.binary_path.is_file() and os.access(self.binary_path, os.X_OK)
        except OSError:
            return False

    @staticmethod
    def normalize_xy(x: float, y: float, width: int, height: int) -> tuple[float, float]:
        """
        Convert pixel coordinates to normalized [0, 1] coordinates and clamp.

        Parameters
        ----------
        x, y : float
            Pixel coordinates (x = column, y = row).
        width, height : int
            Image dimensions in pixels.

        Returns
        -------
        tuple[float, float]
            ``(x_norm, y_norm)`` clamped to ``[0.0, 1.0]``.

        Raises
        ------
        ValueError
            If ``width`` or ``height`` is non-positive.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Image dimensions must be positive (got {width}x{height})")
        return (
            max(0.0, min(1.0, float(x) / float(width))),
            max(0.0, min(1.0, float(y) / float(height))),
        )

    @staticmethod
    def build_batch_manifest(jobs: Iterable[CandleSam3Job]) -> dict:
        """
        Serialize an iterable of jobs into the JSON structure expected by
        candle-sam3's ``--batch-manifest`` flag.

        Boxes and points emit empty arrays when absent rather than being
        omitted, mirroring the upstream example manifest.
        """
        return {
            "jobs": [
                {
                    "name": job.name,
                    "image": str(job.image),
                    **({"prompt": job.prompt} if job.prompt else {}),
                    "points": [
                        {"x": p.x, "y": p.y, "label": p.label} for p in job.points
                    ],
                    "boxes": [
                        {"cx": b.cx, "cy": b.cy, "w": b.w, "h": b.h, "label": b.label}
                        for b in job.boxes
                    ],
                }
                for job in jobs
            ]
        }

    @staticmethod
    def load_mask_png(path: Path) -> np.ndarray:
        """
        Load a mask PNG written by candle-sam3 as a binary ``uint8`` array
        (0 or 255). Tolerates grayscale or RGBA inputs by collapsing to L and
        thresholding at 127.
        """
        with Image.open(path) as img:
            arr = np.asarray(img.convert("L"))
        return (arr > 127).astype(np.uint8) * 255

    def predict_image_batch(
        self,
        jobs: Sequence[CandleSam3Job],
        output_dir: Path,
        *,
        mask_filename: str = DEFAULT_MASK_FILENAME,
    ) -> list[np.ndarray]:
        """
        Run a batch of image-prediction jobs through candle-sam3 and return
        the masks in input order.

        Parameters
        ----------
        jobs : sequence of CandleSam3Job
            Jobs to submit. Each job's ``name`` must be unique and filesystem-
            safe; it is used as the per-job subdirectory under ``output_dir``.
        output_dir : pathlib.Path
            Directory to write the manifest and receive job outputs. Created
            if missing.
        mask_filename : str
            Filename within each job subdirectory to load as the binary mask.
            Defaults to ``mask.png``.

        Returns
        -------
        list of numpy.ndarray
            One ``uint8`` mask array per job, in the order of ``jobs``.

        Raises
        ------
        CandleSam3AdapterError
            If a job's expected mask file is missing after a successful run.
        """
        if not jobs:
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(self.build_batch_manifest(jobs)))

        cmd = self._base_cmd() + [
            "--batch-manifest", str(manifest_path),
            "--output-dir", str(output_dir),
        ]
        self._run_binary(cmd)

        masks: list[np.ndarray] = []
        for job in jobs:
            mask_path = output_dir / job.name / mask_filename
            if not mask_path.is_file():
                raise CandleSam3AdapterError(
                    f"candle-sam3 produced no mask for job '{job.name}' at {mask_path}"
                )
            masks.append(self.load_mask_png(mask_path))
        return masks

    def _base_cmd(self) -> list[str]:
        """Build the base ``sam3`` invocation independent of subcommand args."""
        cmd: list[str] = [str(self.binary_path)]
        if self.checkpoint_path:
            cmd += ["--checkpoint", str(self.checkpoint_path)]
        if self.tokenizer_path:
            cmd += ["--tokenizer", str(self.tokenizer_path)]
        if self.cpu:
            cmd.append("--cpu")
        cmd += self.extra_args
        return cmd

    def _run_binary(self, cmd: Sequence[str]) -> subprocess.CompletedProcess:
        """
        Execute ``sam3`` and raise on non-zero exit. The single subprocess
        call site so tests can monkeypatch this method.
        """
        try:
            return subprocess.run(
                list(cmd), check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise CandleSam3AdapterError(
                f"candle-sam3 binary failed (exit {exc.returncode}): {stderr}"
            ) from exc
