import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile as tf

_TEST_VOLUME_ROOT = tempfile.mkdtemp(prefix="ouroboros-test-volume-")
os.environ["VOLUME_MOUNT_PATH"] = _TEST_VOLUME_ROOT

from backend.app.main import (
    _annotation_point_for_frame,
    _annotation_samples_for_video,
    load_annotation_points,
)


class AnnotationPointTests(unittest.TestCase):
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

            points = load_annotation_points(tif_path)
            self.assertIsNotNone(points)
            self.assertEqual(points.shape, (2, 3))
            np.testing.assert_allclose(points, np.array(metadata["annotation_points"], dtype=np.float32))

    def test_load_annotation_points_from_directory_first_tiff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            metadata = {"annotation_points": [[1, 2, 3], [4, 5, 6]]}
            tf.imwrite(folder / "000.tif", np.zeros((4, 4), dtype=np.uint8), description=json.dumps(metadata))
            tf.imwrite(folder / "001.tif", np.zeros((4, 4), dtype=np.uint8), description=json.dumps({}))

            points = load_annotation_points(folder)
            self.assertIsNotNone(points)
            np.testing.assert_allclose(points, np.array(metadata["annotation_points"], dtype=np.float32))

    def test_load_annotation_points_missing_metadata_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tif_path = Path(tmpdir, "volume.tif")
            tf.imwrite(tif_path, np.zeros((8, 8), dtype=np.uint8), description=json.dumps({}))
            points = load_annotation_points(tif_path)
            self.assertIsNone(points)

    def test_annotation_samples_for_video_rounds_and_groups_z(self):
        annotation_points = np.array(
            [
                [5.0, 6.0, 0.49],   # rounds to frame 0
                [7.0, 8.0, 0.51],   # rounds to frame 1
                [9.0, 10.0, 1.4],   # rounds to frame 1
                [3.0, 4.0, 99.0],   # out of bounds for 5 frames
            ],
            dtype=np.float32,
        )

        samples = _annotation_samples_for_video(annotation_points, num_frames=5)

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

        point = _annotation_point_for_frame(annotation_points, frame_idx=5)
        np.testing.assert_allclose(point, np.array([[5.0, 10.0]], dtype=np.float32))

    def test_annotation_point_for_frame_clamps_outside_range(self):
        annotation_points = np.array(
            [
                [2.0, 4.0, 3.0],
                [10.0, 12.0, 8.0],
            ],
            dtype=np.float32,
        )

        before = _annotation_point_for_frame(annotation_points, frame_idx=0)
        after = _annotation_point_for_frame(annotation_points, frame_idx=50)

        np.testing.assert_allclose(before, np.array([[2.0, 4.0]], dtype=np.float32))
        np.testing.assert_allclose(after, np.array([[10.0, 12.0]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
