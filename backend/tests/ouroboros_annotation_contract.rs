use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use ouroboros_autoseg_plugin_backend::imaging::{
    annotations::{annotation_samples_for_video, interpolated_point_for_frame},
    tiff_io::inspect_volume,
};
use uuid::Uuid;

const FIXTURE_GENERATOR: &str = r#"
import sys
import types
from pathlib import Path

import numpy as np
import tifffile

if "cloudvolume" not in sys.modules:
    cloudvolume = types.ModuleType("cloudvolume")
    cloudvolume.CloudVolume = object
    cloudvolume.VolumeCutout = object
    cloudvolume.Bbox = object
    sys.modules["cloudvolume"] = cloudvolume

from ouroboros.pipeline.slice_parallel_pipeline import build_straightened_tiff_metadata


class FakeVolumeCache:
    def get_resolution_um(self):
        return np.array([0.5, 0.25, 1.5], dtype=np.float32)


annotation_points = np.array(
    [
        [4.5, 8.25, 0.0],
        [6.75, 9.5, 2.0],
    ],
    dtype=np.float32,
)


def metadata_kwargs():
    return build_straightened_tiff_metadata(
        volume_cache=FakeVolumeCache(),
        has_color_channels=False,
        num_color_channels=None,
        annotation_points=annotation_points,
    )


output_dir = Path(sys.argv[1])
output_dir.mkdir(parents=True, exist_ok=True)

stack_path = output_dir / "straightened.tif"
stack = tifffile.memmap(
    stack_path,
    shape=(3, 4, 5),
    dtype=np.uint8,
    **metadata_kwargs(),
)
stack[:] = np.arange(stack.size, dtype=np.uint8).reshape(stack.shape)
stack.flush()
del stack

frame_dir = output_dir / "straightened"
frame_dir.mkdir()
metadata = metadata_kwargs()
for frame_index in range(3):
    tifffile.imwrite(
        frame_dir / f"{frame_index:02}.tif",
        np.full((4, 5), frame_index, dtype=np.uint8),
        **metadata,
    )
"#;

#[tokio::test]
async fn consumes_annotation_points_written_by_ouroboros_outputs() {
    let Some(ouroboros_root) = ouroboros_root() else {
        eprintln!("skipping: unable to resolve the sibling Ouroboros checkout");
        return;
    };
    if !ouroboros_root.join("python/ouroboros").is_dir() {
        eprintln!(
            "skipping: Ouroboros checkout not found at {}",
            ouroboros_root.display()
        );
        return;
    }

    let fixture_dir = unique_temp_dir();
    if !generate_ouroboros_fixtures(&ouroboros_root, &fixture_dir) {
        return;
    }

    assert_ouroboros_annotation_contract(&fixture_dir.join("straightened.tif")).await;
    assert_ouroboros_annotation_contract(&fixture_dir.join("straightened")).await;
}

fn ouroboros_root() -> Option<PathBuf> {
    if let Some(path) = env::var_os("OUROBOROS_REPO_DIR") {
        return Some(PathBuf::from(path));
    }

    let backend_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    Some(backend_dir.parent()?.parent()?.join("ouroboros"))
}

fn unique_temp_dir() -> PathBuf {
    let path = env::temp_dir().join(format!(
        "ouroboros-autoseg-annotation-contract-{}",
        Uuid::new_v4()
    ));
    fs::create_dir_all(&path).expect("create temp dir");
    path
}

fn generate_ouroboros_fixtures(ouroboros_root: &Path, fixture_dir: &Path) -> bool {
    let script_path = fixture_dir.join("generate_ouroboros_annotation_fixtures.py");
    fs::write(&script_path, FIXTURE_GENERATOR).expect("write fixture generator");

    let pythonpath = ouroboros_root.join("python");
    let configured_python = env::var_os("OUROBOROS_PYTHON").map(PathBuf::from);
    let candidates = configured_python
        .clone()
        .map(|python| vec![python])
        .unwrap_or_else(|| vec![PathBuf::from("python"), PathBuf::from("python3")]);

    let mut failures = Vec::new();
    for python in candidates {
        match Command::new(&python)
            .arg(&script_path)
            .arg(fixture_dir)
            .env("PYTHONPATH", &pythonpath)
            .output()
        {
            Ok(output) if output.status.success() => return true,
            Ok(output) => failures.push(format!(
                "{} exited with {}\nstdout:\n{}\nstderr:\n{}",
                python.display(),
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )),
            Err(err) => failures.push(format!("{} failed to start: {err}", python.display())),
        }
    }

    let message = failures.join("\n\n");
    if configured_python.is_some() {
        panic!("OUROBOROS_PYTHON could not generate Ouroboros fixtures:\n{message}");
    }
    eprintln!(
        "skipping: no default Python could generate Ouroboros fixtures; set OUROBOROS_PYTHON. Attempts:\n{message}"
    );
    false
}

async fn assert_ouroboros_annotation_contract(path: &Path) {
    let volume = inspect_volume(path)
        .await
        .expect("rs plugin reads the Ouroboros TIFF output");
    assert_eq!(volume.geometry.frames, 3);
    assert_eq!(volume.geometry.height, 4);
    assert_eq!(volume.geometry.width, 5);

    let points = volume
        .annotation_points
        .expect("rs plugin reads Ouroboros annotation_points metadata");
    assert_eq!(points.len(), 2);
    assert_close(points[0].x, 4.5);
    assert_close(points[0].y, 8.25);
    assert_close(points[0].z, 0.0);
    assert_close(points[1].x, 6.75);
    assert_close(points[1].y, 9.5);
    assert_close(points[1].z, 2.0);

    let samples = annotation_samples_for_video(&points, volume.geometry.frames);
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].frame_index, 0);
    assert_close(samples[0].points[0].x, 4.5);
    assert_close(samples[0].points[0].y, 8.25);
    assert_eq!(samples[1].frame_index, 2);
    assert_close(samples[1].points[0].x, 6.75);
    assert_close(samples[1].points[0].y, 9.5);

    let interpolated = interpolated_point_for_frame(&points, 1)
        .expect("annotation module interpolates parsed metadata");
    assert_close(interpolated.x, 5.625);
    assert_close(interpolated.y, 8.875);
}

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 0.0001,
        "expected {expected}, got {actual}"
    );
}
