use std::path::PathBuf;

use super::write_mask_stack;
use crate::{
    imaging::tiff_io::{inspect_volume, read_volume_frames},
    inference::image::FrameMask,
};
use uuid::Uuid;

fn unique_temp_dir() -> PathBuf {
    let path = std::env::temp_dir().join(format!("ouroboros-rust-phase3-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&path).expect("create temp dir");
    path
}

#[tokio::test]
async fn write_mask_stack_creates_multipage_tiff() {
    let root = unique_temp_dir();
    let output_path = root.join("mask-stack.tif");
    let masks = vec![
        FrameMask {
            width: 2,
            height: 2,
            pixels: vec![0, 255, 1, 2],
        },
        FrameMask {
            width: 2,
            height: 2,
            pixels: vec![3, 4, 5, 6],
        },
    ];

    write_mask_stack(&output_path, &masks)
        .await
        .expect("write mask stack");

    let volume = inspect_volume(&output_path).await.expect("inspect output");
    assert_eq!(volume.geometry.frames, 2);
    let frames = read_volume_frames(&output_path)
        .await
        .expect("read mask stack");
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].channels, 1);
    assert_eq!(frames[0].pixels, masks[0].pixels);
    assert_eq!(frames[1].pixels, masks[1].pixels);
}

#[tokio::test]
async fn write_mask_stack_rejects_inconsistent_geometry() {
    let root = unique_temp_dir();
    let output_path = root.join("mask-stack.tif");
    let error = write_mask_stack(
        &output_path,
        &[
            FrameMask {
                width: 2,
                height: 2,
                pixels: vec![0, 1, 2, 3],
            },
            FrameMask {
                width: 3,
                height: 1,
                pixels: vec![0, 1, 2],
            },
        ],
    )
    .await
    .expect_err("inconsistent geometry should fail");

    assert_eq!(
        error.to_string(),
        "Mask stack frames must all share the same geometry"
    );
}
