use std::path::PathBuf;

use super::{stage_input_frames, PreparedFrame, PreparedFrameEncoding};
use crate::imaging::tiff_io::{inspect_volume, read_volume_frames, ImageFrame};
use uuid::Uuid;

fn unique_temp_dir() -> PathBuf {
    let path = std::env::temp_dir().join(format!("ouroboros-rust-phase3-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&path).expect("create temp dir");
    path
}

#[tokio::test]
async fn stage_input_frames_writes_rgb_tiff_frames() {
    let root = unique_temp_dir();
    let target = root.join("frames").join("0");
    stage_input_frames(&[PreparedFrame {
        frame: ImageFrame {
            width: 2,
            height: 1,
            channels: 1,
            pixels: vec![0, 255],
        },
        target: target.clone(),
        encoding: PreparedFrameEncoding::Tiff,
    }])
    .await
    .expect("stage TIFF frame");

    let staged_path = target.with_extension("tif");
    let volume = inspect_volume(&staged_path)
        .await
        .expect("inspect staged tif");
    assert_eq!(volume.geometry.frames, 1);
    let frames = read_volume_frames(&staged_path)
        .await
        .expect("read staged tif");
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].channels, 3);
    assert_eq!(frames[0].pixels, vec![0, 0, 0, 255, 255, 255]);
}

#[tokio::test]
async fn stage_input_frames_writes_jpeg_frames() {
    let root = unique_temp_dir();
    let target = root.join("frames").join("0");
    stage_input_frames(&[PreparedFrame {
        frame: ImageFrame {
            width: 2,
            height: 1,
            channels: 1,
            pixels: vec![0, 255],
        },
        target: target.clone(),
        encoding: PreparedFrameEncoding::Jpeg,
    }])
    .await
    .expect("stage JPEG frame");

    let staged_path = target.with_extension("jpg");
    let bytes = std::fs::read(&staged_path).expect("read staged jpeg");
    assert!(bytes.starts_with(&[0xFF, 0xD8]));
}
