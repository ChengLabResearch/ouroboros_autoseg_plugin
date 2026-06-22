use std::path::PathBuf;

use super::{inspect_volume, read_volume_frames, write_tiff_pages, WritableTiffPage};
use uuid::Uuid;

fn unique_temp_dir() -> PathBuf {
    let path = std::env::temp_dir().join(format!("ouroboros-rust-phase3-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&path).expect("create temp dir");
    path
}

#[tokio::test]
async fn inspect_volume_reads_geometry_and_metadata_from_single_file() {
    let root = unique_temp_dir();
    let stack_path = root.join("stack.tif");
    write_tiff_pages(
        &stack_path,
        &[
            WritableTiffPage {
                width: 2,
                height: 2,
                channels: 1,
                pixels: vec![0, 1, 2, 3],
                description: Some(
                    r#"{"annotation_points": [[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]]}"#.to_string(),
                ),
            },
            WritableTiffPage {
                width: 2,
                height: 2,
                channels: 1,
                pixels: vec![4, 5, 6, 7],
                description: None,
            },
        ],
    )
    .expect("write stack");

    let volume = inspect_volume(&stack_path).await.expect("inspect volume");
    assert_eq!(volume.geometry.frames, 2);
    assert_eq!(volume.geometry.height, 2);
    assert_eq!(volume.geometry.width, 2);
    let points = volume.annotation_points.expect("metadata annotations");
    assert_eq!(points.len(), 2);
    assert_eq!(points[0].x, 1.0);
    assert_eq!(points[1].z, 1.0);
}

#[tokio::test]
async fn inspect_volume_rejects_malformed_annotation_metadata_from_single_file() {
    let root = unique_temp_dir();
    let stack_path = root.join("stack.tif");
    write_tiff_pages(
        &stack_path,
        &[WritableTiffPage {
            width: 2,
            height: 2,
            channels: 1,
            pixels: vec![0, 1, 2, 3],
            description: Some(r#"{"annotation_points": [[1.0, 2.0]]}"#.to_string()),
        }],
    )
    .expect("write stack");

    let err = inspect_volume(&stack_path)
        .await
        .expect_err("malformed metadata should be rejected");
    assert!(
        err.to_string()
            .contains("Malformed annotation_points metadata"),
        "{err}"
    );
}

#[tokio::test]
async fn inspect_volume_rejects_malformed_annotation_metadata_from_directory() {
    let root = unique_temp_dir();
    let frame_dir = root.join("frames");
    std::fs::create_dir_all(&frame_dir).expect("create frame dir");
    write_tiff_pages(
        &frame_dir.join("00.tif"),
        &[WritableTiffPage {
            width: 2,
            height: 2,
            channels: 1,
            pixels: vec![0, 1, 2, 3],
            description: Some(r#"{"annotation_points": "center"}"#.to_string()),
        }],
    )
    .expect("write malformed frame");
    write_tiff_pages(
        &frame_dir.join("01.tif"),
        &[WritableTiffPage {
            width: 2,
            height: 2,
            channels: 1,
            pixels: vec![4, 5, 6, 7],
            description: None,
        }],
    )
    .expect("write frame");

    let err = inspect_volume(&frame_dir)
        .await
        .expect_err("malformed metadata should be rejected");
    assert!(
        err.to_string()
            .contains("Malformed annotation_points metadata"),
        "{err}"
    );
}

#[tokio::test]
async fn read_volume_frames_round_trips_written_tiff_pages() {
    let root = unique_temp_dir();
    let stack_path = root.join("stack.tif");
    write_tiff_pages(
        &stack_path,
        &[
            WritableTiffPage {
                width: 2,
                height: 2,
                channels: 1,
                pixels: vec![10, 11, 12, 13],
                description: None,
            },
            WritableTiffPage {
                width: 2,
                height: 2,
                channels: 3,
                pixels: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                description: None,
            },
        ],
    )
    .expect("write stack");

    let frames = read_volume_frames(&stack_path).await.expect("read frames");
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].channels, 1);
    assert_eq!(frames[0].pixels, vec![10, 11, 12, 13]);
    assert_eq!(frames[1].channels, 3);
    assert_eq!(
        frames[1].pixels,
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    );
}

#[tokio::test]
async fn inspect_volume_supports_directory_inputs() {
    let root = unique_temp_dir();
    let frame_dir = root.join("frames");
    std::fs::create_dir_all(&frame_dir).expect("create frame dir");
    for (index, pixels) in [vec![0, 1, 2, 3], vec![4, 5, 6, 7]].into_iter().enumerate() {
        write_tiff_pages(
            &frame_dir.join(format!("{index:02}.tif")),
            &[WritableTiffPage {
                width: 2,
                height: 2,
                channels: 1,
                pixels,
                description: if index == 0 {
                    Some(r#"{"annotation_points": [[2.0, 1.0, 0.0]]}"#.to_string())
                } else {
                    None
                },
            }],
        )
        .expect("write frame");
    }

    let volume = inspect_volume(&frame_dir)
        .await
        .expect("inspect volume dir");
    assert_eq!(volume.geometry.frames, 2);
    assert_eq!(volume.geometry.height, 2);
    assert_eq!(volume.geometry.width, 2);
    assert_eq!(
        volume.annotation_points.expect("annotation points").len(),
        1
    );

    let frames = read_volume_frames(&frame_dir)
        .await
        .expect("read dir frames");
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[1].pixels, vec![4, 5, 6, 7]);
}
