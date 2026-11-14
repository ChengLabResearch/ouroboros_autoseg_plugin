use std::path::PathBuf;

use super::{write_annotation_overlay_stack, write_mask_stack};
use crate::{
    imaging::annotations::{fill_prompt_gaps, AnnotationOrigin, AnnotationPoint},
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

#[tokio::test]
async fn write_annotation_overlay_stack_preserves_source_masks_and_marks_points() {
    let root = unique_temp_dir();
    let output_path = root.join("mask-overlay.tif");
    let masks = vec![
        FrameMask {
            width: 5,
            height: 5,
            pixels: vec![0; 25],
        },
        FrameMask {
            width: 5,
            height: 5,
            pixels: vec![0; 25],
        },
    ];
    let annotations = vec![AnnotationPoint {
        origin: AnnotationOrigin::Metadata,
        x: 2.0,
        y: 2.0,
        z: 0.0,
    }];

    write_annotation_overlay_stack(&output_path, &masks, &annotations, 127)
        .await
        .expect("write overlay stack");

    assert!(masks[0].pixels.iter().all(|pixel| *pixel == 0));
    let frames = read_volume_frames(&output_path)
        .await
        .expect("read overlay stack");
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].pixels[2 + 2 * 5], 127);
    assert_eq!(frames[1].pixels[2 + 2 * 5], 127);
}

#[tokio::test]
async fn write_annotation_overlay_stack_marks_only_added_prompt_frames_with_hollow_diamonds() {
    let root = tempfile::tempdir().expect("overlay fixture");
    let output_path = root.path().join("gap-overlay.tif");
    let masks = vec![
        FrameMask {
            width: 9,
            height: 9,
            pixels: vec![0; 81]
        };
        5
    ];
    let mut annotations = vec![
        AnnotationPoint {
            x: 4.0,
            y: 4.0,
            z: 0.0,
            origin: AnnotationOrigin::Metadata,
        },
        AnnotationPoint {
            x: 4.0,
            y: 4.0,
            z: 4.0,
            origin: AnnotationOrigin::Metadata,
        },
    ];
    assert_eq!(fill_prompt_gaps(&mut annotations, masks.len(), Some(2)), 1);

    write_annotation_overlay_stack(&output_path, &masks, &annotations, 127)
        .await
        .expect("write overlay");

    let frames = read_volume_frames(&output_path)
        .await
        .expect("read overlay");
    for (index, frame) in frames.iter().enumerate() {
        if index == 2 {
            assert_eq!(frame.pixels[4 * 9 + 4], 0, "gap marker is hollow");
            assert_eq!(frame.pixels[9 + 4], 127, "diamond top");
            assert_eq!(frame.pixels[4 * 9 + 7], 127, "diamond right");
        } else {
            assert_eq!(
                frame.pixels[4 * 9 + 4],
                127,
                "existing overlay stars remain"
            );
            assert_eq!(frame.pixels[9 + 4], 0);
            assert_eq!(frame.pixels[4 * 9 + 7], 0);
        }
    }
    assert!(masks
        .iter()
        .all(|mask| mask.pixels.iter().all(|pixel| *pixel == 0)));
}
