use std::path::PathBuf;

use super::{write_annotation_overlay_stack, write_mask_stack};
use crate::{
    imaging::annotations::{
        annotation_samples_for_video, fill_prompt_gaps, AnnotationOrigin, AnnotationPoint,
    },
    imaging::cleanup::{FrameFlagReason, FrameRepairAction, FrameRepairFrame, FrameRepairReport},
    imaging::tiff_io::{inspect_volume, read_volume_frames},
    inference::{
        image::{FrameMask, PointPrompt},
        video::VideoFramePrompt,
    },
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

    write_annotation_overlay_stack(
        &output_path,
        &masks,
        &annotation_samples_for_video(&annotations, masks.len()),
        127,
        None,
    )
    .await
    .expect("write overlay stack");

    assert!(masks[0].pixels.iter().all(|pixel| *pixel == 0));
    let frames = read_volume_frames(&output_path)
        .await
        .expect("read overlay stack");
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].pixels[2 + 2 * 5], 127);
    assert_eq!(frames[1].pixels, masks[1].pixels);
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

    write_annotation_overlay_stack(
        &output_path,
        &masks,
        &annotation_samples_for_video(&annotations, masks.len()),
        127,
        None,
    )
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
        } else if index == 0 || index == 4 {
            assert_eq!(
                frame.pixels[4 * 9 + 4],
                127,
                "existing overlay stars remain"
            );
            assert_eq!(frame.pixels[9 + 4], 0);
            assert_eq!(frame.pixels[4 * 9 + 7], 0);
        } else {
            assert_eq!(frame.pixels, masks[index].pixels, "unprompted frame");
        }
    }
    assert!(masks
        .iter()
        .all(|mask| mask.pixels.iter().all(|pixel| *pixel == 0)));
}

#[tokio::test]
async fn overlay_uses_every_prepared_point_with_its_original_kind_and_frame() {
    let root = tempfile::tempdir().expect("overlay fixture");
    let masks = vec![
        FrameMask {
            width: 24,
            height: 24,
            pixels: vec![255; 24 * 24],
        };
        11
    ];
    let mut annotations = vec![
        AnnotationPoint {
            x: 4.2,
            y: 5.6,
            z: 3.2,
            origin: AnnotationOrigin::Metadata,
        },
        AnnotationPoint {
            x: 12.6,
            y: 5.2,
            z: 3.4,
            origin: AnnotationOrigin::Metadata,
        },
        AnnotationPoint {
            x: 5.2,
            y: 14.8,
            z: 7.0,
            origin: AnnotationOrigin::Fallback,
        },
    ];
    assert_eq!(fill_prompt_gaps(&mut annotations, masks.len(), Some(2)), 1);
    let mut prompts = annotation_samples_for_video(&annotations, masks.len());
    assert_eq!(
        prompts
            .iter()
            .map(|prompt| prompt.frame_index)
            .collect::<Vec<_>>(),
        [3, 5, 7]
    );
    assert_eq!(
        prompts[0].points.len(),
        2,
        "same-frame positives are retained"
    );
    for prompt in &mut prompts {
        prompt.points.push(PointPrompt::negative(18.6, 18.4));
    }

    for (positive_intensity, negative_intensity) in [(127, 64), (64, 63)] {
        let output_path = root
            .path()
            .join(format!("overlay-{positive_intensity}.tif"));
        write_annotation_overlay_stack(&output_path, &masks, &prompts, positive_intensity, None)
            .await
            .expect("write overlay");
        let frames = read_volume_frames(&output_path)
            .await
            .expect("read overlay");
        for (index, frame) in frames.iter().enumerate() {
            let positive_pixels = frame
                .pixels
                .iter()
                .filter(|&&p| p == positive_intensity)
                .count();
            let negative_pixels = frame
                .pixels
                .iter()
                .filter(|&&p| p == negative_intensity)
                .count();
            match index {
                3 => {
                    assert_eq!(positive_pixels, 26, "two metadata stars, 13 pixels each");
                    assert_eq!(frame.pixels[6 * 24 + 4], positive_intensity);
                    assert_eq!(frame.pixels[5 * 24 + 13], positive_intensity);
                }
                5 => {
                    assert_eq!(positive_pixels, 12, "gap-fill hollow diamond");
                    assert_eq!(frame.pixels[9 * 24 + 9], 255, "diamond centre is untouched");
                    assert_eq!(frame.pixels[6 * 24 + 9], positive_intensity, "diamond top");
                }
                7 => {
                    assert_eq!(positive_pixels, 13, "fallback star");
                    assert_eq!(frame.pixels[15 * 24 + 5], positive_intensity);
                }
                _ => {
                    assert_eq!(
                        frame.pixels, masks[index].pixels,
                        "unprompted frame {index}"
                    );
                    continue;
                }
            }
            assert_eq!(negative_pixels, 5, "negative cross");
            assert_eq!(frame.pixels[18 * 24 + 19], negative_intensity);
        }
    }
    assert!(masks
        .iter()
        .all(|mask| mask.pixels.iter().all(|&pixel| pixel == 255)));
}

#[tokio::test]
async fn overlay_without_prepared_prompts_is_byte_identical_to_primary_stack() {
    let root = tempfile::tempdir().expect("overlay fixture");
    let primary_path = root.path().join("primary.tif");
    let overlay_path = root.path().join("overlay.tif");
    let masks = vec![
        FrameMask {
            width: 2,
            height: 2,
            pixels: vec![0, 255, 255, 0],
        },
        FrameMask {
            width: 2,
            height: 2,
            pixels: vec![255, 0, 0, 255],
        },
    ];
    write_mask_stack(&primary_path, &masks)
        .await
        .expect("write primary");
    write_annotation_overlay_stack(&overlay_path, &masks, &[], 127, None)
        .await
        .expect("write overlay without points");
    assert_eq!(
        std::fs::read(primary_path).unwrap(),
        std::fs::read(overlay_path).unwrap()
    );
}

#[tokio::test]
async fn repair_markers_follow_report_actions_without_requiring_frame_prompts() {
    let root = tempfile::tempdir().expect("overlay fixture");
    let masks = vec![
        FrameMask {
            width: 8,
            height: 8,
            pixels: vec![0; 64]
        };
        5
    ];
    let report = FrameRepairReport {
        frames: [
            (FrameRepairAction::Kept, vec![], false),
            (
                FrameRepairAction::Clipped,
                vec![FrameFlagReason::EdgeContact],
                false,
            ),
            (
                FrameRepairAction::Kept,
                vec![FrameFlagReason::EdgeContact],
                false,
            ),
            (
                FrameRepairAction::Kept,
                vec![FrameFlagReason::AreaJump],
                true,
            ),
            (
                FrameRepairAction::Replaced,
                vec![FrameFlagReason::EmptyComponent],
                false,
            ),
        ]
        .into_iter()
        .enumerate()
        .map(
            |(frame_index, (repair, reasons, anchor_recovered))| FrameRepairFrame {
                component_area: if frame_index == 4 { 0 } else { 9 },
                reasons,
                anchor_recovered,
                repair,
            },
        )
        .collect(),
    };
    for (positive_intensity, repair_intensity) in [(127, 192), (192, 191)] {
        let path = root.path().join(format!("repair-{positive_intensity}.tif"));
        write_annotation_overlay_stack(&path, &masks, &[], positive_intensity, Some(&report))
            .await
            .expect("repair overlay without prompts");
        let frames = read_volume_frames(&path).await.unwrap();
        for (index, frame) in frames.iter().enumerate() {
            if [1, 4].contains(&index) {
                assert_eq!(
                    frame
                        .pixels
                        .iter()
                        .filter(|&&pixel| pixel == repair_intensity)
                        .count(),
                    9
                );
                for coordinate in [0, 1, 2, 3, 4] {
                    assert_eq!(frame.pixels[coordinate * 8 + coordinate], repair_intensity);
                    assert_eq!(
                        frame.pixels[coordinate * 8 + (4 - coordinate)],
                        repair_intensity
                    );
                }
            } else {
                assert_eq!(
                    frame.pixels, masks[index].pixels,
                    "kept frame is unmarked, including recovered and still-flagged frames"
                );
            }
        }

        let prompts = vec![VideoFramePrompt {
            frame_index: 1,
            points: vec![PointPrompt::positive(2.0, 2.0, AnnotationOrigin::Metadata)],
        }];
        write_annotation_overlay_stack(&path, &masks, &prompts, positive_intensity, Some(&report))
            .await
            .expect("repair glyph over prompt");
        let frames = read_volume_frames(&path).await.unwrap();
        assert_eq!(
            frames[1].pixels[2 * 8 + 2],
            repair_intensity,
            "repair draws after prompt"
        );
        assert_eq!(
            frames[1].pixels[2], positive_intensity,
            "remaining positive star pixels"
        );
    }
    assert!(masks
        .iter()
        .all(|mask| mask.pixels.iter().all(|&pixel| pixel == 0)));
}

#[tokio::test]
async fn report_without_repairs_leaves_existing_prompt_overlay_byte_identical() {
    let root = tempfile::tempdir().expect("overlay fixture");
    let baseline_path = root.path().join("baseline.tif");
    let report_path = root.path().join("report.tif");
    let masks = vec![
        FrameMask {
            width: 8,
            height: 8,
            pixels: vec![255; 64]
        };
        2
    ];
    let prompts = vec![VideoFramePrompt {
        frame_index: 1,
        points: vec![
            PointPrompt::positive(4.0, 4.0, AnnotationOrigin::Metadata),
            PointPrompt::negative(1.0, 1.0),
        ],
    }];
    let report = FrameRepairReport {
        frames: vec![
            FrameRepairFrame {
                component_area: 64,
                reasons: vec![FrameFlagReason::LowIou],
                anchor_recovered: true,
                repair: FrameRepairAction::Kept,
            },
            FrameRepairFrame {
                component_area: 64,
                reasons: vec![FrameFlagReason::EdgeContact],
                anchor_recovered: false,
                repair: FrameRepairAction::Kept,
            },
        ],
    };
    write_annotation_overlay_stack(&baseline_path, &masks, &prompts, 127, None)
        .await
        .unwrap();
    write_annotation_overlay_stack(&report_path, &masks, &prompts, 127, Some(&report))
        .await
        .unwrap();
    assert_eq!(
        std::fs::read(baseline_path).unwrap(),
        std::fs::read(report_path).unwrap()
    );
}
