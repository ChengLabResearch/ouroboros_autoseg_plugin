use super::*;
use crate::imaging::annotations::AnnotationOrigin;

fn frame(width: usize, height: usize, foreground: &[(usize, usize)]) -> FrameMask {
    let mut pixels = vec![0; width * height];
    for &(x, y) in foreground {
        pixels[y * width + x] = 255;
    }
    FrameMask {
        width,
        height,
        pixels,
    }
}

fn trace(x: f32, y: f32) -> [AnnotationPoint; 1] {
    [AnnotationPoint {
        x,
        y,
        z: 0.0,
        origin: AnnotationOrigin::Metadata,
    }]
}

fn rectangle(left: usize, top: usize, width: usize, height: usize) -> Vec<(usize, usize)> {
    (top..top + height)
        .flat_map(|y| (left..left + width).map(move |x| (x, y)))
        .collect()
}

#[test]
fn accepted_frames_share_eight_connected_trace_selection_and_preserve_values() {
    let mut masks = vec![frame(7, 7, &[(2, 2), (3, 3), (4, 4), (0, 6)])];
    masks[0].pixels[2 * 7 + 2] = 1;
    let report =
        repair_inconsistent_frames(&mut masks, &trace(3.0, 3.0), &FrameRepairOptions::default())
            .unwrap();
    assert_eq!(report.frames[0].repair, FrameRepairAction::Accepted);
    assert!(report.frames[0].reasons.is_empty());
    assert_eq!(masks[0].pixels[2 * 7 + 2], 1);
    assert_eq!(masks[0].pixels[4 * 7 + 4], 255);
    assert_eq!(
        masks[0].pixels[6 * 7],
        0,
        "unselected edge region cannot flag the frame"
    );
    assert!(!report.is_repaired(0));
}

#[test]
fn comparisons_use_previous_raw_selected_component_even_when_it_was_flagged() {
    let small = frame(9, 9, &rectangle(3, 3, 2, 2));
    let large = frame(9, 9, &rectangle(1, 1, 6, 6));
    let mut masks = vec![small, large.clone(), large.clone()];
    let report =
        repair_inconsistent_frames(&mut masks, &trace(3.0, 3.0), &FrameRepairOptions::default())
            .unwrap();
    assert_eq!(report.frames[0].repair, FrameRepairAction::Accepted);
    assert_eq!(
        report.frames[1].reasons,
        vec![FrameFlagReason::AreaJump, FrameFlagReason::LowIou]
    );
    assert_eq!(report.frames[1].repair, FrameRepairAction::Interpolated);
    assert_eq!(report.frames[2].repair, FrameRepairAction::Accepted);
    assert!(
        report.frames[2].reasons.is_empty(),
        "frame2 agrees with raw frame1, not the preceding accepted or repaired frame"
    );
    assert_eq!(masks[2].pixels, large.pixels);
}

#[test]
fn area_ratio_and_iou_threshold_equalities_are_accepted() {
    let originals = vec![
        frame(8, 7, &rectangle(2, 2, 2, 2)),
        frame(8, 7, &rectangle(2, 2, 3, 2)),
    ];
    for (max_area_ratio, min_iou, expected) in [
        (1.5, 2.0 / 3.0, vec![]),
        (1.49, 2.0 / 3.0, vec![FrameFlagReason::AreaJump]),
        (1.5, 0.67, vec![FrameFlagReason::LowIou]),
    ] {
        let mut masks = originals.clone();
        let options = FrameRepairOptions {
            max_area_ratio,
            min_iou,
            keep_unrepaired: false,
        };
        let report = repair_inconsistent_frames(&mut masks, &trace(2.0, 2.0), &options).unwrap();
        assert_eq!(report.frames[1].reasons, expected);
    }
}

#[test]
fn contact_with_each_frame_edge_flags_the_selected_component() {
    for (x, y) in [(0, 3), (6, 3), (3, 0), (3, 6)] {
        let mut masks = vec![frame(7, 7, &[(x, y)])];
        let original = masks[0].pixels.clone();
        let report = repair_inconsistent_frames(
            &mut masks,
            &trace(x as f32, y as f32),
            &FrameRepairOptions::default(),
        )
        .unwrap();
        assert_eq!(report.frames[0].reasons, vec![FrameFlagReason::EdgeContact]);
        assert_eq!(report.frames[0].repair, FrameRepairAction::Unrepaired);
        assert_eq!(masks[0].pixels, original);
    }
}

#[test]
fn missing_trace_and_wholly_flagged_stacks_remain_unrepaired() {
    let original = frame(7, 7, &rectangle(2, 2, 2, 2));
    let mut missing = vec![original; 3];
    let report =
        repair_inconsistent_frames(&mut missing, &[], &FrameRepairOptions::default()).unwrap();
    assert!(missing
        .iter()
        .all(|mask| mask.pixels.iter().all(|&pixel| pixel == 0)));
    assert!(report.frames.iter().all(|frame| frame
        .reasons
        .contains(&FrameFlagReason::MissingTrace)
        && frame.reasons.contains(&FrameFlagReason::EmptyComponent)
        && frame.repair == FrameRepairAction::Unrepaired));
    assert_eq!(
        report.frames[0].reasons,
        vec![
            FrameFlagReason::MissingTrace,
            FrameFlagReason::EmptyComponent
        ]
    );
    assert!(
        report.frames[1].reasons.contains(&FrameFlagReason::LowIou),
        "empty union has zero IoU"
    );
    assert!(
        !report.frames[1]
            .reasons
            .contains(&FrameFlagReason::AreaJump),
        "two empty areas have no jump"
    );
    assert_eq!(report.summary().unrepaired_frames, 3);
    assert_eq!(report.summary().repaired_frames, 0);

    let mut flagged = vec![frame(7, 7, &[(0, 0), (1, 1), (5, 5)]); 3];
    let report = repair_inconsistent_frames(
        &mut flagged,
        &trace(0.0, 0.0),
        &FrameRepairOptions::default(),
    )
    .unwrap();
    assert!(report
        .frames
        .iter()
        .all(|frame| frame.repair == FrameRepairAction::Unrepaired));
    for mask in flagged {
        assert_eq!(mask.pixels[0], 255);
        assert_eq!(mask.pixels[8], 255);
        assert_eq!(
            mask.pixels[40], 0,
            "even unrepaired output contains only the selected component"
        );
    }
}

#[test]
fn missing_foreground_is_empty_and_repairable_when_trace_and_anchor_exist() {
    let anchor = frame(32, 24, &rectangle(2, 2, 2, 2));
    let mut masks = vec![
        anchor.clone(),
        frame(32, 24, &[(30, 20)]),
        anchor.clone(),
        anchor.clone(),
    ];
    let report =
        repair_inconsistent_frames(&mut masks, &trace(2.0, 2.0), &FrameRepairOptions::default())
            .unwrap();
    assert!(report.frames[1]
        .reasons
        .contains(&FrameFlagReason::EmptyComponent));
    assert!(!report.frames[1]
        .reasons
        .contains(&FrameFlagReason::MissingTrace));
    assert!(report.frames[1]
        .reasons
        .contains(&FrameFlagReason::AreaJump));
    assert!(
        report.frames[2]
            .reasons
            .contains(&FrameFlagReason::AreaJump),
        "comparison is with the raw empty predecessor"
    );
    assert!(report.is_repaired(1));
    assert!(report.is_repaired(2));
    assert_eq!(masks[1].pixels, anchor.pixels);
    assert_eq!(masks[2].pixels, anchor.pixels);
}

fn brute_signed(mask: &FrameMask) -> Vec<f64> {
    mask.pixels
        .iter()
        .enumerate()
        .map(|(index, &pixel)| {
            let squared = mask
                .pixels
                .iter()
                .enumerate()
                .filter(|(_, other)| (**other != 0) != (pixel != 0))
                .map(|(other, _)| {
                    ((index % mask.width).abs_diff(other % mask.width).pow(2)
                        + (index / mask.width).abs_diff(other / mask.width).pow(2))
                        as f64
                })
                .fold(f64::INFINITY, f64::min);
            if pixel != 0 {
                squared.sqrt()
            } else {
                -squared.sqrt()
            }
        })
        .collect()
}

#[test]
fn leading_internal_and_trailing_runs_use_original_accepted_endpoints() {
    let mut left = frame(9, 7, &rectangle(2, 2, 2, 3));
    left.pixels[2 * 9 + 2] = 1;
    let right = frame(9, 7, &rectangle(4, 2, 3, 3));
    let bad = FrameMask {
        width: 9,
        height: 7,
        pixels: vec![255; 9 * 7],
    };
    let mut masks = vec![
        bad.clone(),
        bad.clone(),
        left.clone(),
        bad.clone(),
        bad.clone(),
        bad.clone(),
        right.clone(),
        bad.clone(),
        bad,
    ];
    let options = FrameRepairOptions {
        max_area_ratio: 100.0,
        min_iou: 0.0,
        keep_unrepaired: true,
    };
    let report = repair_inconsistent_frames(&mut masks, &trace(4.0, 3.0), &options).unwrap();
    assert_eq!(report.options, options);
    assert_eq!(
        report
            .frames
            .iter()
            .map(|frame| frame.repair)
            .collect::<Vec<_>>(),
        vec![
            FrameRepairAction::CopiedNext,
            FrameRepairAction::CopiedNext,
            FrameRepairAction::Accepted,
            FrameRepairAction::Interpolated,
            FrameRepairAction::Interpolated,
            FrameRepairAction::Interpolated,
            FrameRepairAction::Accepted,
            FrameRepairAction::CopiedPrevious,
            FrameRepairAction::CopiedPrevious,
        ]
    );
    for index in [0, 1] {
        assert_eq!(
            masks[index].pixels, left.pixels,
            "one-sided repair copies the accepted endpoint exactly"
        );
    }
    for index in [7, 8] {
        assert_eq!(masks[index].pixels, right.pixels);
    }
    assert_eq!(
        masks[2].pixels, left.pixels,
        "accepted nonzero values remain unchanged"
    );
    assert_eq!(masks[6].pixels, right.pixels);
    let left_distance = brute_signed(&left);
    let right_distance = brute_signed(&right);
    for index in 3..6 {
        let fraction = (index - 2) as f64 / 4.0;
        let expected = left_distance
            .iter()
            .zip(&right_distance)
            .map(|(&left, &right)| {
                if (1.0 - fraction) * left + fraction * right > 0.0 {
                    255
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(
            masks[index].pixels, expected,
            "frame{index} uses Euclidean distances and its original frame position"
        );
    }
    assert_eq!(
        report.summary(),
        FrameRepairSummary {
            total_frames: 9,
            accepted_frames: 2,
            flagged_frames: 7,
            repaired_frames: 7,
            unrepaired_frames: 0,
            runs: vec![
                FrameRepairRun {
                    start_frame: 0,
                    end_frame: 1,
                    flagged_frames: 2,
                    repaired_frames: 2
                },
                FrameRepairRun {
                    start_frame: 3,
                    end_frame: 5,
                    flagged_frames: 3,
                    repaired_frames: 3
                },
                FrameRepairRun {
                    start_frame: 7,
                    end_frame: 8,
                    flagged_frames: 2,
                    repaired_frames: 2
                },
            ],
        }
    );
    let serialized = serde_json::to_value(&report).unwrap();
    assert_eq!(serialized["frames"][0]["repair"], "copied_next");
    assert_eq!(
        serialized["frames"][0]["reasons"],
        serde_json::json!(["edge_contact"])
    );
}

#[test]
fn repair_rejects_invalid_options_and_incompatible_frame_geometry_before_mutation() {
    let original = frame(7, 7, &[(0, 0), (4, 4)]);
    let mut masks = vec![original.clone()];
    let options = FrameRepairOptions {
        min_iou: f64::NAN,
        ..FrameRepairOptions::default()
    };
    assert!(repair_inconsistent_frames(&mut masks, &trace(0.0, 0.0), &options).is_err());
    assert_eq!(masks[0].pixels, original.pixels);
    for other in [
        frame(8, 7, &[]),
        FrameMask {
            width: 7,
            height: 7,
            pixels: vec![0; 3],
        },
    ] {
        let mut masks = vec![original.clone(), other];
        assert!(repair_inconsistent_frames(
            &mut masks,
            &trace(0.0, 0.0),
            &FrameRepairOptions::default()
        )
        .is_err());
        assert_eq!(masks[0].pixels, original.pixels);
    }
}
