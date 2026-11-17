use super::super::tests::mask as frame;
use super::*;
use crate::imaging::annotations::AnnotationOrigin;

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

fn record(component_area: usize, reasons: Vec<FrameFlagReason>) -> FrameRepairFrame {
    FrameRepairFrame {
        component_area,
        reasons,
        anchor_recovered: false,
        repair: FrameRepairAction::Kept,
    }
}

fn accepted_record(mask: &FrameMask) -> FrameRepairFrame {
    record(
        mask.pixels.iter().filter(|&&pixel| pixel != 0).count(),
        vec![],
    )
}

#[test]
fn accepted_frames_share_eight_connected_trace_selection_and_preserve_values() {
    let mut masks = vec![frame(7, 7, &[(2, 2), (3, 3), (4, 4), (0, 6)])];
    masks[0].pixels[2 * 7 + 2] = 1;
    let report =
        repair_inconsistent_frames(&mut masks, &trace(3.0, 3.0), &FrameRepairOptions::default())
            .unwrap();
    assert_eq!(report.frames[0].repair, FrameRepairAction::Kept);
    assert_eq!(report.frames[0].component_area, 3);
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
    assert!(report.frames[0].reasons.is_empty());
    assert_eq!(
        report.frames[1].reasons,
        vec![FrameFlagReason::AreaJump, FrameFlagReason::LowIou]
    );
    assert!(report.frames[1].is_flagged());
    assert!(
        report.frames[2].reasons.is_empty(),
        "frame2 agrees with raw frame1, not the preceding accepted frame"
    );
    assert_eq!(report.frames[2].component_area, 36);
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
fn valid_anchor_after_a_flood_is_recovered_without_changing_its_mask() {
    let mut valid = frame(31, 31, &rectangle(13, 13, 5, 5));
    valid.pixels[15 * 31 + 15] = 1;
    let flood = FrameMask {
        width: 31,
        height: 31,
        pixels: vec![255; 31 * 31],
    };
    let mut masks = vec![valid.clone(), flood, valid.clone(), valid.clone()];
    let report = repair_inconsistent_frames(
        &mut masks,
        &trace(15.0, 15.0),
        &FrameRepairOptions::default(),
    )
    .unwrap();
    assert_eq!(
        report.frames[2].reasons,
        vec![FrameFlagReason::AreaJump, FrameFlagReason::LowIou]
    );
    assert!(report.frames[2].anchor_recovered);
    assert!(!report.frames[2].is_flagged());
    assert_eq!(report.frames[2].repair, FrameRepairAction::Kept);
    assert_eq!(
        masks[2].pixels, valid.pixels,
        "3860-like valid frame retains its as-run shape and values"
    );
    assert!(report.frames[1].is_flagged());
    assert_eq!(report.frames[1].repair, FrameRepairAction::Clipped);
    assert_eq!(report.summary().initially_flagged_frames, 2);
    assert_eq!(report.summary().recovered_anchor_frames, 1);
    assert_eq!(report.summary().flagged_frames, 1);
}

#[test]
fn anchor_recovery_uses_inclusive_area_bounds_and_even_local_median() {
    let mut report = FrameRepairReport {
        options: FrameRepairOptions::default(),
        frames: vec![
            record(8, vec![]),
            record(12, vec![]),
            record(7, vec![FrameFlagReason::LowIou]),
            record(14, vec![FrameFlagReason::AreaJump]),
            record(6, vec![FrameFlagReason::AreaJump]),
            record(15, vec![FrameFlagReason::AreaJump]),
            record(10, vec![FrameFlagReason::EdgeContact]),
            record(0, vec![FrameFlagReason::EmptyComponent]),
            record(10, vec![FrameFlagReason::MissingTrace]),
        ],
    };
    recover_anchors(&mut report);
    assert_eq!(
        report
            .frames
            .iter()
            .map(|frame| frame.anchor_recovered)
            .collect::<Vec<_>>(),
        vec![false, false, true, true, false, false, false, false, false]
    );
    assert_eq!(
        report.frames[2].reasons,
        vec![FrameFlagReason::LowIou],
        "initial evidence is immutable"
    );
}

#[test]
fn recovery_uses_initial_accepted_snapshot_and_inclusive_150_frame_window() {
    let mut report = FrameRepairReport {
        options: FrameRepairOptions::default(),
        frames: (0..302)
            .map(|_| record(30, vec![FrameFlagReason::EdgeContact]))
            .collect(),
    };
    report.frames[0] = record(8, vec![]);
    report.frames[300] = record(12, vec![]);
    report.frames[301] = record(1000, vec![]);
    report.frames[150] = record(7, vec![FrameFlagReason::LowIou]);
    recover_anchors(&mut report);
    assert!(
        report.frames[150].anchor_recovered,
        "both +/-150 endpoints included; +151 excluded"
    );

    let mut report = FrameRepairReport {
        options: FrameRepairOptions::default(),
        frames: (0..152)
            .map(|_| record(30, vec![FrameFlagReason::EdgeContact]))
            .collect(),
    };
    report.frames[0] = record(10, vec![]);
    report.frames[150] = record(14, vec![FrameFlagReason::LowIou]);
    report.frames[151] = record(14, vec![FrameFlagReason::LowIou]);
    recover_anchors(&mut report);
    assert!(report.frames[150].anchor_recovered);
    assert!(
        !report.frames[151].anchor_recovered,
        "recovered150 cannot supply a new anchor to151"
    );
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
        assert!(!report.frames[0].anchor_recovered);
        assert_eq!(report.frames[0].repair, FrameRepairAction::Kept);
        assert_eq!(masks[0].pixels, original);
        assert_eq!(report.summary().unrepaired_frames, 1);
    }
}

#[test]
fn missing_trace_and_wholly_flagged_stacks_remain_empty_or_keep_selected_components() {
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
        && frame.repair == FrameRepairAction::Kept));
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
        .all(|frame| frame.repair == FrameRepairAction::Kept && frame.is_flagged()));
    for mask in flagged {
        assert_eq!(mask.pixels[0], 255);
        assert_eq!(mask.pixels[8], 255);
        assert_eq!(
            mask.pixels[40], 0,
            "unrepaired output still contains only its selected component"
        );
    }
}

#[test]
fn empty_component_is_replaced_when_trace_and_an_accepted_estimate_exist() {
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
    assert_eq!(report.frames[1].component_area, 0);
    assert_eq!(report.frames[1].repair, FrameRepairAction::Replaced);
    assert!(
        report.frames[2]
            .reasons
            .contains(&FrameFlagReason::AreaJump),
        "classification uses raw empty predecessor"
    );
    assert!(report.frames[2].anchor_recovered);
    assert!(!report.is_repaired(2));
    assert_eq!(masks[1].pixels, anchor.pixels);
    assert_eq!(masks[2].pixels, anchor.pixels);
}

#[test]
fn aligned_sdf_estimates_follow_integer_centroid_motion_and_fractional_sampling() {
    let left = frame(40, 15, &rectangle(2, 3, 3, 3));
    let right = frame(40, 15, &rectangle(26, 7, 3, 3));
    let anchors = AlignedAnchors::new(
        &left,
        &accepted_record(&left),
        &right,
        &accepted_record(&right),
    );
    for (fraction, x, y) in [
        (0.0, 2, 3),
        (0.25, 8, 4),
        (0.5, 14, 5),
        (0.75, 20, 6),
        (1.0, 26, 7),
    ] {
        let expected = frame(40, 15, &rectangle(x, y, 3, 3));
        assert_eq!(
            anchors.estimate(fraction),
            expected
                .pixels
                .iter()
                .map(|&pixel| pixel != 0)
                .collect::<Vec<_>>()
        );
    }
    let left = frame(8, 7, &rectangle(2, 2, 3, 3));
    let right = frame(8, 7, &rectangle(3, 2, 3, 3));
    let anchors = AlignedAnchors::new(
        &left,
        &accepted_record(&left),
        &right,
        &accepted_record(&right),
    );
    let expected = frame(8, 7, &rectangle(3, 2, 2, 3));
    assert_eq!(
        anchors.estimate(0.5),
        expected
            .pixels
            .iter()
            .map(|&pixel| pixel != 0)
            .collect::<Vec<_>>(),
        "half-pixel alignment uses bilinear SDF samples and excludes exact zero values"
    );
}

#[test]
fn clipping_preserves_original_detail_and_replacement_cutoff_is_strictly_thirty_percent() {
    let expected = frame(64, 48, &rectangle(20, 20, 5, 2));
    let estimate = Estimate::new(
        expected.pixels.iter().map(|&pixel| pixel != 0).collect(),
        64,
        48,
    );
    let mut detailed = frame(64, 48, &rectangle(20, 20, 26, 1));
    detailed.pixels[20 * 64 + 22] = 1;
    assert_eq!(estimate.apply_to(&mut detailed), FrameRepairAction::Clipped);
    assert_eq!(detailed.pixels[20 * 64 + 22], 1);
    assert_eq!(
        detailed.pixels[20 * 64 + 30],
        255,
        "detail outside the estimate survives within its dilation"
    );
    assert_eq!(detailed.pixels[20 * 64 + 34], 255, "radius10 is included");
    assert_eq!(detailed.pixels[20 * 64 + 35], 0);
    for (count, action) in [
        (3, FrameRepairAction::Kept),
        (2, FrameRepairAction::Replaced),
    ] {
        let mut mask = frame(64, 48, &rectangle(20, 20, count, 1));
        let original = mask.pixels.clone();
        assert_eq!(estimate.apply_to(&mut mask), action);
        if count == 3 {
            assert_eq!(mask.pixels, original);
        } else {
            assert_eq!(mask.pixels, expected.pixels);
        }
    }
}

#[test]
fn empty_estimate_uses_literal_empty_intersection_and_records_actual_changes() {
    let estimate = Estimate::new(vec![false; 7 * 7], 7, 7);
    let mut empty = frame(7, 7, &[]);
    assert_eq!(estimate.apply_to(&mut empty), FrameRepairAction::Kept);
    let mut nonempty = frame(7, 7, &[(3, 3)]);
    assert_eq!(estimate.apply_to(&mut nonempty), FrameRepairAction::Clipped);
    assert!(nonempty.pixels.iter().all(|&pixel| pixel == 0));
}

fn dilated_rectangle(width: usize, height: usize, left: usize, top: usize, side: usize) -> Vec<u8> {
    (0..width * height)
        .map(|index| {
            let x = index % width;
            let y = index / width;
            let dx = if x < left {
                left - x
            } else {
                x.saturating_sub(left + side - 1)
            };
            let dy = if y < top {
                top - y
            } else {
                y.saturating_sub(top + side - 1)
            };
            if dx * dx + dy * dy <= 100 {
                255
            } else {
                0
            }
        })
        .collect()
}

#[test]
fn leading_internal_three_frame_and_trailing_runs_clip_using_accepted_estimates() {
    let mut left = frame(64, 40, &rectangle(20, 18, 3, 3));
    left.pixels[19 * 64 + 21] = 1;
    let right = frame(64, 40, &rectangle(28, 18, 3, 3));
    let flood = FrameMask {
        width: 64,
        height: 40,
        pixels: vec![255; 64 * 40],
    };
    let mut masks = vec![
        flood.clone(),
        flood.clone(),
        left.clone(),
        flood.clone(),
        flood.clone(),
        flood.clone(),
        right.clone(),
        flood.clone(),
        flood,
    ];
    let options = FrameRepairOptions {
        max_area_ratio: 1000.0,
        min_iou: 0.0,
        keep_unrepaired: true,
    };
    let report = repair_inconsistent_frames(&mut masks, &trace(26.0, 19.0), &options).unwrap();
    for (index, expected_x) in [
        (0, 20),
        (1, 20),
        (3, 22),
        (4, 24),
        (5, 26),
        (7, 28),
        (8, 28),
    ] {
        assert_eq!(report.frames[index].repair, FrameRepairAction::Clipped);
        assert_eq!(
            masks[index].pixels,
            dilated_rectangle(64, 40, expected_x, 18, 3),
            "frame{index} preserves the original within the aligned estimate's disk envelope"
        );
    }
    assert_eq!(masks[2].pixels, left.pixels);
    assert_eq!(masks[6].pixels, right.pixels);
    assert_eq!(
        report.summary(),
        FrameRepairSummary {
            total_frames: 9,
            initially_flagged_frames: 7,
            recovered_anchor_frames: 0,
            accepted_frames: 2,
            flagged_frames: 7,
            repaired_frames: 7,
            kept_frames: 2,
            clipped_frames: 7,
            replaced_frames: 0,
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
    assert_eq!(serialized["frames"][0]["repair"], "clipped");
    assert_eq!(serialized["frames"][0]["component_area"], 64 * 40);
    assert_eq!(
        serialized["frames"][0]["reasons"],
        serde_json::json!(["edge_contact"])
    );

    // Moving records changes identity everywhere; no stored index can go stale.
    let mut reordered = report;
    reordered.frames.rotate_left(2);
    let serialized = serde_json::to_value(&reordered).unwrap();
    for (index, frame) in serialized["frames"].as_array().unwrap().iter().enumerate() {
        assert_eq!(frame["frame_index"], index);
        assert_eq!(frame["repair"] != "kept", reordered.is_repaired(index));
    }
    assert_eq!(serialized["frames"][0]["component_area"], 9);
    assert_eq!(
        reordered
            .summary()
            .runs
            .iter()
            .map(|run| (run.start_frame, run.end_frame))
            .collect::<Vec<_>>(),
        [(1, 3), (5, 8)]
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
