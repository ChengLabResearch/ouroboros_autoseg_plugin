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

#[test]
fn accepted_frames_share_eight_connected_trace_selection_and_preserve_values() {
    let mut masks = vec![frame(7, 7, &[(2, 2), (3, 3), (4, 4), (0, 6)])];
    masks[0].pixels[2 * 7 + 2] = 1;
    let report = repair_inconsistent_frames(&mut masks, &trace(3.0, 3.0)).unwrap();
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
fn prepared_classifications_are_consumed_by_cleanup_and_repair_without_remeasurement() {
    let mut valid = frame(40, 40, &rectangle(18, 18, 5, 5));
    valid.pixels[20 * 40 + 20] = 1;
    valid.pixels[0] = 255;
    let raw = vec![
        valid.clone(),
        FrameMask {
            width: 40,
            height: 40,
            pixels: vec![255; 1600],
        },
        frame(40, 40, &rectangle(22, 20, 3, 3)),
        frame(40, 40, &[(39, 39)]),
        valid,
    ];
    for annotations in [trace(20.0, 20.0).to_vec(), vec![]] {
        let mut classifier = TraceFrameClassifier::default();
        let mut classified = Vec::<PreparedTraceClassification>::new();
        for (index, mask) in raw.iter().cloned().enumerate() {
            let previous = classified
                .last()
                .map(|frame| (&frame.component, &frame.record));
            classified.push(
                classifier
                    .prepare(mask, &annotations, index, previous)
                    .unwrap(),
            );
        }
        let initial_reasons = classified
            .iter()
            .map(|frame| frame.record.reasons.as_ptr())
            .collect::<Vec<_>>();
        let mut reused_masks = raw.clone();
        let mut records = Vec::new();
        let mut reused_stats = CleanupStats::default();
        for (mask, frame) in reused_masks.iter_mut().zip(classified) {
            let component_identity = frame.component.pixels.as_ptr();
            let has_seed = frame.cleanup_stats.frames_without_seed == 0;
            let (record, stats) = frame.keep_component(mask);
            if has_seed {
                assert_eq!(
                    mask.pixels.as_ptr(),
                    component_identity,
                    "selected pixels transfer without being recalculated or copied"
                );
            }
            records.push(record);
            reused_stats += stats;
        }
        let mut ordinary_masks = raw.clone();
        let ordinary_stats =
            super::super::keep_centerline_components(&mut ordinary_masks, &annotations).unwrap();
        assert_eq!(reused_stats, ordinary_stats);
        for (reused, ordinary) in reused_masks.iter().zip(&ordinary_masks) {
            assert_eq!(reused.pixels, ordinary.pixels);
        }
        let reused_report =
            repair_classified_frames(&mut reused_masks, FrameRepairReport { frames: records })
                .unwrap();
        let ordinary_report =
            repair_inconsistent_frames(&mut ordinary_masks, &annotations).unwrap();
        assert_eq!(reused_report, ordinary_report);
        for ((record, reasons), (reused, ordinary)) in reused_report
            .frames
            .iter()
            .zip(initial_reasons)
            .zip(reused_masks.iter().zip(&ordinary_masks))
        {
            assert_eq!(
                record.reasons.as_ptr(),
                reasons,
                "the original detector record moves into the final report"
            );
            assert_eq!(reused.pixels, ordinary.pixels);
        }
    }
}

#[test]
fn comparisons_use_previous_raw_selected_component_even_when_it_was_flagged() {
    let small = frame(9, 9, &rectangle(3, 3, 2, 2));
    let large = frame(9, 9, &rectangle(1, 1, 6, 6));
    let mut masks = vec![small, large.clone(), large.clone()];
    let report = repair_inconsistent_frames(&mut masks, &trace(3.0, 3.0)).unwrap();
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
fn fixed_area_ratio_and_iou_threshold_equalities_are_accepted() {
    for (left, right, expected) in [
        (rectangle(2, 2, 2, 2), rectangle(2, 2, 3, 2), vec![]),
        (rectangle(2, 2, 3, 2), rectangle(2, 2, 2, 2), vec![]),
        (
            rectangle(2, 2, 2, 2),
            rectangle(2, 2, 4, 2),
            vec![FrameFlagReason::AreaJump],
        ),
        (rectangle(2, 2, 3, 2), rectangle(3, 2, 3, 2), vec![]),
        (
            rectangle(2, 2, 3, 2),
            rectangle(4, 2, 3, 2),
            vec![FrameFlagReason::LowIou],
        ),
    ] {
        let mut masks = vec![frame(10, 7, &left), frame(10, 7, &right)];
        let report = repair_inconsistent_frames(&mut masks, &trace(3.0, 3.0)).unwrap();
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
    let report = repair_inconsistent_frames(&mut masks, &trace(15.0, 15.0)).unwrap();
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
        let report = repair_inconsistent_frames(&mut masks, &trace(x as f32, y as f32)).unwrap();
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
    let report = repair_inconsistent_frames(&mut missing, &[]).unwrap();
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
    let report = repair_inconsistent_frames(&mut flagged, &trace(0.0, 0.0)).unwrap();
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
    let report = repair_inconsistent_frames(&mut masks, &trace(2.0, 2.0)).unwrap();
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
fn in_place_chessboard_interpolation_preserves_overlap_and_excludes_zero_ties() {
    let left = frame(10, 7, &rectangle(2, 2, 3, 3));
    let right = frame(10, 7, &rectangle(4, 2, 3, 3));
    for expected in [
        vec![
            rectangle(2, 2, 3, 3),
            vec![(4, 2), (3, 3), (4, 3), (5, 3), (4, 4)],
            rectangle(4, 2, 3, 3),
        ],
        // At t=1/3 and 2/3 the outer column has exactly zero blended distance.
        vec![rectangle(3, 2, 2, 3), rectangle(4, 2, 2, 3)],
    ] {
        let mut masks = vec![left.clone()];
        masks.extend((0..expected.len()).map(|_| frame(10, 7, &[])));
        masks.extend([right.clone(), right.clone()]);
        let report = repair_inconsistent_frames(&mut masks, &trace(4.0, 3.0)).unwrap();
        for (offset, pixels) in expected.iter().enumerate() {
            let index = offset + 1;
            assert_eq!(report.frames[index].repair, FrameRepairAction::Replaced);
            assert_eq!(
                masks[index].pixels,
                frame(10, 7, pixels).pixels,
                "frame{index}"
            );
        }
        assert_eq!(masks[0].pixels, left.pixels);
        assert!(report.frames[expected.len() + 1].anchor_recovered);
    }
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

fn dilated_rectangle(
    width: usize,
    height: usize,
    left: usize,
    top: usize,
    columns: usize,
    rows: usize,
) -> Vec<u8> {
    (0..width * height)
        .map(|index| {
            let x = index % width;
            let y = index / width;
            let dx = if x < left {
                left - x
            } else {
                x.saturating_sub(left + columns - 1)
            };
            let dy = if y < top {
                top - y
            } else {
                y.saturating_sub(top + rows - 1)
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
    let right = frame(64, 40, &rectangle(21, 18, 3, 3));
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
        right.clone(), // An initially accepted neighbour supplies the recovery median.
        flood.clone(),
        flood,
    ];
    let report = repair_inconsistent_frames(&mut masks, &trace(21.0, 19.0)).unwrap();
    for (index, expected_x, columns) in [
        (0, 20, 3),
        (1, 20, 3),
        (3, 20, 3),
        (4, 21, 2),
        (5, 21, 3),
        (8, 21, 3),
        (9, 21, 3),
    ] {
        assert_eq!(report.frames[index].repair, FrameRepairAction::Clipped);
        assert_eq!(
            masks[index].pixels,
            dilated_rectangle(64, 40, expected_x, 18, columns, 3),
            "frame{index} preserves the original within the in-place estimate's disc envelope"
        );
    }
    assert_eq!(masks[2].pixels, left.pixels);
    assert_eq!(masks[6].pixels, right.pixels);
    assert_eq!(
        report.summary(),
        FrameRepairSummary {
            total_frames: 10,
            initially_flagged_frames: 9,
            recovered_anchor_frames: 2,
            accepted_frames: 3,
            flagged_frames: 7,
            repaired_frames: 7,
            kept_frames: 3,
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
                    start_frame: 8,
                    end_frame: 9,
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
        [(1, 3), (6, 9)]
    );
}

#[test]
fn repair_rejects_incompatible_frame_geometry_before_mutation() {
    let original = frame(7, 7, &[(0, 0), (4, 4)]);
    for other in [
        frame(8, 7, &[]),
        FrameMask {
            width: 7,
            height: 7,
            pixels: vec![0; 3],
        },
    ] {
        let mut masks = vec![original.clone(), other];
        assert!(repair_inconsistent_frames(&mut masks, &trace(0.0, 0.0)).is_err());
        assert_eq!(masks[0].pixels, original.pixels);
    }
}
