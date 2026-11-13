use super::{
    annotation_samples_for_video, default_annotations, fill_prompt_gaps,
    interpolated_point_for_frame, AnnotationOrigin, AnnotationPoint, VolumeShape,
};

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 1e-6,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn default_annotations_adds_center_points_and_endcap() {
    let points = default_annotations(
        VolumeShape {
            frames: 5,
            height: 4,
            width: 6,
        },
        2,
    );

    assert_eq!(points.len(), 4);
    assert_eq!(
        points[0],
        AnnotationPoint {
            origin: AnnotationOrigin::Fallback,
            x: 3.0,
            y: 2.0,
            z: 0.0
        }
    );
    assert_eq!(
        points[1],
        AnnotationPoint {
            origin: AnnotationOrigin::Fallback,
            x: 3.0,
            y: 2.0,
            z: 2.0
        }
    );
    assert_eq!(
        points[2],
        AnnotationPoint {
            origin: AnnotationOrigin::Fallback,
            x: 3.0,
            y: 2.0,
            z: 4.0
        }
    );
    assert_eq!(
        points[3],
        AnnotationPoint {
            origin: AnnotationOrigin::Fallback,
            x: 3.0,
            y: 2.0,
            z: 4.0
        }
    );
}

#[test]
fn annotation_samples_for_video_groups_valid_points_by_frame() {
    let prompts = annotation_samples_for_video(
        &[
            AnnotationPoint {
                origin: AnnotationOrigin::Metadata,
                x: 1.0,
                y: 2.0,
                z: 0.2,
            },
            AnnotationPoint {
                origin: AnnotationOrigin::Metadata,
                x: 3.0,
                y: 4.0,
                z: 0.4,
            },
            AnnotationPoint {
                origin: AnnotationOrigin::Metadata,
                x: 5.0,
                y: 6.0,
                z: 1.0,
            },
            AnnotationPoint {
                origin: AnnotationOrigin::Metadata,
                x: 9.0,
                y: 9.0,
                z: 8.0,
            },
        ],
        3,
    );

    assert_eq!(prompts.len(), 2);
    assert_eq!(prompts[0].frame_index, 0);
    assert_eq!(prompts[0].points.len(), 2);
    assert_eq!(prompts[1].frame_index, 1);
    assert_eq!(prompts[1].points.len(), 1);
}

#[test]
fn interpolated_point_for_frame_clamps_and_interpolates() {
    let points = [
        AnnotationPoint {
            origin: AnnotationOrigin::Metadata,
            x: 2.0,
            y: 4.0,
            z: 0.0,
        },
        AnnotationPoint {
            origin: AnnotationOrigin::Metadata,
            x: 6.0,
            y: 8.0,
            z: 4.0,
        },
    ];

    let start = interpolated_point_for_frame(&points, 0).expect("start prompt");
    assert_close(start.x, 2.0);
    assert_close(start.y, 4.0);

    let middle = interpolated_point_for_frame(&points, 2).expect("middle prompt");
    assert_close(middle.x, 4.0);
    assert_close(middle.y, 6.0);

    let end = interpolated_point_for_frame(&points, 9).expect("end prompt");
    assert_close(end.x, 6.0);
    assert_close(end.y, 8.0);
}

#[test]
fn fill_prompt_gaps_is_off_for_none_or_zero_and_preserves_small_gaps() {
    let original = vec![
        AnnotationPoint {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            origin: AnnotationOrigin::Metadata,
        },
        AnnotationPoint {
            x: 2.0,
            y: 4.0,
            z: 7.0,
            origin: AnnotationOrigin::Metadata,
        },
        AnnotationPoint {
            x: 3.0,
            y: 6.0,
            z: 10.0,
            origin: AnnotationOrigin::Metadata,
        },
    ];
    for max_gap in [None, Some(0), Some(4)] {
        let mut points = original.clone();
        assert_eq!(fill_prompt_gaps(&mut points, 20, max_gap), 0);
        assert_eq!(points, original);
    }
}

#[test]
fn fill_prompt_gaps_evenly_splits_n_plus_one_two_n_and_two_n_plus_one() {
    // N = 4. Leading/trailing spans are deliberately longer than N.
    for (gap, expected_frames) in [(5, vec![13]), (8, vec![14]), (9, vec![13, 16])] {
        let original = vec![
            AnnotationPoint {
                x: 2.0,
                y: 4.0,
                z: 10.0,
                origin: AnnotationOrigin::Metadata,
            },
            AnnotationPoint {
                x: 2.0 + gap as f32,
                y: 4.0 + 2.0 * gap as f32,
                z: (10 + gap) as f32,
                origin: AnnotationOrigin::Metadata,
            },
        ];
        let mut points = original.clone();
        assert_eq!(
            fill_prompt_gaps(&mut points, 30, Some(4)),
            expected_frames.len()
        );
        assert_eq!(&points[..original.len()], original.as_slice());
        let added = &points[original.len()..];
        assert_eq!(
            added
                .iter()
                .map(|point| point.z as usize)
                .collect::<Vec<_>>(),
            expected_frames
        );
        for point in added {
            assert_eq!(point.origin, AnnotationOrigin::GapFill);
            let offset = point.z - 10.0;
            assert_close(point.x, 2.0 + offset);
            assert_close(point.y, 4.0 + 2.0 * offset);
        }
        let frames = annotation_samples_for_video(&points, 30)
            .into_iter()
            .map(|prompt| prompt.frame_index)
            .collect::<Vec<_>>();
        assert_eq!(frames.first(), Some(&10));
        assert_eq!(frames.last(), Some(&(10 + gap)));
        assert!(frames.windows(2).all(|pair| pair[1] - pair[0] <= 4));
        assert_eq!(fill_prompt_gaps(&mut points, 30, Some(4)), 0);
    }
}

#[test]
fn fill_prompt_gaps_preserves_rounded_annotation_collisions_and_interpolates_fractional_z() {
    let original = vec![
        AnnotationPoint {
            x: 1.0,
            y: 2.0,
            z: 0.2,
            origin: AnnotationOrigin::Metadata,
        },
        AnnotationPoint {
            x: 4.0,
            y: 8.0,
            z: 0.4,
            origin: AnnotationOrigin::Metadata,
        },
        AnnotationPoint {
            x: 44.0,
            y: 88.0,
            z: 4.4,
            origin: AnnotationOrigin::Metadata,
        },
    ];
    let mut points = original.clone();
    assert_eq!(fill_prompt_gaps(&mut points, 6, Some(1)), 3);
    assert_eq!(&points[..original.len()], original.as_slice());
    let prompts = annotation_samples_for_video(&points, 6);
    assert_eq!(
        prompts
            .iter()
            .map(|prompt| prompt.frame_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4]
    );
    assert_eq!(prompts[0].points.len(), 2);
    assert_eq!(prompts[0].points[0].x, 1.0);
    assert_eq!(prompts[0].points[1].x, 4.0);
    assert!(prompts[1..].iter().all(|prompt| prompt.points.len() == 1));
    for point in &points[original.len()..] {
        let expected = interpolated_point_for_frame(&original, point.z as usize).unwrap();
        assert_close(point.x, expected.x);
        assert_close(point.y, expected.y);
    }
    assert_close(prompts[1].points[0].x, 10.0);
    assert_close(prompts[1].points[0].y, 20.0);
}

#[test]
fn fill_prompt_gaps_does_not_add_endpoints_without_two_prompted_frames() {
    for mut points in [
        vec![],
        vec![AnnotationPoint {
            x: 2.0,
            y: 4.0,
            z: 10.0,
            origin: AnnotationOrigin::Metadata,
        }],
    ] {
        let original = points.clone();
        assert_eq!(fill_prompt_gaps(&mut points, 30, Some(1)), 0);
        assert_eq!(points, original);
    }
}
