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

#[test]
fn border_points_follow_rectangle_perimeter_and_preserve_positive_provenance() {
    use crate::{domain::requests::NegativeBorderPoints, inference::image::PointPromptKind};

    let annotations = [AnnotationPoint {
        x: 4.0,
        y: 3.0,
        z: 2.0,
        origin: AnnotationOrigin::GapFill,
    }];
    let mut prompts = annotation_samples_for_video(&annotations, 5);
    super::add_negative_border_points(
        &mut prompts,
        VolumeShape {
            frames: 5,
            width: 9,
            height: 7,
        },
        &NegativeBorderPoints {
            count: 10,
            inset_px: 1.0,
        },
    )
    .unwrap();

    assert_eq!(prompts.len(), 1, "negatives must not add prompted frames");
    assert_eq!(prompts[0].frame_index, 2);
    assert_eq!(
        prompts[0].points[0].kind,
        PointPromptKind::Positive(AnnotationOrigin::GapFill)
    );
    let negatives = &prompts[0].points[1..];
    // Rectangle perimeter is 20 px: each negative advances exactly 2 px.
    let expected = [
        (1.0, 1.0),
        (3.0, 1.0),
        (5.0, 1.0),
        (7.0, 1.0),
        (7.0, 3.0),
        (7.0, 5.0),
        (5.0, 5.0),
        (3.0, 5.0),
        (1.0, 5.0),
        (1.0, 3.0),
    ];
    assert_eq!(negatives.len(), expected.len());
    for (point, (x, y)) in negatives.iter().zip(expected) {
        assert_eq!(point.kind, PointPromptKind::Negative);
        assert_close(point.x, x);
        assert_close(point.y, y);
    }
}

#[test]
fn image_and_video_prompts_share_border_points_without_changing_positive_selection() {
    use crate::domain::requests::NegativeBorderPoints;

    let annotations = [
        AnnotationPoint {
            x: 70.0,
            y: 80.0,
            z: 1.0,
            origin: AnnotationOrigin::Metadata,
        },
        AnnotationPoint {
            x: 90.0,
            y: 80.0,
            z: 3.0,
            origin: AnnotationOrigin::Metadata,
        },
    ];
    let shape = VolumeShape {
        frames: 5,
        width: 160,
        height: 160,
    };
    let mut video = annotation_samples_for_video(&annotations, shape.frames);
    let mut image = super::annotation_samples_for_image(&annotations, shape.frames);
    let original_video = video.clone();
    let original_image = image.clone();
    for prompts in [&mut video, &mut image] {
        super::add_negative_border_points(prompts, shape, &NegativeBorderPoints::default())
            .unwrap();
        assert!(prompts.iter().all(|prompt| prompt.points.len() == 16));
    }
    assert_eq!(
        video.iter().map(|p| p.frame_index).collect::<Vec<_>>(),
        vec![1, 3]
    );
    assert_eq!(
        image.iter().map(|p| p.frame_index).collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4]
    );
    assert_eq!(image[2].points[0].x, 80.0);
    for (actual, original) in video
        .iter()
        .chain(&image)
        .zip(original_video.iter().chain(&original_image))
    {
        assert_eq!(actual.points[0], original.points[0]);
        assert_eq!(&actual.points[1..], &video[0].points[1..]);
        for negative in &actual.points[1..] {
            assert!((12.0..=147.0).contains(&negative.x));
            assert!((12.0..=147.0).contains(&negative.y));
            assert!(
                negative.x == 12.0
                    || negative.x == 147.0
                    || negative.y == 12.0
                    || negative.y == 147.0
            );
        }
    }
}

#[test]
fn border_points_reject_an_inset_that_collapses_the_rectangle() {
    let mut prompts = Vec::new();
    let error = super::add_negative_border_points(
        &mut prompts,
        VolumeShape {
            frames: 1,
            width: 25,
            height: 80,
        },
        &crate::domain::requests::NegativeBorderPoints::default(),
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("inset_px (12) leaves no border rectangle in a 25x80 frame"));
}
