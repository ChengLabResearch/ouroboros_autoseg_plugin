use super::{
    annotation_samples_for_video, default_annotations, interpolated_point_for_frame,
    AnnotationPoint, VolumeShape,
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
            x: 3.0,
            y: 2.0,
            z: 0.0
        }
    );
    assert_eq!(
        points[1],
        AnnotationPoint {
            x: 3.0,
            y: 2.0,
            z: 2.0
        }
    );
    assert_eq!(
        points[2],
        AnnotationPoint {
            x: 3.0,
            y: 2.0,
            z: 4.0
        }
    );
    assert_eq!(
        points[3],
        AnnotationPoint {
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
                x: 1.0,
                y: 2.0,
                z: 0.2,
            },
            AnnotationPoint {
                x: 3.0,
                y: 4.0,
                z: 0.4,
            },
            AnnotationPoint {
                x: 5.0,
                y: 6.0,
                z: 1.0,
            },
            AnnotationPoint {
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
            x: 2.0,
            y: 4.0,
            z: 0.0,
        },
        AnnotationPoint {
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
