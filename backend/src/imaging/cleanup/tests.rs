use super::*;

fn mask(width: usize, height: usize, foreground: &[(usize, usize)]) -> FrameMask {
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

fn clean(mask: &mut FrameMask, x: f32, y: f32) -> CleanupStats {
    keep_centerline_components(
        std::slice::from_mut(mask),
        &[AnnotationPoint { x, y, z: 0.0 }],
    )
    .expect("cleanup")
}

#[test]
fn seeded_component_keeps_original_values_and_removes_other_regions() {
    let mut frame = mask(5, 3, &[(0, 0), (1, 0), (4, 2)]);
    frame.pixels[0] = 1;
    let stats = clean(&mut frame, 0.0, 0.0);
    assert_eq!(frame.pixels[0], 1);
    assert_eq!(frame.pixels[1], 255);
    assert_eq!(frame.pixels[14], 0);
    assert_eq!(
        stats,
        CleanupStats {
            frames_changed: 1,
            pixels_removed: 1,
            ..CleanupStats::default()
        }
    );
}

#[test]
fn background_seed_uses_nearest_foreground_with_row_major_ties() {
    let mut frame = mask(9, 1, &[(1, 0), (3, 0), (6, 0), (8, 0)]);
    let stats = clean(&mut frame, 4.0, 0.0);
    assert_eq!(frame.pixels, vec![0, 0, 0, 255, 0, 0, 0, 0, 0]);
    assert_eq!(stats.seed_misses, 1);
    assert_eq!(stats.frames_without_seed, 0);
    let mut tied = mask(5, 1, &[(0, 0), (4, 0)]);
    clean(&mut tied, 2.0, 0.0);
    assert_eq!(tied.pixels, vec![255, 0, 0, 0, 0]);
}

#[test]
fn seed_search_is_bounded_by_euclidean_radius_and_leaves_misses_unchanged() {
    for foreground in [vec![], vec![(11, 0)], vec![(8, 8)]] {
        let mut frame = mask(12, 12, &foreground);
        let original = frame.pixels.clone();
        let stats = clean(&mut frame, 0.0, 0.0);
        assert_eq!(frame.pixels, original);
        assert_eq!(
            stats,
            CleanupStats {
                seed_misses: 1,
                frames_without_seed: 1,
                ..CleanupStats::default()
            }
        );
    }
    let mut boundary = mask(12, 12, &[(10, 0), (11, 11)]);
    let stats = clean(&mut boundary, 0.0, 0.0);
    assert_eq!(boundary.pixels[10], 255);
    assert_eq!(stats.pixels_removed, 1);
}

#[test]
fn diagonal_bridge_is_connected_but_frames_are_independent() {
    let mut frames = vec![
        mask(4, 4, &[(0, 0), (1, 1), (2, 2), (3, 0)]),
        mask(4, 4, &[(0, 0), (3, 0)]),
    ];
    let stats = keep_centerline_components(
        &mut frames,
        &[AnnotationPoint {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }],
    )
    .unwrap();
    assert_eq!(frames[0].pixels[10], 255);
    assert_eq!(frames[0].pixels[3], 0);
    assert_eq!(frames[1].pixels[3], 0);
    assert_eq!(stats.frames_changed, 2);
    assert_eq!(stats.pixels_removed, 2);
}

#[test]
fn seed_coordinates_are_rounded_and_clamped_at_all_edges() {
    for (x, y, keep) in [
        (-5.0, -2.0, 0),
        (99.0, 0.0, 4),
        (0.0, 99.0, 20),
        (99.0, 99.0, 24),
        (3.6, 3.6, 24),
    ] {
        let mut frame = mask(5, 5, &[(0, 0), (4, 0), (0, 4), (4, 4)]);
        let stats = clean(&mut frame, x, y);
        assert_eq!(frame.pixels[keep], 255);
        assert_eq!(frame.pixels.iter().filter(|&&v| v != 0).count(), 1);
        assert_eq!(stats.seed_misses, 0);
    }
}

#[test]
fn seeds_follow_existing_interpolation_and_default_annotation_fallback() {
    let mut frames = vec![mask(25, 1, &[(0, 0), (12, 0), (24, 0)]); 3];
    let points = [
        AnnotationPoint {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        AnnotationPoint {
            x: 24.0,
            y: 0.0,
            z: 2.0,
        },
    ];
    keep_centerline_components(&mut frames, &points).unwrap();
    for (frame, x) in frames.iter().zip([0, 12, 24]) {
        assert_eq!(frame.pixels[x], 255);
        assert_eq!(frame.pixels.iter().filter(|&&v| v != 0).count(), 1);
    }
    let mut frames = vec![mask(25, 1, &[(0, 0), (12, 0), (24, 0)]); 3];
    let defaults = crate::imaging::annotations::default_annotations(
        crate::imaging::annotations::VolumeShape {
            frames: 3,
            width: 25,
            height: 1,
        },
        200,
    );
    keep_centerline_components(&mut frames, &defaults).unwrap();
    assert!(frames
        .iter()
        .all(|frame| frame.pixels[12] == 255 && frame.pixels[0] == 0 && frame.pixels[24] == 0));
}

#[test]
fn absent_annotations_preserve_masks_and_count_misses() {
    let mut frames = vec![mask(5, 1, &[(0, 0), (4, 0)])];
    let stats = keep_centerline_components(&mut frames, &[]).unwrap();
    assert_eq!(frames[0].pixels, vec![255, 0, 0, 0, 255]);
    assert_eq!(stats.frames_without_seed, 1);
}

#[test]
fn invalid_mask_geometry_fails_before_indexing() {
    for mut frame in [
        mask(0, 0, &[]),
        FrameMask {
            width: 4,
            height: 2,
            pixels: vec![0; 7],
        },
    ] {
        assert!(keep_centerline_components(
            std::slice::from_mut(&mut frame),
            &[AnnotationPoint {
                x: 0.0,
                y: 0.0,
                z: 0.0
            }]
        )
        .is_err());
    }
}
