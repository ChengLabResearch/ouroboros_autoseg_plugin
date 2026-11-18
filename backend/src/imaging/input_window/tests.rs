use super::*;
use crate::{imaging::annotations::AnnotationOrigin, inference::image::PointPrompt};

fn controller(options: InputWindowOptions) -> InputWindowController {
    let annotations = vec![AnnotationPoint {
        x: 64.0,
        y: 64.0,
        z: 0.0,
        origin: AnnotationOrigin::Metadata,
    }];
    let prompts = vec![VideoFramePrompt {
        frame_index: 0,
        points: vec![PointPrompt::positive(
            64.0,
            64.0,
            AnnotationOrigin::Metadata,
        )],
    }];
    InputWindowController::new(options, annotations.into(), prompts.into(), false).unwrap()
}

fn disc(x: usize, radius: usize) -> FrameMask {
    let pixels = (0usize..128 * 128)
        .map(|index| {
            if (index % 128).abs_diff(x).pow(2) + (index / 128).abs_diff(64).pow(2) <= radius.pow(2)
            {
                255
            } else {
                0
            }
        })
        .collect();
    FrameMask {
        width: 128,
        height: 128,
        pixels,
    }
}

fn image(mask: &FrameMask) -> ImageFrame {
    ImageFrame {
        width: mask.width,
        height: mask.height,
        channels: 1,
        pixels: mask
            .pixels
            .iter()
            .map(|&value| if value == 0 { 20 } else { 200 })
            .collect(),
    }
}

#[test]
fn follows_a_moving_disc_without_changing_interior_pixels_or_raw_masks() {
    let mut window = controller(InputWindowOptions::default());
    for frame_index in 0..5 {
        let mask = Arc::new(disc(64 + frame_index, 8));
        let original = image(&mask);
        let mut blanked = original.clone();
        window.blank_frame(frame_index, &mut blanked).unwrap();
        let bounds = window.frames[frame_index].bounds;
        assert!(bounds.left <= 64 + frame_index - 8 && bounds.right > 64 + frame_index + 8);
        for (index, &pixel) in blanked.pixels.iter().enumerate() {
            let inside = (bounds.left..bounds.right).contains(&(index % 128))
                && (bounds.top..bounds.bottom).contains(&(index / 128));
            assert_eq!(pixel, if inside { original.pixels[index] } else { 20 });
        }
        let expected = mask.pixels.clone();
        window.observe(frame_index, mask.clone()).unwrap();
        assert_eq!(mask.pixels, expected);
        assert!(window.frames[frame_index]
            .raw_detection
            .as_ref()
            .unwrap()
            .record
            .reasons
            .is_empty());
    }
    let report = window.finish().unwrap();
    assert_eq!(report.summary().resets, 0);
    assert_eq!(report.summary().border_frames, 1);
}

#[test]
fn automatic_bootstrap_preserves_objects_wider_than_64_pixels() {
    let mut window = controller(InputWindowOptions::default());
    let mask = Arc::new(disc(64, 40));
    let original = image(&mask);
    let mut first = original.clone();
    window.blank_frame(0, &mut first).unwrap();
    assert_eq!(first.pixels, original.pixels);
    window.observe(0, mask.clone()).unwrap();
    assert_eq!(window.automatic_size, Some(162));
    window.blank_frame(1, &mut image(&mask)).unwrap();
    let bounds = window.frames[1].bounds;
    assert!(bounds.left <= 24 && bounds.right >= 105);
    window.observe(1, mask).unwrap();

    // An explicit size still applies immediately, including sizes below 64.
    let mut explicit = controller(InputWindowOptions {
        window_size: Some(48),
        ..Default::default()
    });
    let mut first = original.clone();
    explicit.blank_frame(0, &mut first).unwrap();
    assert_eq!(
        explicit.frames[0].bounds.right - explicit.frames[0].bounds.left,
        48
    );
    assert_eq!(first.pixels[64 * 128 + 24], 20);
}

#[test]
fn empty_automatic_bootstrap_stays_unwindowed_until_a_mask_can_be_measured() {
    let mut window = controller(InputWindowOptions::default());
    let mut empty = disc(64, 8);
    empty.pixels.fill(0);
    let mask = Arc::new(disc(64, 40));
    for (index, raw) in [Arc::new(empty), mask].into_iter().enumerate() {
        let original = image(&raw);
        let mut input = original.clone();
        window.blank_frame(index, &mut input).unwrap();
        assert_eq!(
            window.frames[index].bounds,
            PixelBounds {
                left: 0,
                top: 0,
                right: 128,
                bottom: 128,
            }
        );
        assert_eq!(input.pixels, original.pixels);
        window.observe(index, raw).unwrap();
    }
    assert_eq!(window.automatic_size, Some(162));
}

#[test]
fn each_window_edge_has_a_per_frame_outward_growth_cap() {
    let mut window = controller(InputWindowOptions {
        margin: 20,
        max_growth: 2,
        window_size: Some(64),
    });
    for frame_index in 0..4 {
        let mask = Arc::new(disc(64 + frame_index, 25));
        window.blank_frame(frame_index, &mut image(&mask)).unwrap();
        if frame_index > 0 {
            let previous = window.frames[frame_index - 1].bounds;
            let current = window.frames[frame_index].bounds;
            assert_eq!(current.left, previous.left - 2);
            assert_eq!(current.top, previous.top - 2);
            assert_eq!(current.right, previous.right + 2);
            assert_eq!(current.bottom, previous.bottom + 2);
        }
        window.observe(frame_index, mask).unwrap();
    }
}

#[test]
fn a_flood_resets_the_next_frame_without_inflating_the_automatic_square() {
    let mut window = controller(InputWindowOptions::default());
    let good = Arc::new(disc(64, 8));
    window.blank_frame(0, &mut image(&good)).unwrap();
    window.observe(0, good.clone()).unwrap();
    window.blank_frame(1, &mut image(&good)).unwrap();
    let flood = Arc::new(FrameMask {
        width: 128,
        height: 128,
        pixels: vec![255; 128 * 128],
    });
    window.observe(1, flood.clone()).unwrap();
    window.blank_frame(2, &mut image(&good)).unwrap();
    assert_eq!(window.frames[2].reset, Some(WindowReset::Flag));
    assert_eq!(
        window.frames[2].bounds,
        PixelBounds {
            left: 32,
            top: 32,
            right: 96,
            bottom: 96
        }
    );
    assert_eq!(window.frames[2].background, [20]);
    window.observe(2, good).unwrap();
    let report = window.finish().unwrap();
    assert_eq!(report.summary().flag_resets, 1);
    assert!(flood.pixels.iter().all(|&pixel| pixel == 255));
}

#[test]
fn prepared_prompt_membership_resets_the_window_and_report_indices_are_derived() {
    let annotations: Arc<[_]> = vec![AnnotationPoint {
        x: 64.0,
        y: 64.0,
        z: 0.0,
        origin: AnnotationOrigin::Metadata,
    }]
    .into();
    let prompts = (0..3)
        .map(|frame_index| VideoFramePrompt {
            frame_index,
            points: vec![PointPrompt::positive(
                64.0,
                64.0,
                AnnotationOrigin::Metadata,
            )],
        })
        .collect::<Vec<_>>()
        .into();
    let mut window =
        InputWindowController::new(InputWindowOptions::default(), annotations, prompts, false)
            .unwrap();
    for frame_index in 0..3 {
        let mask = Arc::new(disc(64, 8));
        window.blank_frame(frame_index, &mut image(&mask)).unwrap();
        window.observe(frame_index, mask).unwrap();
    }
    let report = window.finish().unwrap();
    assert_eq!(report.summary().prompt_resets, 2);
    let json = serde_json::to_value(report).unwrap();
    assert_eq!(json["frames"][2]["frame_index"], 2);
    assert_eq!(json["frames"][2]["reset"], "prompt");
}

#[test]
fn background_median_excludes_previous_foreground_per_channel() {
    let frame = ImageFrame {
        width: 4,
        height: 1,
        channels: 3,
        pixels: vec![10, 20, 30, 20, 30, 40, 200, 210, 220, 240, 250, 255],
    };
    let previous = FrameMask {
        width: 4,
        height: 1,
        pixels: vec![0, 0, 255, 255],
    };
    assert_eq!(median_background(&frame, Some(&previous)), [15, 25, 35]);
    assert_eq!(median_background(&frame, None), [110, 120, 130]);
}
