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

fn flood() -> FrameMask {
    FrameMask {
        width: 128,
        height: 128,
        pixels: vec![255; 128 * 128],
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
    // Shrinking from the unrestricted input is not limited by max_growth.
    assert_eq!(
        report.frames[1].bounds,
        PixelBounds {
            left: 48,
            top: 48,
            right: 81,
            bottom: 81,
        }
    );
}

#[test]
fn startup_is_unrestricted_or_explicit_until_an_accepted_component_sets_the_window() {
    let mut window = controller(InputWindowOptions::default());
    let mask = Arc::new(disc(64, 40));
    let original = image(&mask);
    let mut first = original.clone();
    window.blank_frame(0, &mut first).unwrap();
    assert_eq!(first.pixels, original.pixels);
    window.observe(0, mask.clone()).unwrap();
    window.blank_frame(1, &mut image(&mask)).unwrap();
    assert_eq!(
        window.frames[1].bounds,
        PixelBounds {
            left: 16,
            top: 16,
            right: 113,
            bottom: 113,
        }
    );
    window.observe(1, mask).unwrap();

    // The explicit square applies immediately, but only during startup.
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
    let good = Arc::new(disc(64, 8));
    explicit.observe(0, good.clone()).unwrap();
    explicit.blank_frame(1, &mut image(&good)).unwrap();
    assert_eq!(
        explicit.frames[1].bounds.right - explicit.frames[1].bounds.left,
        33
    );
}

#[test]
fn empty_and_flagged_bootstrap_stays_unrestricted_until_a_component_is_accepted() {
    let mut window = controller(InputWindowOptions::default());
    let mut empty = disc(64, 8);
    empty.pixels.fill(0);
    let good = Arc::new(disc(64, 8));
    for (index, raw) in [
        Arc::new(empty),
        Arc::new(flood()),
        good.clone(),
        good.clone(),
    ]
    .into_iter()
    .enumerate()
    {
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
        assert_eq!(window.accepted_window.is_some(), index == 3);
    }
    window.blank_frame(4, &mut image(&good)).unwrap();
    assert_eq!(window.frames[4].bounds.left, 48);
    assert_eq!(window.frames[4].bounds.right, 81);
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
fn disconnected_raw_flood_does_not_set_the_accepted_window() {
    let mut window = controller(InputWindowOptions::default());
    let mut raw = disc(64, 8);
    for row in raw.pixels.chunks_exact_mut(128) {
        row[..20].fill(255);
    }
    let raw = Arc::new(raw);
    let original = raw.pixels.clone();
    window.blank_frame(0, &mut image(&raw)).unwrap();
    window.observe(0, raw.clone()).unwrap();
    let selected = window.frames[0].raw_detection.as_ref().unwrap();
    assert!(selected.record.reasons.is_empty());
    assert_eq!(selected.component.pixels, disc(64, 8).pixels);
    window.blank_frame(1, &mut image(&raw)).unwrap();
    assert_eq!(
        window.frames[1].bounds,
        PixelBounds {
            left: 48,
            top: 48,
            right: 81,
            bottom: 81,
        }
    );
    assert_eq!(raw.pixels, original);
}

#[test]
fn consecutive_flags_hold_the_last_component_derived_window_and_preserve_history() {
    let mut window = controller(InputWindowOptions::default());
    let good = Arc::new(disc(64, 8));
    window.blank_frame(0, &mut image(&good)).unwrap();
    window.observe(0, good.clone()).unwrap();
    let flood = Arc::new(flood());
    let held = PixelBounds {
        left: 48,
        top: 48,
        right: 81,
        bottom: 81,
    };
    for frame_index in 1..=4 {
        window.blank_frame(frame_index, &mut image(&good)).unwrap();
        assert_eq!(window.frames[frame_index].bounds, held);
        assert_eq!(window.frames[frame_index].background, [20]);
        if frame_index > 1 {
            assert_eq!(window.frames[frame_index].reset, Some(WindowReset::Flag));
        }
        window.observe(frame_index, flood.clone()).unwrap();
    }
    let mut original = image(&good);
    original.pixels[0] = 77;
    let mut replay = original.clone();
    window.blank_frame(0, &mut replay).unwrap();
    assert_eq!(replay.pixels, original.pixels);
    window.blank_frame(1, &mut replay).unwrap();
    assert_eq!(replay.pixels[0], 20);
    assert_eq!(window.frames[1].bounds, held);
    let report = window.finish().unwrap();
    assert_eq!(report.summary().flag_resets, 3);
    assert_eq!(report.summary().border_frames, 1);
    assert!(flood.pixels.iter().all(|&pixel| pixel == 255));
}

#[test]
fn prompts_translate_accepted_dimensions_even_after_flags_and_near_borders() {
    let annotations: Arc<[_]> = [
        (64.0, 64.0),
        (64.0, 64.0),
        (90.0, 30.0),
        (127.0, 127.0),
        (0.0, 0.0),
        (64.0, 64.0),
    ]
    .into_iter()
    .enumerate()
    .map(|(frame_index, (x, y))| AnnotationPoint {
        x,
        y,
        z: frame_index as f32,
        origin: AnnotationOrigin::Metadata,
    })
    .collect::<Vec<_>>()
    .into();
    let prompts = [0, 2, 3, 4]
        .into_iter()
        .map(|frame_index| VideoFramePrompt {
            frame_index,
            points: vec![PointPrompt::positive(
                annotations[frame_index].x,
                annotations[frame_index].y,
                AnnotationOrigin::Metadata,
            )],
        })
        .collect::<Vec<_>>()
        .into();
    let mut window =
        InputWindowController::new(InputWindowOptions::default(), annotations, prompts, false)
            .unwrap();
    let mut good = disc(64, 8);
    for (row_index, row) in good.pixels.chunks_exact_mut(128).enumerate() {
        if !(60..69).contains(&row_index) {
            row.fill(0);
        }
    }
    window.blank_frame(0, &mut image(&good)).unwrap();
    window.observe(0, Arc::new(good)).unwrap();
    let accepted = window.accepted_window.unwrap();
    assert_eq!(accepted.right - accepted.left, 33);
    assert_eq!(accepted.bottom - accepted.top, 25);
    let expected = [
        accepted,
        PixelBounds {
            left: 74,
            top: 18,
            right: 107,
            bottom: 43,
        },
        PixelBounds {
            left: 95,
            top: 103,
            right: 128,
            bottom: 128,
        },
        PixelBounds {
            left: 0,
            top: 0,
            right: 33,
            bottom: 25,
        },
        accepted,
    ];
    for (frame_index, expected) in (1..=5).zip(expected) {
        let mask = Arc::new(flood());
        window.blank_frame(frame_index, &mut image(&mask)).unwrap();
        assert_eq!(window.frames[frame_index].bounds, expected);
        if (2..=4).contains(&frame_index) {
            assert_eq!(window.frames[frame_index].reset, Some(WindowReset::Prompt));
        }
        window.observe(frame_index, mask).unwrap();
        assert_eq!(window.accepted_window, Some(accepted));
    }
    let report = window.finish().unwrap();
    assert_eq!(report.summary().prompt_resets, 3);
    assert_eq!(report.summary().flag_resets, 1);
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
