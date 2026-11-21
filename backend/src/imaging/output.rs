use std::path::Path;

use crate::{
    error::AppError,
    imaging::{
        cleanup::FrameRepairReport,
        input_window::InputWindowReport,
        overlay::{draw_overlay_marker, OverlayMarker},
        tiff_io::{write_tiff_pages, WritableTiffPage},
    },
    inference::{
        image::{FrameMask, PointPromptKind},
        video::VideoFramePrompt,
    },
};

pub async fn write_mask_stack(target: &Path, masks: &[FrameMask]) -> Result<(), AppError> {
    write_mask_stack_with_progress(target, masks, |_, _| {}).await
}

pub async fn write_mask_stack_with_progress<F>(
    target: &Path,
    masks: &[FrameMask],
    mut on_frame: F,
) -> Result<(), AppError>
where
    F: FnMut(usize, usize),
{
    if masks.is_empty() {
        return Err(AppError::bad_request(
            "Mask stack must contain at least one frame",
        ));
    }

    let expected_width = masks[0].width;
    let expected_height = masks[0].height;
    let pages = masks
        .iter()
        .enumerate()
        .map(|(frame_index, mask)| {
            if mask.width != expected_width || mask.height != expected_height {
                return Err(AppError::bad_request(
                    "Mask stack frames must all share the same geometry",
                ));
            }
            if mask.pixels.len() != mask.width * mask.height {
                return Err(AppError::bad_request(
                    "Mask pixels do not match the declared frame geometry",
                ));
            }

            let page = WritableTiffPage {
                width: mask.width,
                height: mask.height,
                channels: 1,
                pixels: mask.pixels.clone(),
                description: None,
            };
            on_frame(frame_index, masks.len());
            Ok(page)
        })
        .collect::<Result<Vec<_>, _>>()?;

    write_tiff_pages(target, &pages)
}

pub async fn write_annotation_overlay_stack(
    target: &Path,
    masks: &[FrameMask],
    prompts: &[VideoFramePrompt],
    intensity: u8,
    repair_report: Option<&FrameRepairReport>,
    input_window_report: Option<&InputWindowReport>,
) -> Result<(), AppError> {
    if input_window_report.is_some_and(|report| report.frames.len() != masks.len()) {
        return Err(AppError::bad_request(
            "Input window report must contain one record per mask frame",
        ));
    }
    let mut overlay_masks = masks.to_vec();
    let mut prompts = prompts.iter().peekable();
    for (frame_index, mask) in overlay_masks.iter_mut().enumerate() {
        if let Some(report) = input_window_report {
            draw_overlay_marker(
                &mut mask.pixels,
                mask.width,
                mask.height,
                0.0,
                0.0,
                if intensity == 96 { 95 } else { 96 },
                OverlayMarker::InputWindow(report.frames[frame_index].bounds),
            );
        }
        if prompts
            .peek()
            .is_some_and(|prompt| prompt.frame_index == frame_index)
        {
            for point in &prompts.next().expect("matching prepared prompt").points {
                let marker_intensity = if point.kind == PointPromptKind::Negative {
                    if intensity == 64 {
                        63
                    } else {
                        64
                    }
                } else {
                    intensity
                };
                draw_overlay_marker(
                    &mut mask.pixels,
                    mask.width,
                    mask.height,
                    point.x,
                    point.y,
                    marker_intensity,
                    OverlayMarker::Prompt(point.kind),
                );
            }
        }
        if repair_report.is_some_and(|report| report.is_repaired(frame_index)) {
            // A fixed top-left X marks repaired frames even without prompts.
            // Draw it last so a prompt at this location cannot conceal repair.
            draw_overlay_marker(
                &mut mask.pixels,
                mask.width,
                mask.height,
                2.0,
                2.0,
                if intensity == 192 { 191 } else { 192 },
                OverlayMarker::RepairedFrame,
            );
        }
    }

    write_mask_stack(target, &overlay_masks).await
}

#[cfg(test)]
mod tests;
