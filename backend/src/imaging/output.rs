use std::path::Path;

use crate::{
    error::AppError,
    imaging::{
        annotations::{interpolated_point_for_frame, AnnotationPoint},
        overlay::draw_annotation_star,
        tiff_io::{write_tiff_pages, WritableTiffPage},
    },
    inference::image::FrameMask,
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
    annotation_points: &[AnnotationPoint],
    intensity: u8,
) -> Result<(), AppError> {
    let mut overlay_masks = masks.to_vec();
    for (frame_index, mask) in overlay_masks.iter_mut().enumerate() {
        if let Some(point) = interpolated_point_for_frame(annotation_points, frame_index) {
            draw_annotation_star(
                &mut mask.pixels,
                mask.width,
                mask.height,
                point.x,
                point.y,
                intensity,
            );
        }
    }

    write_mask_stack(target, &overlay_masks).await
}

#[cfg(test)]
mod tests;
