use std::path::Path;

use crate::{
    error::AppError,
    imaging::tiff_io::{write_tiff_pages, WritableTiffPage},
    inference::image::FrameMask,
};

pub async fn write_mask_stack(target: &Path, masks: &[FrameMask]) -> Result<(), AppError> {
    if masks.is_empty() {
        return Err(AppError::bad_request(
            "Mask stack must contain at least one frame",
        ));
    }

    let expected_width = masks[0].width;
    let expected_height = masks[0].height;
    let pages = masks
        .iter()
        .map(|mask| {
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

            Ok(WritableTiffPage {
                width: mask.width,
                height: mask.height,
                channels: 1,
                pixels: mask.pixels.clone(),
                description: None,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    write_tiff_pages(target, &pages)
}

#[cfg(test)]
mod tests;
