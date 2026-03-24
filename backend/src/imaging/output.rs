use std::path::Path;

use crate::{error::AppError, inference::image::FrameMask};

pub async fn write_mask_stack(_target: &Path, _masks: &[FrameMask]) -> Result<(), AppError> {
    Err(AppError::not_implemented(
        "TIFF output writing is scaffolded but not implemented yet",
    ))
}
