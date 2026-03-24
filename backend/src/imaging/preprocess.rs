use std::path::PathBuf;

use crate::error::AppError;

#[derive(Debug, Clone)]
pub enum PreparedFrameEncoding {
    Tiff,
    Jpeg,
}

#[derive(Debug, Clone)]
pub struct PreparedFrame {
    pub source: PathBuf,
    pub target: PathBuf,
    pub encoding: PreparedFrameEncoding,
}

pub async fn stage_input_frames(_frames: &[PreparedFrame]) -> Result<(), AppError> {
    Err(AppError::not_implemented(
        "Frame staging is scaffolded but not implemented yet",
    ))
}
