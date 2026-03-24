use std::path::{Path, PathBuf};

use crate::{error::AppError, imaging::annotations::AnnotationPoint};

#[derive(Debug, Clone)]
pub struct ImageFrame {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub pixels: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct VolumeGeometry {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
}

#[derive(Debug, Clone)]
pub struct VolumeInput {
    pub source: PathBuf,
    pub geometry: VolumeGeometry,
    pub annotation_points: Option<Vec<AnnotationPoint>>,
}

pub async fn inspect_volume(_path: &Path) -> Result<VolumeInput, AppError> {
    Err(AppError::not_implemented(
        "TIFF inspection is scaffolded but not implemented yet",
    ))
}
