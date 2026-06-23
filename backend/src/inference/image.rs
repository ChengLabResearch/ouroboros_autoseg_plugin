use async_trait::async_trait;

use crate::{error::AppError, imaging::tiff_io::ImageFrame};

#[derive(Debug, Clone)]
pub struct PositivePointPrompt {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone)]
pub struct FrameMask {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

#[async_trait]
pub trait ImageSegmenter: Send + Sync {
    async fn segment(
        &self,
        frame: &ImageFrame,
        prompts: &[PositivePointPrompt],
    ) -> Result<FrameMask, AppError>;
}
