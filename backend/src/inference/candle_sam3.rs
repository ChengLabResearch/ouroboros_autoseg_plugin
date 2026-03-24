use std::path::PathBuf;

use async_trait::async_trait;

use crate::{
    error::AppError,
    imaging::tiff_io::ImageFrame,
    inference::{
        image::{FrameMask, ImageSegmenter, PositivePointPrompt},
        video::{VideoFramePrompt, VideoSegmenter},
    },
};

#[derive(Debug, Clone)]
pub struct CandleSam3ImageSegmenter {
    pub checkpoint_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct CandleSam3VideoSegmenter {
    pub checkpoint_path: PathBuf,
}

#[async_trait]
impl ImageSegmenter for CandleSam3ImageSegmenter {
    async fn segment(
        &self,
        _frame: &ImageFrame,
        _prompts: &[PositivePointPrompt],
    ) -> Result<FrameMask, AppError> {
        Err(AppError::not_implemented(format!(
            "Candle SAM3 image inference is not implemented yet for {}",
            self.checkpoint_path.display()
        )))
    }
}

#[async_trait]
impl VideoSegmenter for CandleSam3VideoSegmenter {
    async fn segment_video(
        &self,
        _frame_count: usize,
        _prompts: &[VideoFramePrompt],
    ) -> Result<Vec<FrameMask>, AppError> {
        Err(AppError::not_implemented(format!(
            "Candle SAM3 video inference is not implemented yet for {}",
            self.checkpoint_path.display()
        )))
    }
}
