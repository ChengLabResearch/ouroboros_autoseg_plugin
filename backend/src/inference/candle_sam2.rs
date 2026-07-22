use std::path::{Path, PathBuf};

use async_trait::async_trait;

use crate::{
    error::AppError,
    imaging::tiff_io::ImageFrame,
    inference::{
        image::{FrameMask, ImageSegmenter, PositivePointPrompt},
        video::{FrameProgressCallback, VideoFramePrompt, VideoSegmenter},
    },
};

#[derive(Debug, Clone)]
pub struct CandleSam2ImageSegmenter {
    pub checkpoint_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct CandleSam2VideoSegmenter {
    pub checkpoint_path: PathBuf,
}

#[async_trait]
impl ImageSegmenter for CandleSam2ImageSegmenter {
    async fn segment(
        &self,
        _frame: &ImageFrame,
        _prompts: &[PositivePointPrompt],
    ) -> Result<FrameMask, AppError> {
        Err(AppError::not_implemented(format!(
            "Candle SAM2 image inference is not implemented yet for {}",
            self.checkpoint_path.display()
        )))
    }
}

#[async_trait]
impl VideoSegmenter for CandleSam2VideoSegmenter {
    async fn segment_video(
        &self,
        _frames_dir: &Path,
        _prompts: &[VideoFramePrompt],
        _progress: Option<FrameProgressCallback>,
    ) -> Result<Vec<FrameMask>, AppError> {
        Err(AppError::not_implemented(format!(
            "Candle SAM2 video inference is not implemented yet for {}",
            self.checkpoint_path.display()
        )))
    }
}
