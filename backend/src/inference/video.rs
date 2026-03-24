use async_trait::async_trait;

use crate::{
    error::AppError,
    inference::image::{FrameMask, PositivePointPrompt},
};

#[derive(Debug, Clone)]
pub struct VideoFramePrompt {
    pub frame_index: usize,
    pub points: Vec<PositivePointPrompt>,
}

#[async_trait]
pub trait VideoSegmenter: Send + Sync {
    async fn segment_video(
        &self,
        frame_count: usize,
        prompts: &[VideoFramePrompt],
    ) -> Result<Vec<FrameMask>, AppError>;
}
