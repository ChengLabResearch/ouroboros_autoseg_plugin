use std::{path::Path, sync::Arc};

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

pub type FrameProgressCallback = Arc<dyn Fn(usize, usize) + Send + Sync>;

/// Segmenter that propagates prompts through a stack of staged frame images.
///
/// `frames_dir` must be a directory of JPEG files as written by
/// `imaging::preprocess::stage_input_frames` with `PreparedFrameEncoding::Jpeg`.
/// The returned `Vec<FrameMask>` has one entry per frame in `frames_dir`,
/// in the same order as the files would be sorted by file name.
#[async_trait]
pub trait VideoSegmenter: Send + Sync {
    async fn segment_video(
        &self,
        frames_dir: &Path,
        prompts: &[VideoFramePrompt],
        progress: Option<FrameProgressCallback>,
    ) -> Result<Vec<FrameMask>, AppError>;
}
