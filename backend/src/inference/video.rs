use std::{path::Path, sync::Arc};

use async_trait::async_trait;

use crate::{
    error::AppError,
    imaging::input_window::SharedInputWindow,
    inference::{image::PointPrompt, SegmentationOutput},
};

#[derive(Debug, Clone)]
pub struct VideoFramePrompt {
    pub frame_index: usize,
    pub points: Vec<PointPrompt>,
}

pub type FrameProgressCallback = Arc<dyn Fn(usize, usize) + Send + Sync>;

/// Segmenter that propagates prompts through a stack of staged frame images.
///
/// `frames_dir` must be a directory of JPEG files as written by
/// `imaging::preprocess::stage_input_frames` with `PreparedFrameEncoding::Jpeg`.
/// The returned masks have one entry per frame in `frames_dir`,
/// in the same order as the files would be sorted by file name.
#[async_trait]
pub trait VideoSegmenter: Send + Sync {
    async fn segment_video(
        &self,
        frames_dir: &Path,
        prompts: Arc<[VideoFramePrompt]>,
        input_window: Option<SharedInputWindow>,
        progress: Option<FrameProgressCallback>,
    ) -> Result<SegmentationOutput, AppError>;
}
