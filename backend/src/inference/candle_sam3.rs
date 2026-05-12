use std::sync::Arc;

use async_trait::async_trait;
use candle_core::DType;
use candle_transformers::models::sam3;

use crate::{
    app_state::Sam3ModelHandle,
    error::AppError,
    imaging::tiff_io::ImageFrame,
    inference::{
        candle_sam3_helpers::{
            build_geometry_prompt, frame_to_chw_tensor, normalize_for_sam3,
            threshold_mask_logits_to_frame,
        },
        image::{FrameMask, ImageSegmenter, PositivePointPrompt},
        video::{VideoFramePrompt, VideoSegmenter},
    },
};

#[derive(Clone)]
pub struct CandleSam3ImageSegmenter {
    pub handle: Arc<Sam3ModelHandle>,
}

#[derive(Clone)]
pub struct CandleSam3VideoSegmenter {
    pub handle: Arc<Sam3ModelHandle>,
}

/// Load SAM3 image and tracker models from a `.pt` checkpoint.
pub fn load_sam3_handle(
    checkpoint_path: &std::path::Path,
    device: candle_core::Device,
) -> Result<Sam3ModelHandle, AppError> {
    let config = sam3::Config::default();
    let source = sam3::Sam3CheckpointSource::upstream_pth(checkpoint_path);
    let image_model =
        sam3::Sam3ImageModel::from_checkpoint_source(&config, &source, DType::F32, &device)
            .map_err(|e| AppError::internal(e.to_string()))?;
    let tracker = sam3::Sam3TrackerModel::from_checkpoint_source(&config, &source, DType::F32, &device)
        .map_err(|e| AppError::internal(e.to_string()))?;
    Ok(Sam3ModelHandle {
        image_model: Arc::new(image_model),
        tracker: Arc::new(tracker),
        device,
    })
}

#[async_trait]
impl ImageSegmenter for CandleSam3ImageSegmenter {
    async fn segment(
        &self,
        frame: &ImageFrame,
        prompts: &[PositivePointPrompt],
    ) -> Result<FrameMask, AppError> {
        let handle = self.handle.clone();
        let frame = frame.clone();
        let prompts = prompts.to_vec();
        tokio::task::spawn_blocking(move || run_image_inference(&handle, &frame, &prompts))
            .await
            .map_err(|e| AppError::internal(e.to_string()))?
    }
}

fn run_image_inference(
    handle: &Sam3ModelHandle,
    frame: &ImageFrame,
    prompts: &[PositivePointPrompt],
) -> Result<FrameMask, AppError> {
    let model = &*handle.image_model;
    let device = &handle.device;
    let config = model.config();

    let chw = frame_to_chw_tensor(frame, device)?;
    let preprocessed = normalize_for_sam3(
        &chw.unsqueeze(0).map_err(|e| AppError::internal(e.to_string()))?,
        config.image.image_size,
        &config.image.image_mean,
        &config.image.image_std,
        device,
    )?;

    let visual = model
        .encode_image_features(&preprocessed)
        .map_err(|e| AppError::internal(e.to_string()))?;

    let geometry = build_geometry_prompt(prompts, frame.width, frame.height, device)?;
    let geo_encoded = model
        .encode_geometry_prompt(&geometry, &visual)
        .map_err(|e| AppError::internal(e.to_string()))?;
    let fused = model
        .encode_fused_prompt(&visual, &geo_encoded)
        .map_err(|e| AppError::internal(e.to_string()))?;
    let decoder = model
        .decode_grounding(&fused, &geo_encoded)
        .map_err(|e| AppError::internal(e.to_string()))?;
    let segmentation = model
        .segment_grounding(&visual, &decoder, &fused, &geo_encoded)
        .map_err(|e| AppError::internal(e.to_string()))?;

    threshold_mask_logits_to_frame(
        &segmentation.mask_logits,
        frame.width,
        frame.height,
        config.image.image_size,
    )
}

#[async_trait]
impl VideoSegmenter for CandleSam3VideoSegmenter {
    async fn segment_video(
        &self,
        _frame_count: usize,
        _prompts: &[VideoFramePrompt],
    ) -> Result<Vec<FrameMask>, AppError> {
        Err(AppError::not_implemented(
            "Candle SAM3 video inference not yet wired — use branch rust/candle-sam3-video-inference",
        ))
    }
}
