use std::{path::Path, sync::Arc};

use async_trait::async_trait;
use candle_core::DType;
use candle_transformers::models::sam3;

use crate::{
    app_state::Sam3ModelHandle,
    error::AppError,
    imaging::tiff_io::ImageFrame,
    inference::{
        candle_sam3_helpers::{
            build_geometry_prompt, first_frame_dimensions, frame_to_chw_tensor, normalize_for_sam3,
            threshold_mask_logits_to_frame, video_frame_to_mask,
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
    model_name: String,
    checkpoint_path: &std::path::Path,
    device: candle_core::Device,
) -> Result<Sam3ModelHandle, AppError> {
    let config = sam3::Config::default();
    let source = sam3::Sam3CheckpointSource::upstream_pth(checkpoint_path);
    let image_model =
        sam3::Sam3ImageModel::from_checkpoint_source(&config, &source, DType::F32, &device)
            .map_err(|e| AppError::internal(e.to_string()))?;
    let tracker =
        sam3::Sam3TrackerModel::from_checkpoint_source(&config, &source, DType::F32, &device)
            .map_err(|e| AppError::internal(e.to_string()))?;
    Ok(Sam3ModelHandle {
        model_name,
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
        &chw.unsqueeze(0)
            .map_err(|e| AppError::internal(e.to_string()))?,
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
        frames_dir: &Path,
        prompts: &[VideoFramePrompt],
    ) -> Result<Vec<FrameMask>, AppError> {
        let handle = self.handle.clone();
        let frames_dir = frames_dir.to_path_buf();
        let prompts = prompts.to_vec();
        tokio::task::spawn_blocking(move || run_video_inference(&handle, &frames_dir, &prompts))
            .await
            .map_err(|e| AppError::internal(e.to_string()))?
    }
}

fn run_video_inference(
    handle: &Sam3ModelHandle,
    frames_dir: &Path,
    prompts: &[VideoFramePrompt],
) -> Result<Vec<FrameMask>, AppError> {
    let (frame_width, frame_height) = first_frame_dimensions(frames_dir)?;

    let source =
        sam3::VideoSource::from_path(frames_dir).map_err(|e| AppError::internal(e.to_string()))?;
    let session_options = low_memory_video_session_options();

    let model_ref = &*handle.image_model;
    let tracker_ref = &*handle.tracker;
    let device = &handle.device;

    let mut predictor = sam3::Sam3VideoPredictor::new(model_ref, tracker_ref, device);
    let session_id = predictor
        .start_session(source, session_options)
        .map_err(|e| AppError::internal(e.to_string()))?;
    predictor
        .reset_session(&session_id)
        .map_err(|e| AppError::internal(e.to_string()))?;

    let w = frame_width as f32;
    let h = frame_height as f32;
    for vfp in prompts {
        let session_prompt = sam3::SessionPrompt {
            text: None,
            points: if vfp.points.is_empty() {
                None
            } else {
                Some(
                    vfp.points
                        .iter()
                        .map(|p| ((p.x / w).clamp(0.0, 1.0), (p.y / h).clamp(0.0, 1.0)))
                        .collect(),
                )
            },
            point_labels: if vfp.points.is_empty() {
                None
            } else {
                Some(vec![1u32; vfp.points.len()])
            },
            boxes: None,
            box_labels: None,
        };
        predictor
            .add_prompt(
                &session_id,
                vfp.frame_index,
                session_prompt,
                None,
                true,
                false,
            )
            .map_err(|e| AppError::internal(e.to_string()))?;
    }

    let output = predictor
        .propagate_in_video(
            &session_id,
            sam3::PropagationOptions {
                direction: sam3::PropagationDirection::Forward,
                start_frame_idx: None,
                max_frame_num_to_track: None,
                output_prob_threshold: Some(0.5),
            },
        )
        .map_err(|e| AppError::internal(e.to_string()))?;

    output
        .frames
        .iter()
        .map(|frame| video_frame_to_mask(&frame.objects, frame_width, frame_height))
        .collect()
}

fn low_memory_video_session_options() -> sam3::VideoSessionOptions {
    sam3::VideoSessionOptions {
        tokenizer_path: None,
        memory_profile: sam3::VideoMemoryProfile::LowMemory,
        offload_frames_to_cpu: false,
        offload_state_to_cpu: false,
        prefetch_ahead: 0,
        prefetch_behind: 0,
        max_feature_cache_entries: 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_session_options_use_low_memory_no_prefetch_profile() {
        let options = low_memory_video_session_options();

        assert!(matches!(
            options.memory_profile,
            sam3::VideoMemoryProfile::LowMemory
        ));
        assert!(!options.offload_frames_to_cpu);
        assert!(!options.offload_state_to_cpu);
        assert_eq!(options.prefetch_ahead, 0);
        assert_eq!(options.prefetch_behind, 0);
        assert_eq!(options.max_feature_cache_entries, 2);
        assert!(options.tokenizer_path.is_none());
    }
}
