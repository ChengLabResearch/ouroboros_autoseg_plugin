use std::{path::Path, sync::Arc};

use async_trait::async_trait;
use candle_core::DType;
use candle_transformers::models::sam3;
use tracing::{info, warn};

use crate::{
    app_state::Sam3ModelHandle,
    error::AppError,
    imaging::tiff_io::ImageFrame,
    inference::{
        candle_sam3_frame_source::StagedJpegFrameSource,
        candle_sam3_helpers::{
            build_geometry_prompt, frame_to_chw_tensor, normalize_for_sam3,
            threshold_mask_logits_to_frame, video_frame_to_mask,
        },
        image::{FrameMask, ImageSegmenter, PositivePointPrompt},
        video::{FrameProgressCallback, VideoFramePrompt, VideoSegmenter},
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
    let compute_dtype = configured_compute_dtype()?;
    let device_kind = if device.is_cuda() { "cuda" } else { "cpu" };
    let cuda_ordinal = device
        .is_cuda()
        .then(|| configured_cuda_ordinal().to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let plugin_sha = option_env!("PLUGIN_GIT_SHA").unwrap_or("unknown");
    let candle_sha = option_env!("CANDLE_SAM3_GIT_SHA").unwrap_or("unknown");
    info!(
        model = %model_name,
        checkpoint = %checkpoint_path.display(),
        device = %device_kind,
        cuda_ordinal = %cuda_ordinal,
        compute_dtype = compute_dtype_name(compute_dtype),
        plugin_sha = %plugin_sha,
        candle_sha = %candle_sha,
        "loading SAM3 model and tracker"
    );
    let config = sam3::Config::default();
    let source = sam3::Sam3CheckpointSource::upstream_pth(checkpoint_path);
    let image_model =
        sam3::Sam3ImageModel::from_checkpoint_source(&config, &source, compute_dtype, &device)
            .map_err(|e| AppError::internal(e.to_string()))?;
    let mut tracker_config = sam3::Sam3TrackerConfig::from_sam3_config(&config);
    tracker_config.predictor.trim_past_non_cond_mem_for_eval = configured_trim_past_non_cond_mem()?;
    tracker_config.predictor.hotstart_delay = configured_hotstart_delay()?;
    info!(
        trim_past_non_cond_mem_for_eval = tracker_config.predictor.trim_past_non_cond_mem_for_eval,
        hotstart_delay = tracker_config.predictor.hotstart_delay,
        "configured SAM3 tracker retention controls"
    );
    let tracker = sam3::Sam3TrackerModel::new(
        &tracker_config,
        source
            .load_tracker_var_builder(compute_dtype, &device)
            .map_err(|e| AppError::internal(e.to_string()))?,
    )
    .map_err(|e| AppError::internal(e.to_string()))?;
    Ok(Sam3ModelHandle {
        model_name,
        image_model: Arc::new(image_model),
        tracker: Arc::new(tracker),
        device,
    })
}

pub fn configured_cuda_ordinal() -> usize {
    std::env::var("CUDA_DEVICE_ORDINAL")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
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
        progress: Option<FrameProgressCallback>,
    ) -> Result<Vec<FrameMask>, AppError> {
        let handle = self.handle.clone();
        let frames_dir = frames_dir.to_path_buf();
        let prompts = prompts.to_vec();
        tokio::task::spawn_blocking(move || {
            run_video_inference(&handle, &frames_dir, &prompts, progress.as_ref())
        })
        .await
        .map_err(|e| AppError::internal(e.to_string()))?
    }
}

fn run_video_inference(
    handle: &Sam3ModelHandle,
    frames_dir: &Path,
    prompts: &[VideoFramePrompt],
    progress: Option<&FrameProgressCallback>,
) -> Result<Vec<FrameMask>, AppError> {
    let session_config = configured_low_memory_video_session()?;
    let session_options = session_config.options.clone();
    info!(
        state_profile = session_config.state_profile.as_str(),
        offload_state_to_cpu = session_options.offload_state_to_cpu,
        offload_frames_to_cpu = session_options.offload_frames_to_cpu,
        prefetch_ahead = session_options.prefetch_ahead,
        prefetch_behind = session_options.prefetch_behind,
        max_feature_cache_entries = session_options.max_feature_cache_entries,
        max_non_cond_tracker_states = ?session_options.max_non_cond_tracker_states,
        retained_state_dtype = ?session_options.retained_state_dtype,
        "configured low-memory SAM3 video session"
    );

    let model_ref = &*handle.image_model;
    let tracker_ref = &*handle.tracker;
    let device = &handle.device;
    let config = model_ref.config();
    let source = StagedJpegFrameSource::new(
        frames_dir,
        config.image.image_size,
        config.image.image_mean,
        config.image.image_std,
    )
    .map_err(|e| AppError::internal(e.to_string()))?;
    let source_size = source.source_size();
    let (frame_width, frame_height) = (source_size.width, source_size.height);

    let mut predictor = sam3::Sam3VideoPredictor::new(model_ref, tracker_ref, device);
    let session_id = predictor
        .start_session_with_frame_source(Box::new(source), session_options)
        .map_err(|e| AppError::internal(e.to_string()))?;
    with_deterministic_session_close(
        &mut predictor,
        |predictor| {
            predictor
                .reset_session(&session_id)
                .map_err(|e| AppError::internal(e.to_string()))?;

            let w = frame_width as f32;
            let h = frame_height as f32;
            register_single_trajectory_prompts(prompts, |vfp, obj_id| {
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
                        obj_id,
                        true,
                        false,
                    )
                    .map_err(|e| AppError::internal(e.to_string()))
            })?;

            let frame_count = predictor
                .session_frame_count(&session_id)
                .map_err(|e| AppError::internal(e.to_string()))?;
            let mut masks = VideoMaskCollector::new(frame_count)?;
            predictor
                .propagate_in_video_stream(
                    &session_id,
                    sam3::PropagationOptions {
                        direction: sam3::PropagationDirection::Forward,
                        start_frame_idx: None,
                        max_frame_num_to_track: None,
                        output_prob_threshold: Some(0.5),
                    },
                    |frame| {
                        let mask = video_frame_to_mask(&frame.objects, frame_width, frame_height)
                            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                        masks
                            .push_forward(frame.frame_idx, mask)
                            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                        if let Some(callback) = progress {
                            callback(frame.frame_idx, frame_count);
                        }
                        Ok(())
                    },
                )
                .map_err(|e| AppError::internal(e.to_string()))?;
            let cache_stats = predictor
                .session_cache_stats(&session_id)
                .map_err(|e| AppError::internal(e.to_string()))?;
            info!(
                state_profile = session_config.state_profile.as_str(),
                loaded_frame_count = cache_stats.loaded_frame_count,
                cached_feature_entries = cache_stats.cached_feature_entries,
                cached_output_frames = cache_stats.cached_output_frames,
                tracked_objects = cache_stats.tracked_objects,
                retained_tracker_states = cache_stats.retained_tracker_states,
                retained_non_cond_tracker_states = cache_stats.retained_non_cond_tracker_states,
                retained_output_frame_indices = cache_stats.retained_output_frame_indices,
                retained_state_dtype = ?cache_stats.retained_state_dtype,
                postprocess_foreground_scalar_reads =
                    cache_stats.postprocess_foreground_scalar_reads,
                postprocess_score_scalar_reads = cache_stats.postprocess_score_scalar_reads,
                cpu_low_res_mask_bytes = cache_stats.cpu_low_res_mask_bytes,
                device_low_res_mask_bytes = cache_stats.device_low_res_mask_bytes,
                hotstart_buffered_frames = cache_stats.hotstart_buffered_frames,
                hotstart_buffered_cpu_bytes = cache_stats.hotstart_buffered_cpu_bytes,
                hotstart_buffered_device_bytes = cache_stats.hotstart_buffered_device_bytes,
                peak_hotstart_buffered_frames = cache_stats.peak_hotstart_buffered_frames,
                peak_hotstart_buffered_cpu_bytes = cache_stats.peak_hotstart_buffered_cpu_bytes,
                peak_hotstart_buffered_device_bytes =
                    cache_stats.peak_hotstart_buffered_device_bytes,
                cpu_total_bytes = cache_stats.cpu_bytes.total(),
                cpu_frame_bytes = cache_stats.cpu_bytes.frames,
                cpu_visual_feature_bytes = cache_stats.cpu_bytes.visual_features,
                cpu_tracker_state_bytes = cache_stats.cpu_bytes.tracker_states,
                cpu_packed_prompt_history_bytes = cache_stats.cpu_bytes.packed_prompt_history,
                cpu_text_cache_bytes = cache_stats.cpu_bytes.text_cache,
                cpu_cached_output_bytes = cache_stats.cpu_bytes.cached_outputs,
                device_total_bytes = cache_stats.device_bytes.total(),
                device_frame_bytes = cache_stats.device_bytes.frames,
                device_visual_feature_bytes = cache_stats.device_bytes.visual_features,
                device_tracker_state_bytes = cache_stats.device_bytes.tracker_states,
                device_packed_prompt_history_bytes = cache_stats.device_bytes.packed_prompt_history,
                device_text_cache_bytes = cache_stats.device_bytes.text_cache,
                device_cached_output_bytes = cache_stats.device_bytes.cached_outputs,
                "completed streamed SAM3 video propagation"
            );
            if cache_stats.cached_output_frames != 0 {
                return Err(AppError::upstream(format!(
                    "Low-memory video stream retained {} cached output frames after propagation",
                    cache_stats.cached_output_frames
                )));
            }
            masks.finish()
        },
        |predictor| {
            predictor
                .close_session(&session_id)
                .map_err(|e| AppError::internal(e.to_string()))
        },
    )
}

struct VideoMaskCollector {
    masks: Vec<Option<FrameMask>>,
    last_frame_idx: Option<usize>,
}

impl VideoMaskCollector {
    fn new(frame_count: usize) -> Result<Self, AppError> {
        if frame_count == 0 {
            return Err(AppError::bad_request(
                "Cannot collect streamed masks for an empty video",
            ));
        }
        Ok(Self {
            masks: vec![None; frame_count],
            last_frame_idx: None,
        })
    }

    fn push_forward(&mut self, frame_idx: usize, mask: FrameMask) -> Result<(), AppError> {
        if frame_idx >= self.masks.len() {
            return Err(AppError::upstream(format!(
                "Video stream emitted out-of-range frame {frame_idx} for {} frames",
                self.masks.len()
            )));
        }
        if self.masks[frame_idx].is_some() {
            return Err(AppError::upstream(format!(
                "Video stream emitted duplicate frame {frame_idx}"
            )));
        }
        if self.last_frame_idx.is_some_and(|last| frame_idx <= last) {
            return Err(AppError::upstream(format!(
                "Video stream emitted invalid forward order: frame {frame_idx} after {}",
                self.last_frame_idx.unwrap_or_default()
            )));
        }

        self.masks[frame_idx] = Some(mask);
        self.last_frame_idx = Some(frame_idx);
        Ok(())
    }

    #[cfg(test)]
    fn push_backward(&mut self, frame_idx: usize, mask: FrameMask) -> Result<(), AppError> {
        if frame_idx >= self.masks.len() {
            return Err(AppError::upstream(format!(
                "Video stream emitted out-of-range frame {frame_idx} for {} frames",
                self.masks.len()
            )));
        }
        if self.masks[frame_idx].is_some() {
            return Err(AppError::upstream(format!(
                "Video stream emitted duplicate frame {frame_idx}"
            )));
        }
        if self.last_frame_idx.is_some_and(|last| frame_idx >= last) {
            return Err(AppError::upstream(format!(
                "Video stream emitted invalid backward order: frame {frame_idx} after {}",
                self.last_frame_idx.unwrap_or_default()
            )));
        }

        self.masks[frame_idx] = Some(mask);
        self.last_frame_idx = Some(frame_idx);
        Ok(())
    }

    fn finish(self) -> Result<Vec<FrameMask>, AppError> {
        let missing = self
            .masks
            .iter()
            .enumerate()
            .filter_map(|(index, mask)| mask.is_none().then_some(index))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(AppError::upstream(format!(
                "Video stream omitted frame indices {missing:?}"
            )));
        }
        Ok(self.masks.into_iter().flatten().collect())
    }
}

fn with_deterministic_session_close<P, T>(
    predictor: &mut P,
    operation: impl FnOnce(&mut P) -> Result<T, AppError>,
    close: impl FnOnce(&mut P) -> Result<(), AppError>,
) -> Result<T, AppError> {
    let operation_result = operation(predictor);
    let close_result = close(predictor);
    match (operation_result, close_result) {
        (Ok(value), Ok(())) => Ok(value),
        (Ok(_), Err(close_error)) => Err(close_error),
        (Err(operation_error), Ok(())) => Err(operation_error),
        (Err(operation_error), Err(close_error)) => {
            warn!(%close_error, "failed to close video session after inference error");
            Err(operation_error)
        }
    }
}

fn register_single_trajectory_prompts(
    prompts: &[VideoFramePrompt],
    mut register: impl FnMut(&VideoFramePrompt, Option<u32>) -> Result<u32, AppError>,
) -> Result<u32, AppError> {
    let mut trajectory_id = None;

    for prompt in prompts {
        let returned_id = register(prompt, trajectory_id)?;
        match trajectory_id {
            Some(expected_id) if returned_id != expected_id => {
                return Err(AppError::upstream(format!(
                    "Biological single-trajectory prompt registration returned object {returned_id}, expected {expected_id}"
                )));
            }
            None => trajectory_id = Some(returned_id),
            Some(_) => {}
        }
    }

    trajectory_id.ok_or_else(|| {
        AppError::bad_request("No valid annotation frames were available for video inference")
    })
}

const VIDEO_STATE_PROFILE_ENV: &str = "SAM3_VIDEO_STATE_PROFILE";
const VIDEO_FEATURE_CACHE_ENTRIES_ENV: &str = "SAM3_VIDEO_FEATURE_CACHE_ENTRIES";
const TRACKER_TRIM_PAST_NON_COND_MEM_ENV: &str = "SAM3_TRACKER_TRIM_PAST_NON_COND_MEM";
const MAX_NON_COND_TRACKER_STATES_ENV: &str = "SAM3_MAX_NON_COND_TRACKER_STATES";
const VIDEO_HOTSTART_DELAY_ENV: &str = "SAM3_VIDEO_HOTSTART_DELAY";
const COMPUTE_DTYPE_ENV: &str = "SAM3_COMPUTE_DTYPE";
const RETAINED_STATE_DTYPE_ENV: &str = "SAM3_RETAINED_STATE_DTYPE";

fn configured_compute_dtype() -> Result<DType, AppError> {
    parse_compute_dtype(std::env::var(COMPUTE_DTYPE_ENV).ok().as_deref())
}

fn parse_compute_dtype(value: Option<&str>) -> Result<DType, AppError> {
    match value.unwrap_or("f32") {
        "f32" => Ok(DType::F32),
        "f16" => Ok(DType::F16),
        value => Err(AppError::bad_request(format!(
            "Invalid {COMPUTE_DTYPE_ENV}={value:?}; expected f32 or f16"
        ))),
    }
}

fn compute_dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        _ => "unsupported",
    }
}

fn parse_retained_state_dtype(value: Option<&str>) -> Result<sam3::RetainedStateDType, AppError> {
    match value.unwrap_or("bf16") {
        "f32" => Ok(sam3::RetainedStateDType::F32),
        "bf16" => Ok(sam3::RetainedStateDType::BF16),
        value => Err(AppError::bad_request(format!(
            "Invalid {RETAINED_STATE_DTYPE_ENV}={value:?}; expected f32 or bf16"
        ))),
    }
}

fn configured_hotstart_delay() -> Result<usize, AppError> {
    parse_hotstart_delay(std::env::var(VIDEO_HOTSTART_DELAY_ENV).ok().as_deref())
}

fn parse_hotstart_delay(value: Option<&str>) -> Result<usize, AppError> {
    value.unwrap_or("0").parse::<usize>().map_err(|_| {
        AppError::bad_request(format!(
            "Invalid {VIDEO_HOTSTART_DELAY_ENV}={:?}; expected a non-negative integer",
            value.unwrap_or("0")
        ))
    })
}

fn configured_trim_past_non_cond_mem() -> Result<bool, AppError> {
    parse_trim_past_non_cond_mem(
        std::env::var(TRACKER_TRIM_PAST_NON_COND_MEM_ENV)
            .ok()
            .as_deref(),
    )
}

fn parse_trim_past_non_cond_mem(value: Option<&str>) -> Result<bool, AppError> {
    match value.unwrap_or("true") {
        "true" => Ok(true),
        "false" => Ok(false),
        value => Err(AppError::bad_request(format!(
            "Invalid {TRACKER_TRIM_PAST_NON_COND_MEM_ENV}={value:?}; expected true or false"
        ))),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VideoStateProfile {
    GpuResident,
    CpuOffload,
}

impl VideoStateProfile {
    fn as_str(self) -> &'static str {
        match self {
            Self::GpuResident => "gpu-resident",
            Self::CpuOffload => "cpu-offload",
        }
    }
}

#[derive(Debug, Clone)]
struct LowMemoryVideoSessionConfig {
    state_profile: VideoStateProfile,
    options: sam3::VideoSessionOptions,
}

fn configured_low_memory_video_session() -> Result<LowMemoryVideoSessionConfig, AppError> {
    low_memory_video_session_config(
        std::env::var(VIDEO_STATE_PROFILE_ENV).ok().as_deref(),
        std::env::var(VIDEO_FEATURE_CACHE_ENTRIES_ENV)
            .ok()
            .as_deref(),
        std::env::var(MAX_NON_COND_TRACKER_STATES_ENV)
            .ok()
            .as_deref(),
        std::env::var(RETAINED_STATE_DTYPE_ENV).ok().as_deref(),
    )
}

fn low_memory_video_session_config(
    state_profile: Option<&str>,
    feature_cache_entries: Option<&str>,
    max_non_cond_tracker_states: Option<&str>,
    retained_state_dtype: Option<&str>,
) -> Result<LowMemoryVideoSessionConfig, AppError> {
    let state_profile = match state_profile.unwrap_or("cpu-offload") {
        "gpu-resident" => VideoStateProfile::GpuResident,
        "cpu-offload" => VideoStateProfile::CpuOffload,
        value => {
            return Err(AppError::bad_request(format!(
                "Invalid {VIDEO_STATE_PROFILE_ENV}={value:?}; expected gpu-resident or cpu-offload"
            )))
        }
    };
    let max_feature_cache_entries = feature_cache_entries
        .unwrap_or("1")
        .parse::<usize>()
        .map_err(|_| {
            AppError::bad_request(format!(
                "Invalid {VIDEO_FEATURE_CACHE_ENTRIES_ENV}; expected 1 or 2"
            ))
        })?;
    if !matches!(max_feature_cache_entries, 1 | 2) {
        return Err(AppError::bad_request(format!(
            "Invalid {VIDEO_FEATURE_CACHE_ENTRIES_ENV}={max_feature_cache_entries}; expected 1 or 2"
        )));
    }
    let max_non_cond_tracker_states = match max_non_cond_tracker_states {
        None | Some("") => None,
        Some(value) => {
            let parsed = value.parse::<usize>().map_err(|_| {
                AppError::bad_request(format!(
                    "Invalid {MAX_NON_COND_TRACKER_STATES_ENV}={value:?}; expected a positive integer or empty"
                ))
            })?;
            if parsed == 0 {
                return Err(AppError::bad_request(format!(
                    "Invalid {MAX_NON_COND_TRACKER_STATES_ENV}=0; expected a positive integer or empty"
                )));
            }
            Some(parsed)
        }
    };

    Ok(LowMemoryVideoSessionConfig {
        state_profile,
        options: sam3::VideoSessionOptions {
            tokenizer_path: None,
            memory_profile: sam3::VideoMemoryProfile::LowMemory,
            offload_frames_to_cpu: false,
            offload_state_to_cpu: matches!(state_profile, VideoStateProfile::CpuOffload),
            retained_state_dtype: parse_retained_state_dtype(retained_state_dtype)?,
            prefetch_ahead: 0,
            prefetch_behind: 0,
            max_feature_cache_entries,
            max_non_cond_tracker_states,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::image::PositivePointPrompt;
    use candle_core::Device;
    use candle_transformers::models::sam3::FrameSource;
    use image::{ImageBuffer, Rgb};

    fn test_mask(value: u8) -> FrameMask {
        FrameMask {
            width: 1,
            height: 1,
            pixels: vec![value],
        }
    }

    fn write_test_jpeg(path: &Path, color: [u8; 3], width: u32, height: u32) {
        ImageBuffer::from_pixel(width, height, Rgb(color))
            .save(path)
            .expect("write staged JPEG");
    }

    #[test]
    fn video_session_options_default_to_cpu_offload_and_one_feature() {
        let config =
            low_memory_video_session_config(None, None, None, None).expect("default profile");
        let options = config.options;

        assert_eq!(config.state_profile, VideoStateProfile::CpuOffload);
        assert!(matches!(
            options.memory_profile,
            sam3::VideoMemoryProfile::LowMemory
        ));
        assert!(!options.offload_frames_to_cpu);
        assert!(options.offload_state_to_cpu);
        assert_eq!(options.prefetch_ahead, 0);
        assert_eq!(options.prefetch_behind, 0);
        assert_eq!(options.max_feature_cache_entries, 1);
        assert_eq!(options.max_non_cond_tracker_states, None);
        assert_eq!(options.retained_state_dtype, sam3::RetainedStateDType::BF16);
        assert!(options.tokenizer_path.is_none());
    }

    #[test]
    fn video_session_options_expose_benchmark_variants_and_reject_ambiguity() {
        let gpu = low_memory_video_session_config(
            Some("gpu-resident"),
            Some("2"),
            Some("32"),
            Some("f32"),
        )
        .expect("variant B");
        assert_eq!(gpu.state_profile, VideoStateProfile::GpuResident);
        assert!(!gpu.options.offload_state_to_cpu);
        assert_eq!(gpu.options.max_feature_cache_entries, 2);
        assert_eq!(gpu.options.max_non_cond_tracker_states, Some(32));
        assert_eq!(
            gpu.options.retained_state_dtype,
            sam3::RetainedStateDType::F32
        );

        let cpu =
            low_memory_video_session_config(Some("cpu-offload"), Some("1"), None, Some("bf16"))
                .expect("variant C");
        assert_eq!(cpu.state_profile, VideoStateProfile::CpuOffload);
        assert!(cpu.options.offload_state_to_cpu);

        assert!(low_memory_video_session_config(Some("auto"), None, None, None).is_err());
        assert!(low_memory_video_session_config(None, Some("0"), None, None).is_err());
        assert!(low_memory_video_session_config(None, Some("3"), None, None).is_err());
        assert!(low_memory_video_session_config(None, Some("many"), None, None).is_err());
        assert!(low_memory_video_session_config(None, None, Some("many"), None).is_err());
        assert!(low_memory_video_session_config(None, None, Some("0"), None).is_err());
        assert!(low_memory_video_session_config(None, None, None, Some("f16")).is_err());
    }

    #[test]
    fn compute_and_retained_storage_dtypes_are_independent_and_validated() {
        assert_eq!(
            parse_compute_dtype(None).expect("default compute"),
            DType::F32
        );
        assert_eq!(
            parse_compute_dtype(Some("f16")).expect("f16 compute"),
            DType::F16
        );
        assert!(parse_compute_dtype(Some("bf16")).is_err());

        assert_eq!(
            parse_retained_state_dtype(None).expect("default retained"),
            sam3::RetainedStateDType::BF16
        );
        assert_eq!(
            parse_retained_state_dtype(Some("f32")).expect("f32 retained"),
            sam3::RetainedStateDType::F32
        );
        assert!(parse_retained_state_dtype(Some("f16")).is_err());
    }

    #[test]
    fn tracker_trim_control_defaults_on_and_supports_an_untrimmed_reference() {
        assert!(parse_trim_past_non_cond_mem(None).expect("default trim"));
        assert!(parse_trim_past_non_cond_mem(Some("true")).expect("trimmed control"));
        assert!(!parse_trim_past_non_cond_mem(Some("false")).expect("untrimmed control"));
        assert!(parse_trim_past_non_cond_mem(Some("yes")).is_err());
    }

    #[test]
    fn hotstart_delay_defaults_off_and_supports_a_bounded_certification_control() {
        assert_eq!(parse_hotstart_delay(None).expect("default delay"), 0);
        assert_eq!(parse_hotstart_delay(Some("4")).expect("bounded delay"), 4);
        assert!(parse_hotstart_delay(Some("many")).is_err());
        assert!(parse_hotstart_delay(Some("-1")).is_err());
    }

    #[test]
    fn video_stream_matches_collected_masks_for_forward_and_backward_callbacks() {
        let mut forward = VideoMaskCollector::new(3).expect("collector");
        for (frame_idx, value) in [(0, 10), (1, 20), (2, 30)] {
            forward
                .push_forward(frame_idx, test_mask(value))
                .expect("forward callback");
        }
        let forward = forward.finish().expect("complete forward stream");
        assert_eq!(
            forward
                .iter()
                .map(|mask| mask.pixels[0])
                .collect::<Vec<_>>(),
            vec![10, 20, 30]
        );

        let mut backward = VideoMaskCollector::new(3).expect("collector");
        // These callbacks may arrive in a burst, but retain directional order.
        for (frame_idx, value) in [(2, 30), (1, 20), (0, 10)] {
            backward
                .push_backward(frame_idx, test_mask(value))
                .expect("backward callback");
        }
        let backward = backward.finish().expect("complete backward stream");
        assert_eq!(
            backward
                .iter()
                .map(|mask| mask.pixels[0])
                .collect::<Vec<_>>(),
            vec![10, 20, 30]
        );
    }

    #[test]
    fn video_stream_accepts_delayed_prompt_frames_and_burst_callbacks() {
        let mut forward = VideoMaskCollector::new(5).expect("collector");
        let forward_batches = [vec![], vec![], vec![0], vec![1], vec![2, 3, 4]];
        for batch in forward_batches {
            for frame_idx in batch {
                forward
                    .push_forward(frame_idx, test_mask(frame_idx as u8))
                    .expect("delayed forward callback");
            }
        }
        let forward = forward.finish().expect("complete delayed forward stream");
        assert_eq!(
            forward
                .iter()
                .map(|mask| mask.pixels[0])
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );

        let mut backward = VideoMaskCollector::new(5).expect("collector");
        let backward_batches = [vec![], vec![], vec![4], vec![3], vec![2, 1, 0]];
        for batch in backward_batches {
            for frame_idx in batch {
                backward
                    .push_backward(frame_idx, test_mask(frame_idx as u8))
                    .expect("delayed backward callback");
            }
        }
        let backward = backward.finish().expect("complete delayed backward stream");
        assert_eq!(
            backward
                .iter()
                .map(|mask| mask.pixels[0])
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn video_output_validation_rejects_missing_duplicate_out_of_range_and_invalid_order() {
        let mut missing = VideoMaskCollector::new(3).expect("collector");
        missing.push_forward(0, test_mask(0)).expect("frame 0");
        missing.push_forward(2, test_mask(2)).expect("frame 2");
        assert!(missing
            .finish()
            .expect_err("missing frame must fail")
            .to_string()
            .contains("[1]"));

        let mut duplicate = VideoMaskCollector::new(2).expect("collector");
        duplicate.push_forward(0, test_mask(0)).expect("frame 0");
        assert!(duplicate
            .push_forward(0, test_mask(0))
            .expect_err("duplicate frame must fail")
            .to_string()
            .contains("duplicate frame 0"));

        let mut out_of_range = VideoMaskCollector::new(2).expect("collector");
        assert!(out_of_range
            .push_forward(2, test_mask(2))
            .expect_err("out-of-range frame must fail")
            .to_string()
            .contains("out-of-range frame 2"));

        let mut invalid_order = VideoMaskCollector::new(3).expect("collector");
        invalid_order
            .push_forward(1, test_mask(1))
            .expect("frame 1");
        assert!(invalid_order
            .push_forward(0, test_mask(0))
            .expect_err("reverse callback in forward stream must fail")
            .to_string()
            .contains("invalid forward order"));
    }

    #[test]
    fn video_stream_closes_session_after_success_and_conversion_failure() {
        #[derive(Default)]
        struct FakePredictor {
            operated: bool,
            closed: bool,
        }

        let mut success = FakePredictor::default();
        let value = with_deterministic_session_close(
            &mut success,
            |predictor| {
                predictor.operated = true;
                Ok(42)
            },
            |predictor| {
                predictor.closed = true;
                Ok(())
            },
        )
        .expect("successful operation and close");
        assert_eq!(value, 42);
        assert!(success.operated);
        assert!(success.closed);

        let mut failure = FakePredictor::default();
        let error = with_deterministic_session_close(
            &mut failure,
            |predictor| {
                predictor.operated = true;
                Err::<(), _>(AppError::internal("injected callback conversion failure"))
            },
            |predictor| {
                predictor.closed = true;
                Ok(())
            },
        )
        .expect_err("conversion failure must propagate");
        assert!(error
            .to_string()
            .contains("injected callback conversion failure"));
        assert!(failure.operated);
        assert!(failure.closed);

        let mut double_failure = FakePredictor::default();
        let error = with_deterministic_session_close(
            &mut double_failure,
            |predictor| {
                predictor.operated = true;
                Err::<(), _>(AppError::internal("original inference failure"))
            },
            |predictor| {
                predictor.closed = true;
                Err(AppError::internal("secondary close failure"))
            },
        )
        .expect_err("the original operation failure must be retained");
        assert!(error.to_string().contains("original inference failure"));
        assert!(!error.to_string().contains("secondary close failure"));
        assert!(double_failure.closed);
    }

    #[test]
    fn caller_owned_staged_source_closes_after_success_and_source_failure() {
        struct FakeSessionOwner {
            source: Box<dyn FrameSource>,
            closed: bool,
            loaded_after_close: Option<usize>,
            inject_close_error: bool,
        }

        impl FakeSessionOwner {
            fn load(&mut self, frame_idx: usize) -> Result<(), AppError> {
                self.source
                    .get_frame(frame_idx, &Device::Cpu)
                    .map(|_| ())
                    .map_err(|error| AppError::internal(error.to_string()))
            }

            fn close(&mut self) -> Result<(), AppError> {
                self.source.close();
                self.loaded_after_close = Some(self.source.loaded_frame_count());
                self.closed = true;
                if self.inject_close_error {
                    Err(AppError::internal("injected close failure"))
                } else {
                    Ok(())
                }
            }
        }

        let success_dir = tempfile::tempdir().expect("success fixture");
        write_test_jpeg(&success_dir.path().join("0.jpg"), [10, 20, 30], 3, 2);
        let success_source = StagedJpegFrameSource::new(success_dir.path(), 4, [0.0; 3], [1.0; 3])
            .expect("success source");
        let mut success = FakeSessionOwner {
            source: Box::new(success_source),
            closed: false,
            loaded_after_close: None,
            inject_close_error: false,
        };
        with_deterministic_session_close(
            &mut success,
            |owner| {
                owner.load(0)?;
                assert_eq!(owner.source.loaded_frame_count(), 1);
                Ok(())
            },
            FakeSessionOwner::close,
        )
        .expect("successful source session");
        assert!(success.closed);
        assert_eq!(success.loaded_after_close, Some(0));

        let failure_dir = tempfile::tempdir().expect("failure fixture");
        write_test_jpeg(&failure_dir.path().join("0.jpg"), [1, 2, 3], 3, 2);
        write_test_jpeg(&failure_dir.path().join("1.jpg"), [4, 5, 6], 4, 2);
        let failure_source = StagedJpegFrameSource::new(failure_dir.path(), 4, [0.0; 3], [1.0; 3])
            .expect("failure source");
        let mut failure = FakeSessionOwner {
            source: Box::new(failure_source),
            closed: false,
            loaded_after_close: None,
            inject_close_error: true,
        };
        let error = with_deterministic_session_close(
            &mut failure,
            |owner| {
                owner.load(0)?;
                owner.load(1)
            },
            FakeSessionOwner::close,
        )
        .expect_err("source geometry failure must propagate");
        assert!(error.to_string().contains("session expects 2x3"));
        assert!(!error.to_string().contains("injected close failure"));
        assert!(failure.closed);
        assert_eq!(failure.loaded_after_close, Some(0));
    }

    #[test]
    fn video_prompt_registration_reuses_one_object_for_fifty_annotations() {
        let prompts = (0..50usize)
            .map(|annotation_index| VideoFramePrompt {
                // Representative z distribution whose independent-object workload is
                // the measured 142,384 remaining object-frame evaluations.
                frame_index: match annotation_index {
                    0 => 0,
                    1..=46 => annotation_index * 84 - 5,
                    47 => annotation_index * 84 - 4,
                    _ => annotation_index * 84,
                },
                points: vec![PositivePointPrompt { x: 10.0, y: 20.0 }],
            })
            .collect::<Vec<_>>();
        let mut requested_ids = Vec::new();

        let object_id = register_single_trajectory_prompts(&prompts, |_, requested_id| {
            requested_ids.push(requested_id);
            Ok(7)
        })
        .expect("single trajectory registers");

        assert_eq!(object_id, 7);
        assert_eq!(requested_ids[0], None);
        assert!(requested_ids[1..].iter().all(|id| *id == Some(7)));
        let independent_object_workload = prompts
            .iter()
            .map(|prompt| 4_901 - prompt.frame_index)
            .sum::<usize>();
        assert_eq!(independent_object_workload, 142_384);
        assert_eq!(4_901 * 1, 4_901, "one object-frame evaluation per frame");
    }

    #[test]
    fn video_prompt_registration_preserves_grouped_points() {
        let prompts = vec![VideoFramePrompt {
            frame_index: 12,
            points: vec![
                PositivePointPrompt { x: 1.0, y: 2.0 },
                PositivePointPrompt { x: 3.0, y: 4.0 },
            ],
        }];

        register_single_trajectory_prompts(&prompts, |prompt, requested_id| {
            assert_eq!(requested_id, None);
            assert_eq!(prompt.frame_index, 12);
            assert_eq!(prompt.points.len(), 2);
            Ok(3)
        })
        .expect("grouped prompt registers");
    }

    #[test]
    fn video_prompt_registration_rejects_empty_prompts_and_object_fanout() {
        let empty_error = register_single_trajectory_prompts(&[], |_, _| Ok(1))
            .expect_err("empty prompt list must fail");
        assert!(empty_error
            .to_string()
            .contains("No valid annotation frames"));

        let prompts = vec![
            VideoFramePrompt {
                frame_index: 0,
                points: vec![PositivePointPrompt { x: 1.0, y: 2.0 }],
            },
            VideoFramePrompt {
                frame_index: 1,
                points: vec![PositivePointPrompt { x: 2.0, y: 3.0 }],
            },
        ];
        let mut calls = 0;
        let fanout_error = register_single_trajectory_prompts(&prompts, |_, requested_id| {
            calls += 1;
            if calls == 1 {
                assert_eq!(requested_id, None);
                Ok(10)
            } else {
                assert_eq!(requested_id, Some(10));
                Ok(11)
            }
        })
        .expect_err("object fan-out must fail");
        assert!(fanout_error.to_string().contains("returned object 11"));
        assert!(fanout_error.to_string().contains("expected 10"));
    }
}
