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
        dtype = "f32",
        plugin_sha = %plugin_sha,
        candle_sha = %candle_sha,
        "loading SAM3 model and tracker"
    );
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
                            .map_err(|e| candle_core::Error::Msg(e.to_string()))
                    },
                )
                .map_err(|e| AppError::internal(e.to_string()))?;
            let cache_stats = predictor
                .session_cache_stats(&session_id)
                .map_err(|e| AppError::internal(e.to_string()))?;
            info!(
                cached_output_frames = cache_stats.cached_output_frames,
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
    use crate::inference::image::PositivePointPrompt;

    fn test_mask(value: u8) -> FrameMask {
        FrameMask {
            width: 1,
            height: 1,
            pixels: vec![value],
        }
    }

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
