use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use crate::{
    app_state::{AppState, Sam3ModelHandle},
    domain::{
        jobs::JobPhase,
        paths::{parse_mixed_path, MixedPathParts},
        requests::ProcessRequest,
    },
    error::AppError,
    imaging::{
        annotations::{
            annotation_samples_for_image, annotation_samples_for_video, default_annotations,
            fill_prompt_gaps, AnnotationPoint, VolumeShape,
        },
        cleanup::{
            keep_centerline_components, repair_classified_frames, repair_inconsistent_frames,
            CleanupStats, FrameRepairReport,
        },
        input_window::{InputWindowController, InputWindowReport, SharedInputWindow},
        output::{
            write_annotation_overlay_stack, write_mask_stack, write_mask_stack_with_progress,
        },
        preprocess::{stage_input_frames_with_progress, PreparedFrame, PreparedFrameEncoding},
        tiff_io::{inspect_volume, read_volume_frames, VolumeInput},
    },
    inference::{
        candle_sam3::{load_sam3_handle, CandleSam3ImageSegmenter, CandleSam3VideoSegmenter},
        image::{FrameMask, ImageSegmenter},
        video::{FrameProgressCallback, VideoFramePrompt, VideoSegmenter},
        SegmentationOutput,
    },
    services::{
        progress::{report_frame, report_phase, ProgressSender},
        volume::{self, VolumeFileMapping},
    },
};

#[derive(Debug, Clone)]
pub struct PipelinePlan {
    pub host_source: String,
    pub host_output: MixedPathParts,
    pub volume_source: PathBuf,
    pub temp_volume_dir: PathBuf,
    pub volume_output: PathBuf,
}

#[derive(Debug, Clone)]
pub struct PreparedPipelineInput {
    pub plan: PipelinePlan,
    pub volume: VolumeInput,
    /// Sole trace geometry for per-frame component selection in both cleanup steps.
    pub annotation_points: Arc<[AnnotationPoint]>,
    /// The sole prepared prompt list consumed by inference and prompt overlays.
    pub prompts: Arc<[VideoFramePrompt]>,
    pub staged_frames: Vec<PreparedFrame>,
}

impl PipelinePlan {
    pub fn copy_to_volume_mappings(&self) -> Vec<VolumeFileMapping> {
        vec![VolumeFileMapping::new(self.host_source.clone(), "")]
    }

    pub fn copy_to_host_mappings(&self, request: &ProcessRequest) -> Vec<VolumeFileMapping> {
        let mut files = vec![VolumeFileMapping::new(
            self.host_output.raw.clone(),
            "Segmentation",
        )];
        let mut artifacts = Vec::new();
        if request.overlay_annotation_points {
            artifacts.push(annotation_overlay_path(&self.volume_output));
        }
        if request.repair_inconsistent_frames || request.input_window.is_some() {
            artifacts.push(frame_repair_report_path(&self.volume_output));
            if request.repair_inconsistent_frames && request.keep_unrepaired {
                artifacts.push(unrepaired_mask_path(&self.volume_output));
            }
        }
        for artifact in artifacts {
            let file_name = artifact
                .file_name()
                .and_then(|name| name.to_str())
                .expect("artifact filename derives from a UTF-8 request path");
            files.push(VolumeFileMapping::new(
                self.host_output.with_file_name(file_name),
                "Segmentation",
            ));
        }
        files
    }
}

pub fn plan(state: &AppState, request: &ProcessRequest) -> Result<PipelinePlan, AppError> {
    let input = parse_mixed_path(&request.file_path)?;
    let output = parse_mixed_path(&request.output_file)?;
    let plugin_root = state.config().plugin_root();

    Ok(PipelinePlan {
        host_source: input.raw,
        volume_source: plugin_root.join(&input.file_name),
        temp_volume_dir: plugin_root.join(format!("{}_temp", input.stem)),
        volume_output: plugin_root.join("Segmentation").join(&output.file_name),
        host_output: output,
    })
}

pub async fn prepare(
    state: &AppState,
    request: &ProcessRequest,
) -> Result<PreparedPipelineInput, AppError> {
    request.validate()?;

    let plan = plan(state, request)?;
    volume::copy_to_volume(
        state.http_client(),
        state.config(),
        &plan.copy_to_volume_mappings(),
    )
    .await?;
    let mut volume = inspect_volume(&plan.volume_source).await?;
    let frames = read_volume_frames(&plan.volume_source).await?;
    let mut annotation_points = volume.annotation_points.take().unwrap_or_else(|| {
        default_annotations(
            VolumeShape {
                frames: volume.geometry.frames,
                height: volume.geometry.height,
                width: volume.geometry.width,
            },
            state.config().fallback_annotation_interval,
        )
    });
    if request.predictor_type == "VideoPredictor" {
        let added_prompts = fill_prompt_gaps(
            &mut annotation_points,
            volume.geometry.frames,
            request.max_prompt_gap,
        );
        tracing::info!(
            file_path = %request.file_path,
            max_prompt_gap = ?request.max_prompt_gap,
            added_prompts,
            "Prepared video annotation prompts"
        );
    }
    let prompts = match request.predictor_type.as_str() {
        "ImagePredictor" => annotation_samples_for_image(&annotation_points, frames.len()),
        "VideoPredictor" => annotation_samples_for_video(&annotation_points, frames.len()),
        _ => unreachable!("validated by ProcessRequest::validate"),
    };
    let frame_name_width = frame_name_width(frames.len());
    let encoding = stage_encoding(&request.predictor_type)?;
    let staged_frames = frames
        .into_iter()
        .enumerate()
        .map(|(index, frame)| PreparedFrame {
            frame,
            target: plan
                .temp_volume_dir
                .join(format!("{index:0frame_name_width$}")),
            encoding,
        })
        .collect();

    Ok(PreparedPipelineInput {
        plan,
        volume,
        annotation_points: annotation_points.into(),
        prompts: prompts.into(),
        staged_frames,
    })
}

pub async fn run(
    state: &AppState,
    request: &ProcessRequest,
    progress: &ProgressSender,
) -> Result<(), AppError> {
    request.validate()?;

    let handle = model_handle_for_request(state, &request.model_type).await?;

    let prepared = prepare(state, request).await?;
    if prepared.plan.temp_volume_dir.exists() {
        tokio::fs::remove_dir_all(&prepared.plan.temp_volume_dir).await?;
    }
    tokio::fs::create_dir_all(&prepared.plan.temp_volume_dir).await?;
    let result = run_prepared(state, &handle, &prepared, request, progress).await;
    let cleanup = tokio::fs::remove_dir_all(&prepared.plan.temp_volume_dir).await;
    match (result, cleanup) {
        (Err(error), _) => Err(error),
        (Ok(()), Err(error)) => Err(error.into()),
        (Ok(()), Ok(())) => Ok(()),
    }
}

async fn run_prepared(
    state: &AppState,
    handle: &Arc<Sam3ModelHandle>,
    prepared: &PreparedPipelineInput,
    request: &ProcessRequest,
    progress: &ProgressSender,
) -> Result<(), AppError> {
    stage_input_frames_with_progress(&prepared.staged_frames, |frame, total| {
        report_frame(progress, JobPhase::Transferring, frame, total)
    })
    .await?;
    report_phase(progress, JobPhase::Inference);

    let input_window = request
        .input_window
        .map(|options| {
            InputWindowController::new(
                options,
                prepared.annotation_points.clone(),
                prepared.prompts.clone(),
                request.keep_centerline_component || request.repair_inconsistent_frames,
            )
            .map(|controller| Arc::new(Mutex::new(controller)))
        })
        .transpose()?;
    let mut output = match request.predictor_type.as_str() {
        "ImagePredictor" => run_image_predictor(handle, prepared, input_window, progress).await?,
        "VideoPredictor" => run_video_predictor(handle, prepared, input_window, progress).await?,
        _ => unreachable!("validated by ProcessRequest::validate"),
    };

    save_masks(
        state,
        prepared,
        request,
        &mut output.masks,
        output.input_window.as_mut(),
        progress,
    )
    .await
}

async fn save_masks(
    state: &AppState,
    prepared: &PreparedPipelineInput,
    request: &ProcessRequest,
    masks: &mut [FrameMask],
    mut input_window: Option<&mut InputWindowReport>,
    progress: &ProgressSender,
) -> Result<(), AppError> {
    report_phase(progress, JobPhase::Saving);
    let mut classifications = None;
    if request.keep_centerline_component {
        let stats = if let Some(window) = input_window.as_deref_mut() {
            let (report, stats) = consume_window_classifications(window, masks)?;
            classifications = Some(report);
            stats
        } else {
            keep_centerline_components(masks, &prepared.annotation_points)?
        };
        tracing::info!(
            output = %request.output_file,
            frames_changed = stats.frames_changed,
            pixels_removed = stats.pixels_removed,
            seed_misses = stats.seed_misses,
            frames_without_seed = stats.frames_without_seed,
            "centre-line component cleanup complete"
        );
    }
    if let Some(parent) = prepared.plan.volume_output.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let mut document = serde_json::json!({});
    let repair_report = if request.repair_inconsistent_frames {
        if request.keep_unrepaired {
            write_mask_stack(&unrepaired_mask_path(&prepared.plan.volume_output), masks).await?;
        }
        if classifications.is_none() {
            if let Some(window) = input_window.as_deref_mut() {
                classifications = Some(consume_window_classifications(window, masks)?.0);
            }
        }
        let report = match classifications.take() {
            Some(report) => repair_classified_frames(masks, report)?,
            None => repair_inconsistent_frames(masks, &prepared.annotation_points)?,
        };
        document = serde_json::to_value(&report)?;
        document["schema_version"] = 2.into();
        document["options"] = serde_json::json!({
            "repair_inconsistent_frames": request.repair_inconsistent_frames,
            "keep_unrepaired": request.keep_unrepaired,
        });
        document["summary"] = serde_json::to_value(report.summary())?;
        // Filter using runtime decisions, then log the same indexed output records
        // as the sidecar. Serialized values never drive repair or overlay behavior.
        let flagged_frames: Vec<_> = report
            .frames
            .iter()
            .enumerate()
            .filter(|(_, frame)| frame.is_flagged())
            .map(|(index, _)| &document["frames"][index])
            .collect();
        let recovered_anchors: Vec<_> = report
            .frames
            .iter()
            .enumerate()
            .filter(|(_, frame)| frame.anchor_recovered)
            .map(|(index, _)| &document["frames"][index])
            .collect();
        tracing::info!(
            output = %request.output_file,
            options = %document["options"],
            summary = %document["summary"],
            flagged_frames = %serde_json::to_string(&flagged_frames)?,
            recovered_anchors = %serde_json::to_string(&recovered_anchors)?,
            "inconsistent-frame repair complete"
        );
        Some(report)
    } else {
        if let Some(report) = &classifications {
            // Centre-line-only jobs still retain the original detector records;
            // ownership has moved out of the per-window records, not been copied.
            document = serde_json::to_value(report)?;
        }
        classifications
    };
    let input_window = input_window.as_deref();
    if let Some(report) = input_window {
        let summary = report.summary();
        tracing::info!(
            output = %request.output_file,
            resets = summary.resets,
            flag_resets = summary.flag_resets,
            prompt_resets = summary.prompt_resets,
            border_frames = summary.border_frames,
            border_fraction = summary.border_fraction,
            "input window feedback complete"
        );
        document["schema_version"] = 3.into();
        document["options"] = serde_json::json!({
            "keep_centerline_component": request.keep_centerline_component,
            "repair_inconsistent_frames": request.repair_inconsistent_frames,
            "keep_unrepaired": request.keep_unrepaired,
            "input_window": request.input_window,
        });
        document["input_window"] = serde_json::to_value(report)?;
        document["input_window"]["summary"] = serde_json::to_value(summary)?;
    }
    if repair_report.is_some() || input_window.is_some() {
        tokio::fs::write(
            frame_repair_report_path(&prepared.plan.volume_output),
            serde_json::to_vec_pretty(&document)?,
        )
        .await?;
    }
    write_mask_stack_with_progress(&prepared.plan.volume_output, masks, |frame, total| {
        report_frame(progress, JobPhase::Saving, frame, total)
    })
    .await?;
    if request.overlay_annotation_points {
        write_annotation_overlay_stack(
            &annotation_overlay_path(&prepared.plan.volume_output),
            masks,
            &prepared.prompts,
            request.annotation_overlay_intensity,
            repair_report.as_ref(),
            input_window,
        )
        .await?;
    }
    volume::copy_to_host(
        state.http_client(),
        state.config(),
        &prepared.plan.copy_to_host_mappings(request),
    )
    .await?;
    Ok(())
}

/// Move the online selections into the post-pass without selecting or measuring
/// the same components again. The mask stack becomes the selected-output owner.
fn consume_window_classifications(
    report: &mut InputWindowReport,
    masks: &mut [FrameMask],
) -> Result<(FrameRepairReport, CleanupStats), AppError> {
    if report.frames.len() != masks.len() {
        return Err(AppError::internal(
            "Input window classifications do not match the output frame count",
        ));
    }
    // Released predecessor pixels cannot support postprocessing. Check before
    // consuming any records so a retention mistake cannot silently skip cleanup.
    for (frame, mask) in report.frames.iter().zip(masks.iter()) {
        let retained = frame.raw_detection.as_ref().is_some_and(|classified| {
            classified.component.width == mask.width
                && classified.component.height == mask.height
                && classified.component.pixels.len() == mask.pixels.len()
        });
        if !retained {
            return Err(AppError::internal(
                "Input window did not retain the classified components required for postprocessing",
            ));
        }
    }
    let mut classifications = FrameRepairReport {
        frames: Vec::with_capacity(masks.len()),
    };
    let mut stats = CleanupStats::default();
    for (frame, mask) in report.frames.iter_mut().zip(masks) {
        let (record, frame_stats) = frame
            .raw_detection
            .take()
            .expect("validated retained classification")
            .keep_component(mask);
        classifications.frames.push(record);
        stats += frame_stats;
    }
    Ok((classifications, stats))
}

async fn model_handle_for_request(
    state: &AppState,
    model_type: &str,
) -> Result<Arc<Sam3ModelHandle>, AppError> {
    let descriptor = state
        .config()
        .model_descriptor(model_type)
        .ok_or_else(|| AppError::bad_request(format!("Unknown model type: {model_type}")))?;
    if !matches!(descriptor.model_name, "sam3" | "medical_sam3") {
        return Err(AppError::bad_request(format!(
            "Unsupported SAM3 model type: {model_type}"
        )));
    }
    if let Some(handle) = state.sam3_handle(descriptor.model_name).await {
        return Ok(handle);
    }

    let checkpoint_path = state.config().checkpoint_path(descriptor.model_name);
    if !checkpoint_path.exists() {
        return Err(AppError::not_found(format!(
            "SAM3 checkpoint missing — download {} first",
            descriptor.checkpoint_file
        )));
    }

    let model_name = descriptor.model_name.to_string();
    let handle = tokio::task::spawn_blocking(move || {
        let device = inference_device()?;
        load_sam3_handle(model_name, &checkpoint_path, device)
    })
    .await
    .map_err(|e| AppError::internal(e.to_string()))??;

    Ok(state.set_sam3_handle(handle).await)
}

fn inference_device() -> Result<candle_core::Device, AppError> {
    #[cfg(feature = "cuda")]
    {
        let ordinal = crate::inference::candle_sam3::configured_cuda_ordinal();
        candle_core::Device::new_cuda(ordinal).map_err(|error| {
            AppError::internal(format!(
                "CUDA backend was requested but device {ordinal} is unavailable: {error}"
            ))
        })
    }

    #[cfg(not(feature = "cuda"))]
    {
        Ok(candle_core::Device::Cpu)
    }
}

async fn run_image_predictor(
    handle: &Arc<Sam3ModelHandle>,
    prepared: &PreparedPipelineInput,
    input_window: Option<SharedInputWindow>,
    progress: &ProgressSender,
) -> Result<SegmentationOutput, AppError> {
    let segmenter = CandleSam3ImageSegmenter {
        handle: handle.clone(),
    };
    segment_image_frames(&segmenter, prepared, input_window, progress).await
}

async fn segment_image_frames(
    segmenter: &impl ImageSegmenter,
    prepared: &PreparedPipelineInput,
    input_window: Option<SharedInputWindow>,
    progress: &ProgressSender,
) -> Result<SegmentationOutput, AppError> {
    let mut masks = Vec::with_capacity(prepared.staged_frames.len());
    let mut prompts = prepared.prompts.iter().peekable();
    for (i, staged) in prepared.staged_frames.iter().enumerate() {
        let points = if prompts.peek().is_some_and(|prompt| prompt.frame_index == i) {
            prompts
                .next()
                .expect("matching prepared prompt")
                .points
                .as_slice()
        } else {
            &[]
        };
        let mut windowed_frame;
        let frame = if let Some(controller) = &input_window {
            windowed_frame = staged.frame.clone();
            controller
                .lock()
                .map_err(|_| AppError::internal("Input window controller lock poisoned"))?
                .blank_frame(i, &mut windowed_frame)?;
            &windowed_frame
        } else {
            &staged.frame
        };
        let mask = Arc::new(segmenter.segment(frame, points).await?);
        if let Some(controller) = &input_window {
            controller
                .lock()
                .map_err(|_| AppError::internal("Input window controller lock poisoned"))?
                .observe(i, mask.clone())?;
        }
        masks.push(mask);
        report_frame(
            progress,
            JobPhase::Inference,
            i,
            prepared.staged_frames.len(),
        );
    }
    let report = input_window
        .map(|controller| {
            controller
                .lock()
                .map_err(|_| AppError::internal("Input window controller lock poisoned"))?
                .finish()
        })
        .transpose()?;
    let masks = masks
        .into_iter()
        .map(|mask| {
            Arc::try_unwrap(mask)
                .map_err(|_| AppError::internal("Input window retained a completed raw mask"))
        })
        .collect::<Result<_, _>>()?;
    Ok(SegmentationOutput {
        masks,
        input_window: report,
    })
}

async fn run_video_predictor(
    handle: &Arc<Sam3ModelHandle>,
    prepared: &PreparedPipelineInput,
    input_window: Option<SharedInputWindow>,
    progress: &ProgressSender,
) -> Result<SegmentationOutput, AppError> {
    let segmenter = CandleSam3VideoSegmenter {
        handle: handle.clone(),
    };
    let sender = progress.clone();
    let callback: FrameProgressCallback = Arc::new(move |frame, total| {
        report_frame(&sender, JobPhase::Inference, frame, total);
    });
    let output = segmenter
        .segment_video(
            &prepared.plan.temp_volume_dir,
            prepared.prompts.clone(),
            input_window,
            Some(callback),
        )
        .await?;
    validate_video_masks(&output.masks, prepared)?;
    Ok(output)
}

fn validate_video_masks(
    masks: &[FrameMask],
    prepared: &PreparedPipelineInput,
) -> Result<(), AppError> {
    let expected_frames = prepared.staged_frames.len();
    if masks.len() != expected_frames {
        return Err(AppError::upstream(format!(
            "Video inference returned {} masks for {expected_frames} staged frames",
            masks.len()
        )));
    }

    let expected_width = prepared.volume.geometry.width;
    let expected_height = prepared.volume.geometry.height;
    for (index, mask) in masks.iter().enumerate() {
        if mask.width != expected_width || mask.height != expected_height {
            return Err(AppError::upstream(format!(
                "Video inference mask {index} has geometry {}x{}, expected {expected_width}x{expected_height}",
                mask.width, mask.height
            )));
        }
        if mask.pixels.len() != expected_width * expected_height {
            return Err(AppError::upstream(format!(
                "Video inference mask {index} has {} pixels, expected {}",
                mask.pixels.len(),
                expected_width * expected_height
            )));
        }
    }

    Ok(())
}

fn frame_name_width(frame_count: usize) -> usize {
    frame_count.saturating_sub(1).to_string().len().max(1)
}

fn stage_encoding(predictor_type: &str) -> Result<PreparedFrameEncoding, AppError> {
    match predictor_type {
        "ImagePredictor" => Ok(PreparedFrameEncoding::Tiff),
        "VideoPredictor" => Ok(PreparedFrameEncoding::Jpeg),
        other => Err(AppError::bad_request(format!(
            "Unknown predictor type: {other}"
        ))),
    }
}

fn annotation_overlay_path(output_path: &Path) -> PathBuf {
    output_artifact_path(output_path, "annotation_overlay", None)
}

fn frame_repair_report_path(output_path: &Path) -> PathBuf {
    output_artifact_path(output_path, "frame_repair", Some("json"))
}

fn unrepaired_mask_path(output_path: &Path) -> PathBuf {
    output_artifact_path(output_path, "unrepaired", None)
}

fn output_artifact_path(output_path: &Path, suffix: &str, extension: Option<&str>) -> PathBuf {
    let extension = extension
        .or_else(|| output_path.extension().and_then(|value| value.to_str()))
        .unwrap_or("tif");
    let stem = output_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("segmentation");
    output_path.with_file_name(format!("{stem}_{suffix}.{extension}"))
}

#[cfg(test)]
mod tests;
