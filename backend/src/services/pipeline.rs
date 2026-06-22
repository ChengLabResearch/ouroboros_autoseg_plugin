use std::{path::PathBuf, sync::Arc};

use crate::{
    app_state::{AppState, Sam3ModelHandle},
    domain::{paths::parse_mixed_path, requests::ProcessRequest},
    error::AppError,
    imaging::{
        annotations::{
            annotation_samples_for_video, default_annotations, interpolated_point_for_frame,
            AnnotationPoint, VolumeShape,
        },
        output::write_mask_stack,
        preprocess::{stage_input_frames, PreparedFrame, PreparedFrameEncoding},
        tiff_io::{inspect_volume, read_volume_frames, VolumeInput},
    },
    inference::{
        candle_sam3::{load_sam3_handle, CandleSam3ImageSegmenter, CandleSam3VideoSegmenter},
        image::{FrameMask, ImageSegmenter},
        video::VideoSegmenter,
    },
    services::volume::VolumeFileMapping,
};

#[derive(Debug, Clone)]
pub struct PipelinePlan {
    pub host_source: String,
    pub host_output: String,
    pub volume_source: PathBuf,
    pub temp_volume_dir: PathBuf,
    pub volume_output: PathBuf,
}

#[derive(Debug, Clone)]
pub struct PreparedPipelineInput {
    pub plan: PipelinePlan,
    pub volume: VolumeInput,
    pub annotation_points: Vec<AnnotationPoint>,
    pub staged_frames: Vec<PreparedFrame>,
}

impl PipelinePlan {
    pub fn copy_to_volume_mappings(&self) -> Vec<VolumeFileMapping> {
        vec![VolumeFileMapping::new(self.host_source.clone(), "")]
    }

    pub fn copy_to_host_mappings(&self) -> Vec<VolumeFileMapping> {
        vec![VolumeFileMapping::new(
            self.host_output.clone(),
            "Segmentation",
        )]
    }
}

pub fn plan(state: &AppState, request: &ProcessRequest) -> Result<PipelinePlan, AppError> {
    let input = parse_mixed_path(&request.file_path)?;
    let output = parse_mixed_path(&request.output_file)?;
    let plugin_root = state.config().plugin_root();

    Ok(PipelinePlan {
        host_source: input.raw,
        host_output: output.raw,
        volume_source: plugin_root.join(&input.file_name),
        temp_volume_dir: plugin_root.join(format!("{}_temp", input.stem)),
        volume_output: plugin_root.join("Segmentation").join(&output.file_name),
    })
}

pub async fn prepare(
    state: &AppState,
    request: &ProcessRequest,
) -> Result<PreparedPipelineInput, AppError> {
    request.validate()?;

    let plan = plan(state, request)?;
    let volume = inspect_volume(&plan.volume_source).await?;
    let frames = read_volume_frames(&plan.volume_source).await?;
    let annotation_points = volume.annotation_points.clone().unwrap_or_else(|| {
        default_annotations(
            VolumeShape {
                frames: volume.geometry.frames,
                height: volume.geometry.height,
                width: volume.geometry.width,
            },
            state.config().fallback_annotation_interval,
        )
    });
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
        annotation_points,
        staged_frames,
    })
}

pub async fn run(state: &AppState, job_id: &str, request: &ProcessRequest) -> Result<(), AppError> {
    request.validate()?;

    let handle = model_handle_for_request(state, &request.model_type).await?;

    state.update_job_step(job_id, 0, 5).await;

    let prepared = prepare(state, request).await?;
    if prepared.plan.temp_volume_dir.exists() {
        tokio::fs::remove_dir_all(&prepared.plan.temp_volume_dir).await?;
    }
    tokio::fs::create_dir_all(&prepared.plan.temp_volume_dir).await?;
    stage_input_frames(&prepared.staged_frames).await?;
    state.update_job_step(job_id, 1, 30).await;

    let masks = match request.predictor_type.as_str() {
        "ImagePredictor" => run_image_predictor(&handle, &prepared).await?,
        "VideoPredictor" => run_video_predictor(&handle, &prepared).await?,
        _ => unreachable!("validated by ProcessRequest::validate"),
    };
    state.update_job_step(job_id, 2, 70).await;

    if let Some(parent) = prepared.plan.volume_output.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    write_mask_stack(&prepared.plan.volume_output, &masks).await?;
    tokio::fs::remove_dir_all(&prepared.plan.temp_volume_dir).await?;
    state.update_job_step(job_id, 3, 100).await;

    Ok(())
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
        load_sam3_handle(model_name, &checkpoint_path, candle_core::Device::Cpu)
    })
    .await
    .map_err(|e| AppError::internal(e.to_string()))??;

    Ok(state.set_sam3_handle(handle).await)
}

async fn run_image_predictor(
    handle: &Arc<Sam3ModelHandle>,
    prepared: &PreparedPipelineInput,
) -> Result<Vec<FrameMask>, AppError> {
    let segmenter = CandleSam3ImageSegmenter {
        handle: handle.clone(),
    };
    let mut masks = Vec::with_capacity(prepared.staged_frames.len());
    for (i, staged) in prepared.staged_frames.iter().enumerate() {
        let prompts = interpolated_point_for_frame(&prepared.annotation_points, i)
            .map(|p| vec![p])
            .unwrap_or_default();
        let mask = segmenter.segment(&staged.frame, &prompts).await?;
        masks.push(mask);
    }
    Ok(masks)
}

async fn run_video_predictor(
    handle: &Arc<Sam3ModelHandle>,
    prepared: &PreparedPipelineInput,
) -> Result<Vec<FrameMask>, AppError> {
    let segmenter = CandleSam3VideoSegmenter {
        handle: handle.clone(),
    };
    let video_prompts =
        annotation_samples_for_video(&prepared.annotation_points, prepared.staged_frames.len());
    segmenter
        .segment_video(&prepared.plan.temp_volume_dir, &video_prompts)
        .await
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

#[cfg(test)]
mod tests;
