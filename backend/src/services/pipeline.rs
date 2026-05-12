use std::path::PathBuf;

use crate::{
    app_state::AppState,
    domain::{paths::parse_mixed_path, requests::ProcessRequest},
    error::AppError,
    imaging::{
        annotations::{default_annotations, AnnotationPoint, VolumeShape},
        preprocess::{stage_input_frames, PreparedFrame, PreparedFrameEncoding},
        tiff_io::{inspect_volume, read_volume_frames, VolumeInput},
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

    let plan = plan(state, request)?;
    if !plan.volume_source.exists() {
        return Err(AppError::not_implemented(
            "Rust TIFF staging and prompt plumbing are wired, but volume transfer and Candle inference are not wired yet",
        ));
    }

    let prepared = prepare(state, request).await?;
    if prepared.plan.temp_volume_dir.exists() {
        tokio::fs::remove_dir_all(&prepared.plan.temp_volume_dir).await?;
    }
    tokio::fs::create_dir_all(&prepared.plan.temp_volume_dir).await?;
    stage_input_frames(&prepared.staged_frames).await?;
    state.update_job_step(job_id, 1, 5).await;

    Err(AppError::not_implemented(
        "Rust TIFF staging and prompt plumbing are wired, but Candle inference is not wired yet",
    ))
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
