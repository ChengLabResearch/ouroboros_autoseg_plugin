use std::path::PathBuf;

use crate::{
    app_state::AppState,
    domain::{paths::parse_mixed_path, requests::ProcessRequest},
    error::AppError,
};

#[derive(Debug, Clone)]
pub struct PipelinePlan {
    pub host_source: String,
    pub host_output: String,
    pub volume_source: PathBuf,
    pub temp_volume_dir: PathBuf,
    pub volume_output: PathBuf,
}

pub fn plan(state: &AppState, request: &ProcessRequest) -> Result<PipelinePlan, AppError> {
    let input = parse_mixed_path(&request.file_path)?;
    let output = parse_mixed_path(&request.output_file)?;
    let plugin_root = state
        .config()
        .internal_volume_path
        .join(&state.config().plugin_name);

    Ok(PipelinePlan {
        host_source: input.raw,
        host_output: output.raw,
        volume_source: plugin_root.join(&input.file_name),
        temp_volume_dir: plugin_root.join(format!("{}_temp", input.stem)),
        volume_output: plugin_root.join("Segmentation").join(&output.file_name),
    })
}

pub async fn run(
    state: &AppState,
    _job_id: &str,
    request: &ProcessRequest,
) -> Result<(), AppError> {
    let _plan = plan(state, request)?;

    Err(AppError::not_implemented(
        "Rust pipeline scaffold is in place, but TIFF staging and Candle inference are not wired yet",
    ))
}
