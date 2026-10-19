use axum::{extract::State, Json};

use crate::{
    app_state::AppState,
    domain::{
        requests::DownloadRequest,
        responses::{DownloadModelResponse, ModelStatusResponse},
    },
    error::AppError,
};

pub async fn model_status(
    State(state): State<AppState>,
) -> Result<Json<ModelStatusResponse>, AppError> {
    let response = state.checkpoints().model_status().await?;
    Ok(Json(response))
}

pub async fn download_model(
    State(state): State<AppState>,
    Json(request): Json<DownloadRequest>,
) -> Result<Json<DownloadModelResponse>, AppError> {
    let response = state.checkpoints().download_model(&request).await?;
    Ok(Json(response))
}
