use axum::{extract::State, Json};

use crate::{
    app_state::AppState,
    domain::{
        requests::DownloadRequest,
        responses::{DownloadModelResponse, ModelStatusResponse},
    },
    error::AppError,
    services::checkpoints,
};

pub async fn model_status(
    State(state): State<AppState>,
) -> Result<Json<ModelStatusResponse>, AppError> {
    let response = checkpoints::model_status(state.config()).await?;
    Ok(Json(response))
}

pub async fn download_model(
    State(state): State<AppState>,
    Json(request): Json<DownloadRequest>,
) -> Result<Json<DownloadModelResponse>, AppError> {
    let response =
        checkpoints::download_model(state.config(), state.http_client(), &request).await?;
    Ok(Json(response))
}
