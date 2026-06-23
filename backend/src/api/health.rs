use axum::{extract::State, Json};

use crate::{
    app_state::AppState, domain::responses::StartupStatus, error::AppError, services::startup,
};

pub async fn server_active() -> Json<&'static str> {
    Json("Segmentation server is active")
}

pub async fn startup_status(
    State(state): State<AppState>,
) -> Result<Json<StartupStatus>, AppError> {
    let status = startup::refresh(&state).await?;
    Ok(Json(status))
}
