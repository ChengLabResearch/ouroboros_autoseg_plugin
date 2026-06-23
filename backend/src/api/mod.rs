pub mod health;
pub mod jobs;
pub mod models;

use axum::{
    routing::{get, post},
    Router,
};
use tower_http::{cors::CorsLayer, trace::TraceLayer};

use crate::app_state::AppState;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(health::server_active))
        .route("/startup-status", get(health::startup_status))
        .route("/model-status", get(models::model_status))
        .route("/download-model", post(models::download_model))
        .route("/process-stack", post(jobs::process_stack))
        .route("/status/:job_id", get(jobs::status))
        .route("/latest-job", get(jobs::latest_job))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
