use axum::{
    extract::{Path, State},
    Json,
};
use uuid::Uuid;

use crate::{
    app_state::AppState,
    domain::{
        requests::ProcessRequest,
        responses::{JobSubmissionResponse, LatestJobResponse},
    },
    error::AppError,
    services::runner,
};

pub async fn process_stack(
    State(state): State<AppState>,
    Json(request): Json<ProcessRequest>,
) -> Result<Json<JobSubmissionResponse>, AppError> {
    request.validate()?;

    let job_id = Uuid::new_v4().to_string();
    state.create_job(job_id.clone()).await;

    let task_state = state.clone();
    let task_job_id = job_id.clone();
    tokio::spawn(async move {
        runner::execute_job(task_state, task_job_id, request).await;
    });

    Ok(Json(JobSubmissionResponse {
        job_id,
        status: "started".to_string(),
    }))
}

pub async fn status(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<crate::domain::jobs::JobRecord>, AppError> {
    let job = state
        .get_job(&job_id)
        .await
        .ok_or_else(|| AppError::not_found("Job not found"))?;
    Ok(Json(job))
}

pub async fn latest_job(
    State(state): State<AppState>,
) -> Result<Json<LatestJobResponse>, AppError> {
    let response = if let Some((job_id, record)) = state.latest_running_job().await {
        LatestJobResponse {
            job_id: Some(job_id),
            status: record.status,
        }
    } else {
        LatestJobResponse {
            job_id: None,
            status: "none".to_string(),
        }
    };

    Ok(Json(response))
}
