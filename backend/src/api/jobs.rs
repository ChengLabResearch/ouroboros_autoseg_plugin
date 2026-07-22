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

#[cfg(test)]
mod tests {
    use super::{latest_job, status};
    use crate::{app_state::AppState, config::AppConfig, domain::jobs::JobPhase};
    use axum::extract::{Path, State};
    use std::path::PathBuf;

    fn state() -> AppState {
        AppState::new(AppConfig {
            plugin_name: "test".into(),
            volume_name: "test".into(),
            volume_server_url: "http://127.0.0.1:3001".into(),
            huggingface_base_url: "https://huggingface.co".into(),
            checkpoint_dir: PathBuf::from("/tmp/test"),
            internal_volume_path: PathBuf::from("/tmp/test"),
            fallback_annotation_interval: 1,
            bind_address: "127.0.0.1:8686".parse().unwrap(),
        })
        .unwrap()
    }

    #[tokio::test]
    async fn fresh_client_can_discover_latest_running_job_then_fetch_progress() {
        let state = state();
        state.create_job("older".into()).await;
        state.complete_job("older").await;
        state.create_job("running".into()).await;
        state
            .update_job_phase("running", JobPhase::Inference, 42)
            .await;

        let latest = latest_job(State(state.clone())).await.unwrap().0;
        assert_eq!(latest.job_id.as_deref(), Some("running"));
        assert_eq!(latest.status, "running");

        let record = status(State(state), Path("running".into()))
            .await
            .unwrap()
            .0;
        assert_eq!(record.active_phase, Some(JobPhase::Inference));
        assert_eq!(record.steps[0].progress, 100);
        assert_eq!(record.steps[1].progress, 42);
    }

    #[tokio::test]
    async fn latest_job_does_not_reconnect_terminal_jobs() {
        let state = state();
        state.create_job("done".into()).await;
        state.complete_job("done").await;
        let latest = latest_job(State(state)).await.unwrap().0;
        assert_eq!(latest.job_id, None);
        assert_eq!(latest.status, "none");
    }
}
