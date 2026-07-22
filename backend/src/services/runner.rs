use tracing::error;

use crate::{
    app_state::AppState,
    domain::requests::ProcessRequest,
    services::{pipeline, progress},
};

pub async fn execute_job(state: AppState, job_id: String, request: ProcessRequest) {
    let (progress_sender, progress_receiver) = progress::channel();
    let progress_task = tokio::spawn(progress::consume(
        state.clone(),
        job_id.clone(),
        progress_receiver,
    ));
    let result = state
        .with_inference_slot(|| pipeline::run(&state, &request, &progress_sender))
        .await;
    drop(progress_sender);
    if let Err(error_value) = progress_task.await {
        error!(job_id = %job_id, error = %error_value, "progress consumer failed");
    }

    match result {
        Ok(()) => {
            state.complete_job(&job_id).await;
        }
        Err(error_value) => {
            error!(job_id = %job_id, error = %error_value, "job execution failed");
            let phase = state
                .get_job(&job_id)
                .await
                .and_then(|record| record.active_phase)
                .map(|phase| phase.name())
                .unwrap_or("Unknown");
            state
                .fail_job(&job_id, format!("{phase}: {error_value}"))
                .await;
        }
    }
}
