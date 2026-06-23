use tracing::error;

use crate::{app_state::AppState, domain::requests::ProcessRequest, services::pipeline};

pub async fn execute_job(state: AppState, job_id: String, request: ProcessRequest) {
    let result = state
        .with_inference_slot(|| async {
            state.update_job_step(&job_id, 0, 5).await;
            state.update_job_step(&job_id, 1, 1).await;
            pipeline::run(&state, &job_id, &request).await
        })
        .await;

    match result {
        Ok(()) => {
            state.update_job_step(&job_id, 2, 100).await;
            state.set_job_status(&job_id, "completed").await;
        }
        Err(error_value) => {
            error!(job_id = %job_id, error = %error_value, "job execution failed");
            state.set_job_status(&job_id, "error").await;
        }
    }
}
