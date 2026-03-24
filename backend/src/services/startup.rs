use crate::{
    app_state::AppState, domain::responses::StartupStatus, error::AppError, services::volume,
};

pub async fn refresh(state: &AppState) -> Result<StartupStatus, AppError> {
    let mut status = state.startup_status().await;

    status.set_step_status("Building Docker Image", "completed");

    match volume::ping(state.http_client(), state.config()).await {
        Ok(()) => status.set_step_status("Connecting to Volume Server", "completed"),
        Err(_) => status.set_step_status("Connecting to Volume Server", "warning"),
    }

    let ml_status = if state.ml_runtime_ready().await {
        "completed"
    } else {
        "warning"
    };
    status.set_step_status("Initializing ML Models", ml_status);
    status.recalculate_readiness();

    state.set_startup_status(status.clone()).await;
    Ok(status)
}
