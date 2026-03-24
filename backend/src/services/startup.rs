use crate::{
    app_state::AppState, domain::responses::StartupStatus, error::AppError, services::volume,
};

pub async fn refresh(state: &AppState) -> Result<StartupStatus, AppError> {
    let mut status = state.startup_status().await;
    let volume_connected = volume::ping(state.http_client(), state.config())
        .await
        .is_ok();
    let ml_ready = state.ml_runtime_ready().await;

    apply_probe_results(&mut status, volume_connected, ml_ready);

    state.set_startup_status(status.clone()).await;
    Ok(status)
}

fn apply_probe_results(status: &mut StartupStatus, volume_connected: bool, ml_ready: bool) {
    status.set_step_status("Building Docker Image", "completed");
    status.set_step_status(
        "Connecting to Volume Server",
        if volume_connected {
            "completed"
        } else {
            "warning"
        },
    );
    status.set_step_status(
        "Initializing ML Models",
        if ml_ready { "completed" } else { "warning" },
    );
    status.recalculate_readiness();
}

#[cfg(test)]
mod tests;
