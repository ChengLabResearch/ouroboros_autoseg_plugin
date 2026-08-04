use std::error::Error;

use ouroboros_autoseg_plugin_backend::{api, app_state::AppState, config::AppConfig};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt()
        .with_ansi(false)
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            EnvFilter::new("ouroboros_autoseg_plugin_backend=info,tower_http=info")
        }))
        .with_target(false)
        .compact()
        .init();

    let config = AppConfig::from_env()?;
    let state = AppState::new(config)?;
    let bind_address = state.config().bind_address;
    let app = api::router(state);
    let plugin_sha = std::env::var("PLUGIN_GIT_SHA").unwrap_or_else(|_| "unknown".to_string());
    let candle_sha = option_env!("CANDLE_SAM3_GIT_SHA").unwrap_or("unknown");

    info!(plugin_sha = %plugin_sha, candle_sha = %candle_sha, "backend revision metadata");
    info!(address = %bind_address, "Rust backend scaffold listening");

    axum::Server::bind(&bind_address)
        .serve(app.into_make_service())
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};

        if let Ok(mut stream) = signal(SignalKind::terminate()) {
            let _ = stream.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}
