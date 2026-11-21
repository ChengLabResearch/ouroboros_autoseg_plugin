use std::{
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::Router;
use ouroboros_autoseg_plugin_backend::{api, app_state::AppState, config::AppConfig};
use serde_json::Value;

mod support;
use support::Server;

const CHECKPOINT_BYTES: &[u8] = b"checkpoint fixture";

fn fixture_config(root: &Path, upstream: &Server) -> AppConfig {
    let mut config = AppConfig::from_env().expect("environment config");
    config.internal_volume_path = root.to_path_buf();
    config.checkpoint_dir = root
        .join("persist")
        .join(&config.plugin_name)
        .join("chkpts");
    config.huggingface_base_url = upstream.url.clone();
    config
}

fn upstream() -> (Server, Arc<AtomicUsize>) {
    let requests = Arc::new(AtomicUsize::new(0));
    let counter = requests.clone();
    let server = Server::start(Router::new().fallback(move || {
        let counter = counter.clone();
        async move {
            counter.fetch_add(1, Ordering::SeqCst);
            CHECKPOINT_BYTES
        }
    }));
    (server, requests)
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("client")
}

async fn model_status(client: &reqwest::Client, backend: &Server) -> Value {
    client
        .get(format!("{}/model-status", backend.url))
        .send()
        .await
        .expect("status response")
        .error_for_status()
        .expect("successful status")
        .json()
        .await
        .expect("status JSON")
}

async fn download(client: &reqwest::Client, backend: &Server) -> Value {
    client
        .post(format!("{}/download-model", backend.url))
        .json(&serde_json::json!({ "model_type": "medical_sam3", "hf_token": "" }))
        .send()
        .await
        .expect("download response")
        .error_for_status()
        .expect("successful download")
        .json()
        .await
        .expect("download JSON")
}

fn simulate_host_cleanup(root: &Path, preserve_persist: bool) {
    // Filesystem equivalent of the top-level find predicates in Ouroboros
    // resources/processes/volume-cleanup.mjs at
    // 94087e698d79e5c21ac3b2df884f503a1d377787 (#257): dot entries,
    // main-statistics, and persist survive. The released older host omits
    // only the persist exclusion. This exercises actual deletion and backend
    // restart without claiming to launch the desktop or run model inference.
    for entry in std::fs::read_dir(root).expect("volume entries") {
        let entry = entry.expect("volume entry");
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.')
            || name == "main-statistics"
            || (preserve_persist && name == "persist")
        {
            continue;
        }
        if entry.file_type().expect("entry type").is_dir() {
            std::fs::remove_dir_all(entry.path()).expect("clear transient directory");
        } else {
            std::fs::remove_file(entry.path()).expect("clear transient file");
        }
    }
}

#[tokio::test]
async fn startup_migration_survives_host_cleanup_and_restart_without_downloading() {
    let root = tempfile::tempdir().expect("volume");
    let (upstream, requests) = upstream();
    let config = fixture_config(root.path(), &upstream);
    let legacy = config.plugin_root().join("chkpts");
    std::fs::create_dir_all(&legacy).expect("legacy checkpoint directory");
    let source = legacy.join("medical_sam3.pt");
    std::fs::write(&source, CHECKPOINT_BYTES).expect("legacy checkpoint");
    let partial = legacy.join("sam3.pt.part");
    std::fs::write(&partial, b"unfinished").expect("legacy partial");
    let target = config.checkpoint_path("medical_sam3");
    assert!(!config.checkpoint_dir.exists());

    let backend = Server::start(api::router(AppState::new(config.clone()).expect("startup")));
    let client = client();
    let status = model_status(&client, &backend).await;
    assert_eq!(
        status["checkpoint_dir"],
        target.parent().unwrap().display().to_string()
    );
    assert_eq!(status["models"]["medical_sam3"], true);
    assert_eq!(status["models"]["sam3"], false);
    assert_eq!(
        status["downloads"]["medical_sam3"]["bytes_received"],
        CHECKPOINT_BYTES.len()
    );
    assert_eq!(
        std::fs::read(&target).expect("migrated checkpoint"),
        CHECKPOINT_BYTES
    );
    assert!(
        !source.exists(),
        "startup must move, not copy, the legacy file"
    );
    assert!(partial.exists(), "migration must leave partials untouched");
    assert!(!config.checkpoint_path("sam3").exists());
    drop(backend);

    simulate_host_cleanup(root.path(), true);
    assert!(
        !config.plugin_root().exists(),
        "host removed transient plugin storage"
    );
    assert_eq!(
        std::fs::read(&target).expect("preserved checkpoint"),
        CHECKPOINT_BYTES
    );
    let restarted = Server::start(api::router(AppState::new(config.clone()).expect("restart")));
    let status = model_status(&client, &restarted).await;
    assert_eq!(
        status["checkpoint_dir"],
        config.checkpoint_dir.display().to_string()
    );
    assert_eq!(status["models"]["medical_sam3"], true);
    assert_eq!(download(&client, &restarted).await["status"], "exists");
    assert_eq!(requests.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn older_host_cleanup_still_allows_downloading_after_restart() {
    let root = tempfile::tempdir().expect("volume");
    let (upstream, requests) = upstream();
    let config = fixture_config(root.path(), &upstream);
    let target = config.checkpoint_path("medical_sam3");
    let backend = Server::start(api::router(AppState::new(config.clone()).expect("startup")));
    let client = client();
    assert!(
        !config.checkpoint_dir.exists(),
        "startup does not eagerly create storage"
    );
    assert_eq!(
        model_status(&client, &backend).await["models"]["medical_sam3"],
        false
    );
    assert_eq!(download(&client, &backend).await["status"], "success");
    assert_eq!(
        std::fs::read(&target).expect("downloaded checkpoint"),
        CHECKPOINT_BYTES
    );
    assert_eq!(requests.load(Ordering::SeqCst), 1);
    drop(backend);

    simulate_host_cleanup(root.path(), false);
    assert!(!target.exists());
    let restarted = Server::start(api::router(AppState::new(config.clone()).expect("restart")));
    assert_eq!(
        model_status(&client, &restarted).await["models"]["medical_sam3"],
        false
    );
    assert_eq!(download(&client, &restarted).await["status"], "success");
    assert_eq!(requests.load(Ordering::SeqCst), 2);
    assert_eq!(
        std::fs::read(&target).expect("replacement checkpoint"),
        CHECKPOINT_BYTES
    );
    let status = model_status(&client, &restarted).await;
    assert_eq!(
        status["checkpoint_dir"],
        target.parent().unwrap().display().to_string()
    );
    assert_eq!(status["models"]["medical_sam3"], true);
}
