use std::path::PathBuf;

use reqwest::StatusCode;
use uuid::Uuid;

use super::{
    download_failure_message, download_model, model_status, partial_download_path,
    prepare_download_target,
};
use crate::{config::AppConfig, domain::requests::DownloadRequest};

fn unique_temp_dir() -> PathBuf {
    let path = std::env::temp_dir().join(format!("ouroboros-rust-phase2-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&path).expect("create temp dir");
    path
}

fn test_config(base_url: &str, root: PathBuf) -> AppConfig {
    let plugin_name = "sam3-segmentation".to_string();
    let checkpoint_dir = root.join(&plugin_name).join("chkpts");
    AppConfig {
        plugin_name,
        volume_name: "ouroboros-volume".to_string(),
        volume_server_url: "http://127.0.0.1:3001".to_string(),
        huggingface_base_url: base_url.to_string(),
        internal_volume_path: root,
        checkpoint_dir,
        fallback_annotation_interval: 200,
        bind_address: "127.0.0.1:8686".parse().expect("valid socket"),
    }
}

#[tokio::test]
async fn model_status_reports_existing_checkpoints() {
    let root = unique_temp_dir();
    let config = test_config("https://huggingface.co", root.clone());
    std::fs::create_dir_all(&config.checkpoint_dir).expect("checkpoint dir");
    std::fs::write(config.checkpoint_dir.join("sam3.pt"), b"weights").expect("write checkpoint");

    let status = model_status(&config).await.expect("status succeeds");
    assert_eq!(status.models.get("sam2_hiera_base_plus"), Some(&false));
    assert_eq!(status.models.get("sam3"), Some(&true));
}

#[tokio::test]
async fn prepare_download_target_removes_stale_partial_file() {
    let target_dir = unique_temp_dir();
    let target_path = target_dir.join("weights.pt");
    let partial_path = partial_download_path(&target_path);
    tokio::fs::write(&partial_path, b"stale")
        .await
        .expect("write part");

    let prepared = prepare_download_target(&target_path)
        .await
        .expect("prepare target");

    assert_eq!(prepared, partial_path);
    assert!(!prepared.exists());
}

#[tokio::test]
async fn prepare_download_target_removes_stale_partial_directory() {
    let target_dir = unique_temp_dir();
    let target_path = target_dir.join("weights.pt");
    let partial_path = partial_download_path(&target_path);
    tokio::fs::create_dir_all(&partial_path)
        .await
        .expect("create part dir");

    let prepared = prepare_download_target(&target_path)
        .await
        .expect("prepare target");

    assert_eq!(prepared, partial_path);
    assert!(!prepared.exists());
}

#[tokio::test]
async fn download_model_rejects_missing_hf_token() {
    let root = unique_temp_dir();
    let config = test_config("https://huggingface.co", root);
    let client = reqwest::Client::new();

    let error = download_model(
        &config,
        &client,
        &DownloadRequest {
            model_type: "sam3".to_string(),
            hf_token: None,
        },
    )
    .await
    .expect_err("missing token should fail");

    assert_eq!(error.to_string(), "Authentication Token required for SAM 3");
}

#[tokio::test]
async fn download_model_reports_existing_files() {
    let root = unique_temp_dir();
    let config = test_config("https://huggingface.co", root);
    std::fs::create_dir_all(&config.checkpoint_dir).expect("checkpoint dir");
    std::fs::write(config.checkpoint_dir.join("sam3.pt"), b"weights").expect("write checkpoint");
    let client = reqwest::Client::new();

    let response = download_model(
        &config,
        &client,
        &DownloadRequest {
            model_type: "sam3".to_string(),
            hf_token: Some("hf_token".to_string()),
        },
    )
    .await
    .expect("existing response");

    assert_eq!(response.status, "exists");
}

#[test]
fn partial_download_path_appends_part_suffix() {
    let target_path = PathBuf::from("/tmp/checkpoints/sam3.pt");
    assert_eq!(
        partial_download_path(&target_path),
        PathBuf::from("/tmp/checkpoints/sam3.pt.part")
    );
}

#[test]
fn download_failure_message_matches_empty_and_non_empty_responses() {
    assert_eq!(
        download_failure_message(StatusCode::UNAUTHORIZED, "bad token"),
        "Failed to download checkpoint: 401 Unauthorized (bad token)"
    );
    assert_eq!(
        download_failure_message(StatusCode::BAD_GATEWAY, "  "),
        "Failed to download checkpoint: 502 Bad Gateway"
    );
}
