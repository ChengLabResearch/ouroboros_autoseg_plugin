use reqwest::StatusCode;
use serde_json::json;

use super::{
    build_operation_request, is_ping_status_healthy, volume_failure_message, VolumeFileMapping,
};
use crate::config::AppConfig;

fn test_config(base_url: &str) -> AppConfig {
    let internal_volume_path = std::env::temp_dir().join("ouroboros-rust-phase2-volume");
    AppConfig {
        plugin_name: "sam3-segmentation".to_string(),
        volume_name: "ouroboros-volume".to_string(),
        volume_server_url: base_url.to_string(),
        huggingface_base_url: "https://huggingface.co".to_string(),
        checkpoint_dir: internal_volume_path
            .join("sam3-segmentation")
            .join("chkpts"),
        internal_volume_path,
        fallback_annotation_interval: 200,
        bind_address: "127.0.0.1:8686".parse().expect("valid socket"),
    }
}

#[test]
fn ping_accepts_success_not_found_and_method_not_allowed() {
    for status in [
        StatusCode::OK,
        StatusCode::NOT_FOUND,
        StatusCode::METHOD_NOT_ALLOWED,
    ] {
        assert!(
            is_ping_status_healthy(status),
            "status {status} should be accepted"
        );
    }
}

#[test]
fn ping_rejects_other_statuses() {
    assert!(!is_ping_status_healthy(StatusCode::INTERNAL_SERVER_ERROR));
}

#[test]
fn copy_to_volume_sends_expected_payload() {
    let config = test_config("http://127.0.0.1:3001");
    let files = [VolumeFileMapping::new("/host/input.tif", "")];
    let request = build_operation_request(&config, &files);

    assert_eq!(
        serde_json::to_value(&request).expect("request serializes"),
        json!({
            "volumeName": "ouroboros-volume",
            "pluginFolderName": "sam3-segmentation",
            "files": [{
                "sourcePath": "/host/input.tif",
                "targetPath": ""
            }]
        })
    );
}

#[test]
fn copy_to_host_returns_upstream_message_on_error() {
    assert_eq!(
        volume_failure_message(StatusCode::BAD_REQUEST, "copy back failed"),
        "copy back failed"
    );
    assert_eq!(
        volume_failure_message(StatusCode::BAD_GATEWAY, "   "),
        "Volume operation failed with status 502 Bad Gateway"
    );
}
