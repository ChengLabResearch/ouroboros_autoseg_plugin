use std::{
    convert::Infallible,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::{body::Body, http::Response, routing::get, Router};
use futures_util::{stream, StreamExt};
use ouroboros_autoseg_plugin_backend::{api, app_state::AppState, config::AppConfig};
use serde_json::Value;
use tokio::sync::Notify;

mod support;
use support::Server;

async fn wait_for_state(client: &reqwest::Client, url: &str, expected: &str) -> Value {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let snapshot: Value = client
                .get(format!("{url}/model-status"))
                .send()
                .await
                .expect("status response")
                .error_for_status()
                .expect("successful status")
                .json()
                .await
                .expect("status JSON");
            let download = &snapshot["downloads"]["medical_sam3"];
            if download["state"] == expected
                && (expected != "downloading" || download["bytes_received"] == 3)
            {
                return snapshot;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("expected checkpoint state before timeout")
}

#[tokio::test]
async fn routed_requests_share_download_owner_and_reconnect_after_caller_disconnect() {
    let release = Arc::new(Notify::new());
    let upstream_requests = Arc::new(AtomicUsize::new(0));
    let upstream = Server::start(Router::new().route(
        "/ChongCong/Medical-SAM3/resolve/716c9e1fb70edbdbbe65526b781e3a60f382d6cc/checkpoint_3D.pt",
        get({
            let release = release.clone();
            let requests = upstream_requests.clone();
            move || {
                let release = release.clone();
                let requests = requests.clone();
                async move {
                    requests.fetch_add(1, Ordering::SeqCst);
                    let chunks = stream::once(async { Ok::<_, Infallible>("abc") }).chain(
                        stream::once(async move {
                            release.notified().await;
                            Ok::<_, Infallible>("def")
                        }),
                    );
                    Response::builder()
                        .header("content-length", "6")
                        .body(Body::wrap_stream(chunks))
                        .expect("checkpoint response")
                }
            }
        }),
    ));
    let directory = tempfile::tempdir().expect("checkpoint directory");
    let config = AppConfig {
        plugin_name: "sam3-segmentation".into(),
        volume_name: "fixture-volume".into(),
        volume_server_url: "http://127.0.0.1:1".into(),
        huggingface_base_url: upstream.url.clone(),
        internal_volume_path: directory.path().to_path_buf(),
        checkpoint_dir: directory.path().join("sam3-segmentation/chkpts"),
        fallback_annotation_interval: 200,
        bind_address: "127.0.0.1:0".parse().expect("bind address"),
    };
    let target = config.checkpoint_path("medical_sam3");
    let partial = target.with_file_name("medical_sam3.pt.part");
    let backend = Server::start(api::router(AppState::new(config).expect("app state")));
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("client");
    let request = serde_json::json!({ "model_type": "medical_sam3", "hf_token": "" });
    let post = tokio::spawn({
        let client = client.clone();
        let url = format!("{}/download-model", backend.url);
        let request = request.clone();
        async move { client.post(url).json(&request).send().await }
    });

    let downloading = wait_for_state(&client, &backend.url, "downloading").await;
    assert_eq!(downloading["models"]["medical_sam3"], false);
    assert_eq!(downloading["downloads"]["medical_sam3"]["total_bytes"], 6);
    assert_eq!(
        downloading["downloads"]["medical_sam3"]["error"],
        Value::Null
    );
    assert!(!target.exists());
    assert!(
        partial.exists(),
        "active partial file must exist before the duplicate"
    );

    let duplicate = client
        .post(format!("{}/download-model", backend.url))
        .json(&request)
        .send()
        .await
        .expect("duplicate response");
    assert_eq!(duplicate.status(), reqwest::StatusCode::CONFLICT);
    assert_eq!(upstream_requests.load(Ordering::SeqCst), 1);
    assert!(
        partial.exists(),
        "duplicate must not remove the active partial file"
    );
    assert!(
        !post.is_finished(),
        "original POST still awaits installation"
    );

    post.abort();
    let _ = post.await;
    release.notify_one();
    let available = wait_for_state(&client, &backend.url, "available").await;
    assert_eq!(available["models"]["medical_sam3"], true);
    assert_eq!(available["downloads"]["medical_sam3"]["bytes_received"], 6);
    assert_eq!(available["downloads"]["medical_sam3"]["total_bytes"], 6);
    assert_eq!(
        std::fs::read(&target).expect("installed checkpoint"),
        b"abcdef"
    );
    assert!(!partial.exists());

    let existing: Value = client
        .post(format!("{}/download-model", backend.url))
        .json(&request)
        .send()
        .await
        .expect("existing response")
        .error_for_status()
        .expect("existing success")
        .json()
        .await
        .expect("existing JSON");
    assert_eq!(existing["status"], "exists");
    assert_eq!(upstream_requests.load(Ordering::SeqCst), 1);
}
