use std::{path::PathBuf, sync::Arc};

use reqwest::StatusCode;
use uuid::Uuid;

use super::{
    download_failure_message, partial_download_path, prepare_download_target, CheckpointService,
};
use crate::{
    config::{AppConfig, DownloadSource},
    domain::{requests::DownloadRequest, responses::ModelDownloadState},
};

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
    std::fs::write(config.checkpoint_dir.join("medical_sam3.pt"), b"weights")
        .expect("write checkpoint");

    let status = service(&config)
        .model_status()
        .await
        .expect("status succeeds");
    assert_eq!(status.models.get("sam3"), Some(&false));
    assert_eq!(status.models.get("medical_sam3"), Some(&true));
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

    let service = CheckpointService::new(Arc::new(config), client);
    let error = service
        .download_model(&DownloadRequest {
            model_type: "sam3".to_string(),
            hf_token: None,
        })
        .await
        .expect_err("missing token should fail");

    assert_eq!(
        error.to_string(),
        "Authentication Token required for SAM3 (Official)"
    );
}

#[tokio::test]
async fn download_model_reports_existing_files() {
    let root = unique_temp_dir();
    let config = test_config("https://huggingface.co", root);
    std::fs::create_dir_all(&config.checkpoint_dir).expect("checkpoint dir");
    std::fs::write(config.checkpoint_dir.join("medical_sam3.pt"), b"weights")
        .expect("write checkpoint");
    let client = reqwest::Client::new();

    let service = CheckpointService::new(Arc::new(config), client);
    let response = service
        .download_model(&DownloadRequest {
            model_type: "medical_sam3".to_string(),
            hf_token: None,
        })
        .await
        .expect("existing response");

    assert_eq!(response.status, "exists");
}

#[tokio::test]
async fn download_model_uses_catalog_revision_and_preserves_checkpoint_contract() {
    use axum::{http::HeaderMap, routing::get, Router};

    // Only the expected revision exists: a request to main for Medical SAM3
    // returns 404, reproducing the upstream deletion without network access.
    for (model, path, token, stored_name) in [
        (
            "medical_sam3",
            "/ChongCong/Medical-SAM3/resolve/716c9e1fb70edbdbbe65526b781e3a60f382d6cc/checkpoint_3D.pt",
            None,
            "medical_sam3.pt",
        ),
        (
            "sam3",
            "/facebook/sam3/resolve/main/sam3.pt",
            Some("fixture-token"),
            "sam3.pt",
        ),
    ] {
        let app = Router::new().route(
            path,
            get(move |headers: HeaderMap| async move {
                let expected = token.map(|value| format!("Bearer {value}"));
                let actual = headers.get("authorization").and_then(|value| value.to_str().ok());
                if actual != expected.as_deref() {
                    return (StatusCode::UNAUTHORIZED, "unexpected authorization");
                }
                (StatusCode::OK, "checkpoint fixture")
            }),
        );
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind fixture server");
        listener.set_nonblocking(true).expect("nonblocking listener");
        let address = listener.local_addr().expect("fixture address");
        let server = tokio::spawn(
            axum::Server::from_tcp(listener)
                .expect("fixture server")
                .serve(app.into_make_service()),
        );
        let root = tempfile::tempdir().expect("checkpoint directory");
        // Also exercise the existing trailing-slash base URL override contract.
        let config = test_config(&format!("http://{address}/"), root.path().to_path_buf());
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .expect("client");
        let service = CheckpointService::new(Arc::new(config.clone()), client);
        let result = service.download_model(
            &DownloadRequest {
                model_type: model.to_string(),
                hf_token: token.map(str::to_string),
            },
        )
        .await;
        server.abort();

        assert_eq!(result.expect("download from catalog source").status, "success");
        let target = config.checkpoint_dir.join(stored_name);
        assert_eq!(std::fs::read(&target).expect("published checkpoint"), b"checkpoint fixture");
        assert!(!partial_download_path(&target).exists());
        assert_eq!(service.model_status().await.expect("model status").models.get(model), Some(&true));
    }
}

#[test]
fn sam3_descriptor_uses_official_checkpoint() {
    let root = unique_temp_dir();
    let config = test_config("https://huggingface.co", root);
    let descriptor = config
        .model_descriptor("sam3")
        .expect("sam3 descriptor should exist");

    assert_eq!(descriptor.checkpoint_file, "sam3.pt");
    match descriptor.download_source {
        DownloadSource::HuggingFace {
            repo,
            revision,
            filename,
            requires_token,
        } => {
            assert_eq!(repo, "facebook/sam3");
            assert_eq!(revision, "main");
            assert_eq!(filename, "sam3.pt");
            assert!(requires_token);
        }
        DownloadSource::PublicUrl(_) => panic!("sam3 should use Hugging Face"),
    }
}

#[test]
fn medical_sam3_descriptor_uses_public_medical_checkpoint() {
    let root = unique_temp_dir();
    let config = test_config("https://huggingface.co", root);
    let descriptor = config
        .model_descriptor("medical_sam3")
        .expect("medical_sam3 descriptor should exist");

    assert_eq!(descriptor.checkpoint_file, "medical_sam3.pt");
    match descriptor.download_source {
        DownloadSource::HuggingFace {
            repo,
            revision,
            filename,
            requires_token,
        } => {
            assert_eq!(repo, "ChongCong/Medical-SAM3");
            assert_eq!(revision, "716c9e1fb70edbdbbe65526b781e3a60f382d6cc");
            assert_eq!(filename, "checkpoint_3D.pt");
            assert!(!requires_token);
        }
        DownloadSource::PublicUrl(_) => panic!("medical_sam3 should use Hugging Face"),
    }
}

#[test]
fn partial_download_path_appends_part_suffix() {
    let target_path = PathBuf::from("/tmp/checkpoints/medical_sam3.pt");
    assert_eq!(
        partial_download_path(&target_path),
        PathBuf::from("/tmp/checkpoints/medical_sam3.pt.part")
    );
}

#[test]
fn download_failure_message_is_actionable_without_upstream_body() {
    assert_eq!(
        download_failure_message(StatusCode::UNAUTHORIZED),
        "Failed to download checkpoint: 401 Unauthorized. Check your Hugging Face token and access to the model."
    );
    assert_eq!(
        download_failure_message(StatusCode::BAD_GATEWAY),
        "Failed to download checkpoint: 502 Bad Gateway. Try again later."
    );
}

fn service(config: &AppConfig) -> CheckpointService {
    CheckpointService::new(
        Arc::new(config.clone()),
        reqwest::Client::builder().no_proxy().build().unwrap(),
    )
}

struct HttpFixture {
    url: String,
    streams: Vec<Option<tokio::sync::mpsc::Sender<Result<Vec<u8>, std::io::Error>>>>,
    requests: Arc<std::sync::atomic::AtomicUsize>,
    task: tokio::task::JoinHandle<()>,
}

impl HttpFixture {
    async fn new(lengths: &[Option<u64>]) -> Self {
        use axum::{body::Body, response::Response, routing::get, Router};
        use std::{
            collections::VecDeque,
            sync::atomic::{AtomicUsize, Ordering},
        };

        let mut streams = Vec::new();
        let mut responses = VecDeque::new();
        for length in lengths {
            let (sender, receiver) =
                tokio::sync::mpsc::channel::<Result<Vec<u8>, std::io::Error>>(8);
            streams.push(Some(sender));
            responses.push_back((*length, receiver));
        }
        let responses = Arc::new(tokio::sync::Mutex::new(responses));
        let requests = Arc::new(AtomicUsize::new(0));
        let count = requests.clone();
        let app = Router::new().fallback(get(move || {
            let responses = responses.clone();
            let count = count.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                let (length, receiver) = responses
                    .lock()
                    .await
                    .pop_front()
                    .expect("fixture response");
                let stream = futures_util::stream::unfold(receiver, |mut receiver| async move {
                    receiver.recv().await.map(|chunk| (chunk, receiver))
                });
                let mut builder = Response::builder();
                if let Some(length) = length {
                    builder = builder.header("content-length", length);
                }
                builder.body(Body::wrap_stream(stream)).unwrap()
            }
        }));
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("fixture listener");
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::Server::from_tcp(listener)
                .unwrap()
                .serve(app.into_make_service())
                .await
                .unwrap();
        });
        Self {
            url: format!("http://{address}"),
            streams,
            requests,
            task,
        }
    }

    fn stream(
        &mut self,
        index: usize,
    ) -> tokio::sync::mpsc::Sender<Result<Vec<u8>, std::io::Error>> {
        self.streams[index].take().unwrap()
    }
}

impl Drop for HttpFixture {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn download(
    service: &CheckpointService,
) -> tokio::task::JoinHandle<
    Result<crate::domain::responses::DownloadModelResponse, crate::error::AppError>,
> {
    let service = service.clone();
    tokio::spawn(async move {
        service
            .download_model(&DownloadRequest {
                model_type: "medical_sam3".into(),
                hf_token: None,
            })
            .await
    })
}

async fn wait_status(
    service: &CheckpointService,
    predicate: impl Fn(&crate::domain::responses::ModelDownloadStatus) -> bool,
) -> crate::domain::responses::ModelDownloadStatus {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let response = service.model_status().await.unwrap();
            let status = &response.downloads["medical_sam3"];
            assert_eq!(
                response.models["medical_sam3"],
                status.state == ModelDownloadState::Available
            );
            if predicate(status) {
                return status.clone();
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("status transition")
}

#[tokio::test]
async fn streamed_bytes_are_live_and_duplicate_cannot_replace_partial_file() {
    use std::sync::atomic::Ordering;
    let mut fixture = HttpFixture::new(&[Some(6)]).await;
    let stream = fixture.stream(0);
    let root = tempfile::tempdir().unwrap();
    let config = test_config(&fixture.url, root.path().to_path_buf());
    let service = service(&config);
    let request = download(&service);
    stream.send(Ok(b"abc".to_vec())).await.unwrap();
    let status = wait_status(&service, |status| status.bytes_received == 3).await;
    assert_eq!(status.state, ModelDownloadState::Downloading);
    assert_eq!(status.total_bytes, Some(6));
    let partial = partial_download_path(&config.checkpoint_path("medical_sam3"));
    assert!(partial.exists());
    let duplicate = download(&service).await.unwrap().unwrap_err();
    assert!(matches!(duplicate, crate::error::AppError::Conflict(_)));
    assert!(partial.exists());
    assert_eq!(fixture.requests.load(Ordering::SeqCst), 1);
    stream.send(Ok(b"def".to_vec())).await.unwrap();
    drop(stream);
    assert_eq!(request.await.unwrap().unwrap().status, "success");
    let status = wait_status(&service, |status| {
        status.state == ModelDownloadState::Available
    })
    .await;
    assert_eq!(status.bytes_received, 6);
    assert_eq!(status.total_bytes, Some(6));
    assert_eq!(
        tokio::fs::read(config.checkpoint_path("medical_sam3"))
            .await
            .unwrap(),
        b"abcdef"
    );
    assert!(!partial.exists());
}

#[tokio::test]
async fn chunked_download_has_unknown_total_and_outlives_cancelled_caller() {
    let mut fixture = HttpFixture::new(&[None]).await;
    let stream = fixture.stream(0);
    let root = tempfile::tempdir().unwrap();
    let config = test_config(&fixture.url, root.path().to_path_buf());
    let service = service(&config);
    let request = download(&service);
    stream.send(Ok(b"first".to_vec())).await.unwrap();
    let status = wait_status(&service, |status| status.bytes_received == 5).await;
    assert_eq!(status.total_bytes, None);
    assert_eq!(status.state, ModelDownloadState::Downloading);
    request.abort();
    assert!(request.await.unwrap_err().is_cancelled());
    stream.send(Ok(b"second".to_vec())).await.unwrap();
    drop(stream);
    let status = wait_status(&service, |status| {
        status.state == ModelDownloadState::Available
    })
    .await;
    assert_eq!(status.bytes_received, 11);
    assert_eq!(
        tokio::fs::read(config.checkpoint_path("medical_sam3"))
            .await
            .unwrap(),
        b"firstsecond"
    );
}

#[tokio::test]
async fn truncated_stream_cleans_partial_retains_failure_and_can_retry() {
    let mut fixture = HttpFixture::new(&[Some(10), Some(6)]).await;
    let failed_stream = fixture.stream(0);
    let retry_stream = fixture.stream(1);
    let root = tempfile::tempdir().unwrap();
    let config = test_config(&fixture.url, root.path().to_path_buf());
    let service = service(&config);
    let request = download(&service);
    failed_stream.send(Ok(b"abc".to_vec())).await.unwrap();
    wait_status(&service, |status| status.bytes_received == 3).await;
    drop(failed_stream);
    assert!(request.await.unwrap().is_err());
    let partial = partial_download_path(&config.checkpoint_path("medical_sam3"));
    assert!(!partial.exists());
    for _ in 0..2 {
        let status = wait_status(&service, |status| {
            status.state == ModelDownloadState::Failed
        })
        .await;
        assert_eq!(status.bytes_received, 3);
        assert_eq!(status.total_bytes, Some(10));
        assert!(status.error.is_some());
    }
    let retry = download(&service);
    retry_stream.send(Ok(b"de".to_vec())).await.unwrap();
    let status = wait_status(&service, |status| status.bytes_received == 2).await;
    assert_eq!(status.total_bytes, Some(6));
    assert_eq!(status.error, None);
    retry_stream.send(Ok(b"fghi".to_vec())).await.unwrap();
    drop(retry_stream);
    assert_eq!(retry.await.unwrap().unwrap().status, "success");
    assert_eq!(
        tokio::fs::read(config.checkpoint_path("medical_sam3"))
            .await
            .unwrap(),
        b"defghi"
    );
}

#[tokio::test]
async fn rename_failure_is_not_available_and_cleans_partial() {
    let mut fixture = HttpFixture::new(&[Some(6)]).await;
    let stream = fixture.stream(0);
    let root = tempfile::tempdir().unwrap();
    let config = test_config(&fixture.url, root.path().to_path_buf());
    let service = service(&config);
    let request = download(&service);
    stream.send(Ok(b"abc".to_vec())).await.unwrap();
    wait_status(&service, |status| status.bytes_received == 3).await;
    let target = config.checkpoint_path("medical_sam3");
    tokio::fs::create_dir(&target).await.unwrap();
    stream.send(Ok(b"def".to_vec())).await.unwrap();
    drop(stream);
    assert!(request.await.unwrap().is_err());
    let status = wait_status(&service, |status| {
        status.state == ModelDownloadState::Failed
    })
    .await;
    assert_eq!(status.bytes_received, 6);
    assert!(!partial_download_path(&target).exists());
    assert!(target.is_dir());
}

#[tokio::test]
async fn active_writer_is_not_overridden_by_external_target_and_deleted_install_becomes_idle() {
    let mut fixture = HttpFixture::new(&[Some(6)]).await;
    let stream = fixture.stream(0);
    let root = tempfile::tempdir().unwrap();
    let config = test_config(&fixture.url, root.path().to_path_buf());
    let service = service(&config);
    let request = download(&service);
    stream.send(Ok(b"abc".to_vec())).await.unwrap();
    wait_status(&service, |status| status.bytes_received == 3).await;
    let target = config.checkpoint_path("medical_sam3");
    tokio::fs::write(&target, b"external").await.unwrap();
    let status = service.model_status().await.unwrap();
    assert_eq!(
        status.downloads["medical_sam3"].state,
        ModelDownloadState::Downloading
    );
    assert!(!status.models["medical_sam3"]);
    tokio::fs::remove_file(&target).await.unwrap();
    stream.send(Ok(b"def".to_vec())).await.unwrap();
    drop(stream);
    request.await.unwrap().unwrap();
    tokio::fs::remove_file(&target).await.unwrap();
    let status = service.model_status().await.unwrap();
    assert_eq!(
        status.downloads["medical_sam3"].state,
        ModelDownloadState::Idle
    );
    assert!(!status.models["medical_sam3"]);
}

#[tokio::test]
async fn failed_flush_stays_unavailable_and_uses_owner_failure_cleanup() {
    use std::{
        pin::Pin,
        task::{Context, Poll},
    };
    use tokio::io::AsyncWrite;
    struct FlushFailure(tokio::fs::File);
    impl AsyncWrite for FlushFailure {
        fn poll_write(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            bytes: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            Pin::new(&mut self.0).poll_write(cx, bytes)
        }
        fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Err(std::io::Error::other("injected flush failure")))
        }
        fn poll_shutdown(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
        ) -> Poll<std::io::Result<()>> {
            Pin::new(&mut self.0).poll_shutdown(cx)
        }
    }
    let mut fixture = HttpFixture::new(&[Some(3)]).await;
    let stream = fixture.stream(0);
    stream.send(Ok(b"abc".to_vec())).await.unwrap();
    drop(stream);
    let root = tempfile::tempdir().unwrap();
    let config = test_config(&fixture.url, root.path().to_path_buf());
    let service = service(&config);
    let target = config.checkpoint_path("medical_sam3");
    let partial = prepare_download_target(&target).await.unwrap();
    service.downloads.lock().await.insert(
        "medical_sam3".into(),
        crate::domain::responses::ModelDownloadStatus {
            state: ModelDownloadState::Connecting,
            ..Default::default()
        },
    );
    let response = service.client.get(&fixture.url).send().await.unwrap();
    let mut writer = FlushFailure(tokio::fs::File::create(&partial).await.unwrap());
    let result = service
        .write_response("medical_sam3", response, &mut writer)
        .await;
    assert!(result
        .as_ref()
        .unwrap_err()
        .to_string()
        .contains("injected flush failure"));
    let before_cleanup = service.model_status().await.unwrap();
    assert_eq!(
        before_cleanup.downloads["medical_sam3"].state,
        ModelDownloadState::Finalizing
    );
    assert_eq!(before_cleanup.downloads["medical_sam3"].bytes_received, 3);
    assert!(!before_cleanup.models["medical_sam3"]);
    drop(writer);
    service
        .finish_download("medical_sam3", &target, &result)
        .await;
    let after_cleanup = service.model_status().await.unwrap();
    assert_eq!(
        after_cleanup.downloads["medical_sam3"].state,
        ModelDownloadState::Failed
    );
    assert!(!after_cleanup.models["medical_sam3"]);
    assert!(!partial.exists());
}

#[tokio::test]
async fn upstream_failure_body_and_url_credentials_are_not_persisted() {
    use axum::{routing::get, Router};
    let secret = "hf_DO_NOT_PERSIST";
    let app = Router::new().fallback(get(
        move || async move { (StatusCode::UNAUTHORIZED, secret) },
    ));
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(
        axum::Server::from_tcp(listener)
            .unwrap()
            .serve(app.into_make_service()),
    );
    let root = tempfile::tempdir().unwrap();
    let config = test_config(&format!("http://{address}"), root.path().to_path_buf());
    let service = service(&config);
    let error = service
        .download_model(&DownloadRequest {
            model_type: "medical_sam3".into(),
            hf_token: Some(secret.into()),
        })
        .await
        .unwrap_err();
    assert!(error.to_string().contains("401"));
    assert!(!error.to_string().contains(secret));
    let serialized = serde_json::to_string(&service.model_status().await.unwrap()).unwrap();
    assert!(!serialized.contains(secret));
    let network_error = service
        .client
        .get(format!("http://{address}/{secret}"))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap_err();
    assert!(!super::safe_http_error(network_error)
        .to_string()
        .contains(secret));
    server.abort();
}

#[tokio::test]
async fn every_supported_model_uses_the_same_reservation_before_partial_mutation() {
    let root = tempfile::tempdir().unwrap();
    let config = test_config("http://127.0.0.1:1", root.path().to_path_buf());
    let service = service(&config);
    for model in [
        "sam2_hiera_tiny",
        "sam2_hiera_small",
        "sam2_hiera_base_plus",
        "sam2_hiera_large",
        "sam3",
        "medical_sam3",
    ] {
        assert!(config.model_descriptor(model).is_some());
        let target = config.checkpoint_path(model);
        let partial = prepare_download_target(&target).await.unwrap();
        tokio::fs::write(&partial, b"owned by active writer")
            .await
            .unwrap();
        service.downloads.lock().await.insert(
            model.into(),
            crate::domain::responses::ModelDownloadStatus {
                state: ModelDownloadState::Connecting,
                ..Default::default()
            },
        );
        let error = service
            .download_model(&DownloadRequest {
                model_type: model.into(),
                hf_token: None,
            })
            .await
            .unwrap_err();
        assert!(matches!(error, crate::error::AppError::Conflict(_)));
        assert_eq!(
            tokio::fs::read(&partial).await.unwrap(),
            b"owned by active writer"
        );
    }
    let status = service.model_status().await.unwrap();
    assert_eq!(status.models.len(), config.tracked_model_status().len());
    assert_eq!(status.downloads.len(), status.models.len());
}
