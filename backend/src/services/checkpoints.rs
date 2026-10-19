use std::{
    collections::HashMap,
    panic::AssertUnwindSafe,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use futures_util::{FutureExt, StreamExt};
use tokio::{
    io::{AsyncWrite, AsyncWriteExt},
    sync::Mutex,
};

use crate::{
    config::{AppConfig, DownloadSource},
    domain::{
        requests::DownloadRequest,
        responses::{
            DownloadModelResponse, ModelDownloadState, ModelDownloadStatus, ModelStatusResponse,
        },
    },
    error::AppError,
};

const CHECKPOINT_DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(60 * 60);

/// Owns checkpoint installation state for every catalog model. Clones share both
/// the reservation and the live status; HTTP responses are snapshots of this owner.
#[derive(Clone)]
pub struct CheckpointService {
    config: Arc<AppConfig>,
    client: reqwest::Client,
    downloads: Arc<Mutex<HashMap<String, ModelDownloadStatus>>>,
}

impl CheckpointService {
    pub fn new(config: Arc<AppConfig>, client: reqwest::Client) -> Self {
        Self {
            config,
            client,
            downloads: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn model_status(&self) -> Result<ModelStatusResponse, AppError> {
        let mut records = self.downloads.lock().await;
        let mut downloads = HashMap::new();
        for model_name in self.config.tracked_model_status() {
            let status = records.entry((*model_name).to_string()).or_default();
            self.reconcile_file(model_name, status).await?;
            downloads.insert((*model_name).to_string(), status.clone());
        }
        let models = downloads
            .iter()
            .map(|(name, status)| (name.clone(), status.state == ModelDownloadState::Available))
            .collect();

        Ok(ModelStatusResponse {
            checkpoint_dir: self.config.checkpoint_dir.display().to_string(),
            models,
            downloads,
        })
    }

    pub async fn download_model(
        &self,
        request: &DownloadRequest,
    ) -> Result<DownloadModelResponse, AppError> {
        let descriptor = self
            .config
            .model_descriptor(&request.model_type)
            .ok_or_else(|| AppError::bad_request("Unknown model type"))?;
        let mut records = self.downloads.lock().await;
        let status = records.entry(request.model_type.clone()).or_default();
        if status.state.is_active() {
            return Err(AppError::conflict(format!(
                "A download of {} is already in progress.",
                request.model_type
            )));
        }
        self.reconcile_file(&request.model_type, status).await?;
        if status.state == ModelDownloadState::Available {
            return Ok(DownloadModelResponse {
                status: "exists".to_string(),
                message: format!("Model {} already exists.", request.model_type),
            });
        }

        let token = match descriptor.download_source {
            DownloadSource::PublicUrl(_) => None,
            DownloadSource::HuggingFace { requires_token, .. } => {
                let token = request
                    .hf_token
                    .as_deref()
                    .filter(|token| !token.trim().is_empty());
                if requires_token && token.is_none() {
                    return Err(AppError::bad_request(
                        "Authentication Token required for SAM3 (Official)",
                    ));
                }
                token.map(str::to_owned)
            }
        };
        *status = ModelDownloadStatus {
            state: ModelDownloadState::Connecting,
            ..ModelDownloadStatus::default()
        };
        drop(records);

        // No await between reserving and spawning: dropping the HTTP caller can
        // neither cancel an installation nor leave an orphaned reservation.
        let service = self.clone();
        let model_name = request.model_type.clone();
        let url = descriptor
            .download_source
            .url(&self.config.huggingface_base_url);
        tokio::spawn(async move {
            let target_path = service.config.checkpoint_path(&model_name);
            let result = AssertUnwindSafe(service.install(
                &model_name,
                &url,
                token.as_deref(),
                &target_path,
            ))
            .catch_unwind()
            .await
            .unwrap_or_else(|_| Err(AppError::internal("Checkpoint download worker failed.")));

            service
                .finish_download(&model_name, &target_path, &result)
                .await;
            result.map(|()| DownloadModelResponse {
                status: "success".to_string(),
                message: format!("Downloaded {model_name}"),
            })
        })
        .await
        .map_err(|_| AppError::internal("Checkpoint download worker failed."))?
    }

    async fn finish_download(
        &self,
        model_name: &str,
        target_path: &Path,
        result: &Result<(), AppError>,
    ) {
        // install has returned (or unwound), so its file handle is closed
        // before cleanup, including on Windows where open files cannot unlink.
        if result.is_err() {
            let _ = tokio::fs::remove_file(partial_download_path(target_path)).await;
        }
        let mut records = self.downloads.lock().await;
        let status = records.get_mut(model_name).expect("reserved model");
        match result {
            Ok(()) => status.state = ModelDownloadState::Available,
            Err(error) => {
                status.state = ModelDownloadState::Failed;
                status.error = Some(error.to_string());
            }
        }
    }

    async fn reconcile_file(
        &self,
        model_name: &str,
        status: &mut ModelDownloadStatus,
    ) -> Result<(), AppError> {
        // Disk is an input only when no writer owns the checkpoint. In particular,
        // an externally created target must not hide an unfinished installation.
        if status.state.is_active() {
            return Ok(());
        }
        match checkpoint_size(&self.config.checkpoint_path(model_name)).await? {
            Some(size) => {
                *status = ModelDownloadStatus {
                    state: ModelDownloadState::Available,
                    bytes_received: size,
                    total_bytes: Some(size),
                    error: None,
                };
            }
            None if status.state == ModelDownloadState::Available => {
                *status = ModelDownloadStatus::default();
            }
            None => {}
        }
        Ok(())
    }

    async fn install(
        &self,
        model_name: &str,
        url: &str,
        bearer_token: Option<&str>,
        target_path: &Path,
    ) -> Result<(), AppError> {
        self.config.ensure_checkpoint_dir()?;
        if path_is_directory(target_path).await? {
            tokio::fs::remove_dir_all(target_path).await?;
        }
        let partial_path = prepare_download_target(target_path).await?;

        let mut request = self.client.get(url).timeout(CHECKPOINT_DOWNLOAD_TIMEOUT);
        if let Some(token) = bearer_token {
            request = request.bearer_auth(token);
        }
        let response = request.send().await.map_err(safe_http_error)?;
        if !response.status().is_success() {
            // Upstream bodies may echo credentials or signed URLs. Persist only
            // a status and our own actionable explanation.
            return Err(AppError::upstream(download_failure_message(
                response.status(),
            )));
        }
        let mut file = tokio::fs::File::create(&partial_path).await?;
        self.write_response(model_name, response, &mut file).await?;
        drop(file);
        tokio::fs::rename(&partial_path, target_path).await?;
        Ok(())
    }

    async fn write_response<W: AsyncWrite + Unpin>(
        &self,
        model_name: &str,
        response: reqwest::Response,
        file: &mut W,
    ) -> Result<(), AppError> {
        {
            let mut records = self.downloads.lock().await;
            let status = records.get_mut(model_name).expect("reserved model");
            status.state = ModelDownloadState::Downloading;
            status.total_bytes = response.content_length();
        }
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let bytes = chunk.map_err(safe_http_error)?;
            file.write_all(&bytes).await?;
            self.downloads
                .lock()
                .await
                .get_mut(model_name)
                .expect("reserved model")
                .bytes_received += bytes.len() as u64;
        }
        self.downloads
            .lock()
            .await
            .get_mut(model_name)
            .expect("reserved model")
            .state = ModelDownloadState::Finalizing;
        file.flush().await?;
        Ok(())
    }
}

fn safe_http_error(error: reqwest::Error) -> AppError {
    AppError::Http(error.without_url())
}

fn partial_download_path(target_path: &Path) -> PathBuf {
    let file_name = target_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("checkpoint");
    target_path.with_file_name(format!("{file_name}.part"))
}

async fn prepare_download_target(target_path: &Path) -> Result<PathBuf, AppError> {
    if let Some(parent) = target_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let partial_path = partial_download_path(target_path);
    match tokio::fs::metadata(&partial_path).await {
        Ok(metadata) if metadata.is_file() => {
            tokio::fs::remove_file(&partial_path).await?;
        }
        Ok(metadata) if metadata.is_dir() => {
            tokio::fs::remove_dir_all(&partial_path).await?;
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(AppError::Io(error)),
    }

    Ok(partial_path)
}

fn download_failure_message(status: reqwest::StatusCode) -> String {
    let advice = match status {
        reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => {
            " Check your Hugging Face token and access to the model."
        }
        reqwest::StatusCode::NOT_FOUND => " The checkpoint source is unavailable.",
        _ => " Try again later.",
    };
    format!("Failed to download checkpoint: {status}.{advice}")
}

async fn checkpoint_size(path: &Path) -> Result<Option<u64>, AppError> {
    match tokio::fs::metadata(path).await {
        Ok(metadata) => Ok(metadata.is_file().then_some(metadata.len())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(AppError::Io(error)),
    }
}

async fn path_is_directory(path: &Path) -> Result<bool, AppError> {
    match tokio::fs::metadata(path).await {
        Ok(metadata) => Ok(metadata.is_dir()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(AppError::Io(error)),
    }
}

#[cfg(test)]
mod tests;
