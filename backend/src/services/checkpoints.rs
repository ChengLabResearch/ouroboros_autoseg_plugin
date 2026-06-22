use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::Duration,
};

use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;

use crate::{
    config::{AppConfig, DownloadSource},
    domain::{
        requests::DownloadRequest,
        responses::{DownloadModelResponse, ModelStatusResponse},
    },
    error::AppError,
};

const CHECKPOINT_DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(60 * 60);

pub async fn model_status(config: &AppConfig) -> Result<ModelStatusResponse, AppError> {
    let mut models = HashMap::new();
    for model_name in config.tracked_model_status() {
        let path = config.checkpoint_path(model_name);
        models.insert((*model_name).to_string(), file_exists(&path).await?);
    }

    Ok(ModelStatusResponse {
        checkpoint_dir: config.checkpoint_dir.display().to_string(),
        models,
    })
}

pub async fn download_model(
    config: &AppConfig,
    client: &reqwest::Client,
    request: &DownloadRequest,
) -> Result<DownloadModelResponse, AppError> {
    let descriptor = config
        .model_descriptor(&request.model_type)
        .ok_or_else(|| AppError::bad_request("Unknown model type"))?;

    config.ensure_checkpoint_dir()?;
    let target_path = config.checkpoint_path(&request.model_type);

    if file_exists(&target_path).await? {
        return Ok(DownloadModelResponse {
            status: "exists".to_string(),
            message: format!("Model {} already exists.", request.model_type),
        });
    }

    if path_is_directory(&target_path).await? {
        tokio::fs::remove_dir_all(&target_path).await?;
    }

    match descriptor.download_source {
        DownloadSource::PublicUrl(url) => {
            download_to_path(client, url, None, &target_path).await?;
        }
        DownloadSource::HuggingFace {
            repo,
            filename,
            requires_token,
        } => {
            let token = request
                .hf_token
                .as_deref()
                .filter(|token| !token.trim().is_empty());
            if requires_token && token.is_none() {
                return Err(AppError::bad_request(
                    "Authentication Token required for SAM3 (Official)",
                ));
            }
            let url = config.huggingface_resolve_url(repo, filename);
            download_to_path(client, &url, token, &target_path).await?;
        }
    }

    Ok(DownloadModelResponse {
        status: "success".to_string(),
        message: format!("Downloaded {}", request.model_type),
    })
}

async fn download_to_path(
    client: &reqwest::Client,
    url: &str,
    bearer_token: Option<&str>,
    target_path: &Path,
) -> Result<(), AppError> {
    let partial_path = prepare_download_target(target_path).await?;

    let mut request = client.get(url).timeout(CHECKPOINT_DOWNLOAD_TIMEOUT);
    if let Some(token) = bearer_token {
        request = request.bearer_auth(token);
    }

    let response = request.send().await?;
    if !response.status().is_success() {
        let status = response.status();
        let message = response.text().await.unwrap_or_default();
        return Err(AppError::upstream(download_failure_message(
            status, &message,
        )));
    }

    let mut file = tokio::fs::File::create(&partial_path).await?;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let bytes = match chunk {
            Ok(bytes) => bytes,
            Err(error) => {
                let _ = tokio::fs::remove_file(&partial_path).await;
                return Err(AppError::Http(error));
            }
        };
        if let Err(error) = file.write_all(&bytes).await {
            let _ = tokio::fs::remove_file(&partial_path).await;
            return Err(AppError::Io(error));
        }
    }
    if let Err(error) = file.flush().await {
        let _ = tokio::fs::remove_file(&partial_path).await;
        return Err(AppError::Io(error));
    }
    drop(file);
    tokio::fs::rename(&partial_path, target_path).await?;
    Ok(())
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

fn download_failure_message(status: reqwest::StatusCode, message: &str) -> String {
    if message.trim().is_empty() {
        format!("Failed to download checkpoint: {status}")
    } else {
        format!("Failed to download checkpoint: {status} ({message})")
    }
}

async fn file_exists(path: &Path) -> Result<bool, AppError> {
    match tokio::fs::metadata(path).await {
        Ok(metadata) => Ok(metadata.is_file()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
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
