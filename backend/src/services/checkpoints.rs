use std::{collections::HashMap, path::Path};

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
        DownloadSource::HuggingFace { repo, filename } => {
            let token = request
                .hf_token
                .as_deref()
                .filter(|token| !token.trim().is_empty())
                .ok_or_else(|| AppError::bad_request("Authentication Token required for SAM 3"))?;
            let url = format!("https://huggingface.co/{repo}/resolve/main/{filename}");
            download_to_path(client, &url, Some(token), &target_path).await?;
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
    let mut request = client.get(url);
    if let Some(token) = bearer_token {
        request = request.bearer_auth(token);
    }

    let response = request.send().await?;
    if !response.status().is_success() {
        return Err(AppError::upstream(format!(
            "Failed to download checkpoint: {}",
            response.status()
        )));
    }

    let mut file = tokio::fs::File::create(target_path).await?;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let bytes = chunk?;
        file.write_all(&bytes).await?;
    }
    file.flush().await?;
    Ok(())
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
