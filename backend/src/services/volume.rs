use serde::Serialize;

use crate::{config::AppConfig, error::AppError};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VolumeFileMapping {
    pub source_path: String,
    pub target_path: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VolumeOperationRequest<'a> {
    volume_name: &'a str,
    plugin_folder_name: &'a str,
    files: &'a [VolumeFileMapping],
}

pub async fn ping(client: &reqwest::Client, config: &AppConfig) -> Result<(), AppError> {
    let response = client
        .get(format!("{}/", config.volume_server_url))
        .send()
        .await?;
    if response.status().is_success()
        || response.status() == reqwest::StatusCode::NOT_FOUND
        || response.status() == reqwest::StatusCode::METHOD_NOT_ALLOWED
    {
        Ok(())
    } else {
        Err(AppError::upstream(format!(
            "Volume server returned {}",
            response.status()
        )))
    }
}

pub async fn copy_to_volume(
    client: &reqwest::Client,
    config: &AppConfig,
    files: &[VolumeFileMapping],
) -> Result<(), AppError> {
    perform_operation(client, config, "copy-to-volume", files).await
}

pub async fn copy_to_host(
    client: &reqwest::Client,
    config: &AppConfig,
    files: &[VolumeFileMapping],
) -> Result<(), AppError> {
    perform_operation(client, config, "copy-to-host", files).await
}

async fn perform_operation(
    client: &reqwest::Client,
    config: &AppConfig,
    path: &str,
    files: &[VolumeFileMapping],
) -> Result<(), AppError> {
    let url = format!("{}/{}", config.volume_server_url, path);
    let request = VolumeOperationRequest {
        volume_name: "ouroboros-volume",
        plugin_folder_name: &config.plugin_name,
        files,
    };

    let response = client.post(url).json(&request).send().await?;
    if response.status().is_success() {
        Ok(())
    } else {
        let message = response
            .text()
            .await
            .unwrap_or_else(|_| "volume operation failed".to_string());
        Err(AppError::upstream(message))
    }
}
