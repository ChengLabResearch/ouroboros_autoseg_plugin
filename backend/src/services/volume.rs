use serde::{Deserialize, Serialize};

use crate::{config::AppConfig, error::AppError};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VolumeFileMapping {
    pub source_path: String,
    pub target_path: String,
}

impl VolumeFileMapping {
    pub fn new(source_path: impl Into<String>, target_path: impl Into<String>) -> Self {
        Self {
            source_path: source_path.into(),
            target_path: target_path.into(),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VolumeOperationRequest<'a> {
    volume_name: &'a str,
    plugin_folder_name: &'a str,
    files: &'a [VolumeFileMapping],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VolumeOperationOk {
    pub message: String,
}

pub async fn ping(client: &reqwest::Client, config: &AppConfig) -> Result<(), AppError> {
    let response = client
        .get(config.volume_server_endpoint("/"))
        .send()
        .await?;
    if is_ping_status_healthy(response.status()) {
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
) -> Result<VolumeOperationOk, AppError> {
    perform_operation(client, config, "copy-to-volume", files).await
}

pub async fn copy_to_host(
    client: &reqwest::Client,
    config: &AppConfig,
    files: &[VolumeFileMapping],
) -> Result<VolumeOperationOk, AppError> {
    perform_operation(client, config, "copy-to-host", files).await
}

async fn perform_operation(
    client: &reqwest::Client,
    config: &AppConfig,
    path: &str,
    files: &[VolumeFileMapping],
) -> Result<VolumeOperationOk, AppError> {
    let url = config.volume_server_endpoint(path);
    let request = build_operation_request(config, files);

    let response = client.post(url).json(&request).send().await?;
    let status = response.status();
    let message = response.text().await.unwrap_or_default();
    if status.is_success() {
        Ok(VolumeOperationOk { message })
    } else {
        Err(AppError::upstream(volume_failure_message(status, &message)))
    }
}

fn is_ping_status_healthy(status: reqwest::StatusCode) -> bool {
    status.is_success()
        || status == reqwest::StatusCode::NOT_FOUND
        || status == reqwest::StatusCode::METHOD_NOT_ALLOWED
}

fn build_operation_request<'a>(
    config: &'a AppConfig,
    files: &'a [VolumeFileMapping],
) -> VolumeOperationRequest<'a> {
    VolumeOperationRequest {
        volume_name: &config.volume_name,
        plugin_folder_name: &config.plugin_name,
        files,
    }
}

fn volume_failure_message(status: reqwest::StatusCode, message: &str) -> String {
    if message.trim().is_empty() {
        format!("Volume operation failed with status {status}")
    } else {
        message.to_string()
    }
}

#[cfg(test)]
mod tests;
