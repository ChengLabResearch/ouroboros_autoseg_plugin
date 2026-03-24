use serde::{Deserialize, Serialize};

use crate::error::AppError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadRequest {
    pub model_type: String,
    pub hf_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessRequest {
    pub file_path: String,
    pub output_file: String,
    pub model_type: String,
    pub predictor_type: String,
    #[serde(default)]
    pub overlay_annotation_points: bool,
    #[serde(default = "default_annotation_overlay_intensity")]
    pub annotation_overlay_intensity: u8,
}

impl ProcessRequest {
    pub fn validate(&self) -> Result<(), AppError> {
        if self.file_path.trim().is_empty() {
            return Err(AppError::bad_request("file_path must not be empty"));
        }
        if self.output_file.trim().is_empty() {
            return Err(AppError::bad_request("output_file must not be empty"));
        }
        if self.model_type.trim().is_empty() {
            return Err(AppError::bad_request("model_type must not be empty"));
        }

        match self.predictor_type.as_str() {
            "ImagePredictor" | "VideoPredictor" => Ok(()),
            _ => Err(AppError::bad_request(format!(
                "Unknown predictor type: {}",
                self.predictor_type
            ))),
        }
    }
}

const fn default_annotation_overlay_intensity() -> u8 {
    127
}
