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
    pub keep_centerline_component: bool,
    #[serde(default)]
    pub max_prompt_gap: Option<usize>,
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

#[cfg(test)]
mod tests {
    use super::ProcessRequest;
    use serde_json::json;

    #[test]
    fn prompt_gap_defaults_off_and_accepts_only_unsigned_integers_or_null() {
        let original = json!({
            "file_path": "/input.tif",
            "output_file": "/output.tif",
            "model_type": "sam3",
            "predictor_type": "VideoPredictor"
        });
        let request: ProcessRequest = serde_json::from_value(original.clone()).unwrap();
        assert_eq!(request.max_prompt_gap, None);
        for (value, expected) in [
            (json!(null), None),
            (json!(0), Some(0)),
            (json!(64), Some(64)),
        ] {
            let mut body = original.clone();
            body["max_prompt_gap"] = value;
            let request: ProcessRequest = serde_json::from_value(body).unwrap();
            request.validate().unwrap();
            assert_eq!(request.max_prompt_gap, expected);
        }
        for value in [json!(-1), json!(1.5), json!("64")] {
            let mut body = original.clone();
            body["max_prompt_gap"] = value;
            assert!(serde_json::from_value::<ProcessRequest>(body).is_err());
        }
    }
}
