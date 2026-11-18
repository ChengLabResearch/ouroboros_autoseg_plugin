use serde::{Deserialize, Serialize};

use crate::error::AppError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadRequest {
    pub model_type: String,
    pub hf_token: Option<String>,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct InputWindowOptions {
    pub margin: usize,
    pub max_growth: usize,
    pub window_size: Option<usize>,
}

impl Default for InputWindowOptions {
    fn default() -> Self {
        Self {
            margin: 8,
            max_growth: 5,
            window_size: None,
        }
    }
}

impl InputWindowOptions {
    pub fn validate(&self) -> Result<(), AppError> {
        if self.window_size == Some(0) {
            return Err(AppError::bad_request(
                "input_window.window_size must be positive or null",
            ));
        }
        Ok(())
    }
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
    pub repair_inconsistent_frames: bool,
    #[serde(default)]
    pub keep_unrepaired: bool,
    #[serde(default)]
    pub max_prompt_gap: Option<usize>,
    #[serde(default)]
    pub input_window: Option<InputWindowOptions>,
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
        if let Some(options) = self.input_window {
            options.validate()?;
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
    use super::{InputWindowOptions, ProcessRequest};
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

    #[test]
    fn frame_repair_booleans_default_off_and_accept_all_combinations() {
        for predictor in ["ImagePredictor", "VideoPredictor"] {
            let original = json!({
                "file_path": "/input.tif",
                "output_file": "/output.tif",
                "model_type": "sam3",
                "predictor_type": predictor
            });
            let request: ProcessRequest = serde_json::from_value(original.clone()).unwrap();
            assert!(!request.repair_inconsistent_frames);
            assert!(!request.keep_unrepaired);
            for repair in [false, true] {
                for keep_unrepaired in [false, true] {
                    let mut body = original.clone();
                    body["repair_inconsistent_frames"] = json!(repair);
                    body["keep_unrepaired"] = json!(keep_unrepaired);
                    let request: ProcessRequest = serde_json::from_value(body).unwrap();
                    request.validate().unwrap();
                    assert_eq!(request.repair_inconsistent_frames, repair);
                    assert_eq!(request.keep_unrepaired, keep_unrepaired);
                }
            }
        }
    }

    #[test]
    fn frame_repair_options_accept_only_booleans() {
        let original = json!({
            "file_path": "/input.tif",
            "output_file": "/output.tif",
            "model_type": "sam3",
            "predictor_type": "ImagePredictor"
        });
        for key in ["repair_inconsistent_frames", "keep_unrepaired"] {
            for value in [json!(null), json!({}), json!(1), json!("true")] {
                let mut body = original.clone();
                body[key] = value;
                assert!(serde_json::from_value::<ProcessRequest>(body).is_err());
            }
        }
    }

    #[test]
    fn input_window_defaults_off_and_an_empty_block_enables_defaults() {
        for predictor in ["ImagePredictor", "VideoPredictor"] {
            let original = json!({
                "file_path": "/input.tif",
                "output_file": "/output.tif",
                "model_type": "sam3",
                "predictor_type": predictor
            });
            let request: ProcessRequest = serde_json::from_value(original.clone()).unwrap();
            assert_eq!(request.input_window, None);
            for (value, expected) in [
                (json!(null), None),
                (json!({}), Some(InputWindowOptions::default())),
                (
                    json!({"margin": 0, "max_growth": 0, "window_size": 1}),
                    Some(InputWindowOptions {
                        margin: 0,
                        max_growth: 0,
                        window_size: Some(1),
                    }),
                ),
                (
                    json!({"margin": 12, "window_size": null}),
                    Some(InputWindowOptions {
                        margin: 12,
                        ..InputWindowOptions::default()
                    }),
                ),
            ] {
                let mut body = original.clone();
                body["input_window"] = value;
                let request: ProcessRequest = serde_json::from_value(body).unwrap();
                request.validate().unwrap();
                assert_eq!(request.input_window, expected);
            }
            let mut body = original;
            body["input_window"] = json!({"window_size": 0});
            let request: ProcessRequest = serde_json::from_value(body).unwrap();
            assert!(request.validate().is_err());
        }
    }

    #[test]
    fn input_window_dimensions_accept_only_unsigned_integers() {
        for key in ["margin", "max_growth", "window_size"] {
            for value in [json!(-1), json!(1.5), json!("8"), json!(false)] {
                let mut options = json!({});
                options[key] = value;
                assert!(serde_json::from_value::<InputWindowOptions>(options).is_err());
            }
        }
        for key in ["margin", "max_growth"] {
            let mut options = json!({});
            options[key] = json!(null);
            assert!(serde_json::from_value::<InputWindowOptions>(options).is_err());
        }
        InputWindowOptions {
            margin: usize::MAX,
            max_growth: usize::MAX,
            window_size: Some(usize::MAX),
        }
        .validate()
        .unwrap();
    }
}
