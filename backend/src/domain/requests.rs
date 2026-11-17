use serde::{Deserialize, Serialize};

use crate::error::AppError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadRequest {
    pub model_type: String,
    pub hf_token: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct NegativeBorderPoints {
    pub count: usize,
    pub inset_px: f32,
}

impl Default for NegativeBorderPoints {
    fn default() -> Self {
        Self {
            count: 15,
            inset_px: 12.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct FrameRepairOptions {
    pub max_area_ratio: f64,
    pub min_iou: f64,
    pub keep_unrepaired: bool,
}

impl Default for FrameRepairOptions {
    fn default() -> Self {
        Self {
            max_area_ratio: 1.5,
            min_iou: 0.5,
            keep_unrepaired: false,
        }
    }
}

impl FrameRepairOptions {
    pub fn validate(&self) -> Result<(), AppError> {
        if !self.max_area_ratio.is_finite() || self.max_area_ratio < 1.0 {
            return Err(AppError::bad_request(
                "repair_inconsistent_frames.max_area_ratio must be finite and at least 1",
            ));
        }
        if !self.min_iou.is_finite() || !(0.0..=1.0).contains(&self.min_iou) {
            return Err(AppError::bad_request(
                "repair_inconsistent_frames.min_iou must be finite and between 0 and 1",
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
    pub repair_inconsistent_frames: Option<FrameRepairOptions>,
    #[serde(default)]
    pub max_prompt_gap: Option<usize>,
    #[serde(default)]
    pub negative_border_points: Option<NegativeBorderPoints>,
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
        if let Some(options) = self.repair_inconsistent_frames {
            options.validate()?;
        }
        if let Some(points) = self.negative_border_points {
            if points.count == 0 {
                return Err(AppError::bad_request(
                    "negative_border_points.count must be a positive integer",
                ));
            }
            if !points.inset_px.is_finite() || points.inset_px < 0.0 {
                return Err(AppError::bad_request(
                    "negative_border_points.inset_px must be finite and nonnegative",
                ));
            }
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
    use super::{FrameRepairOptions, NegativeBorderPoints, ProcessRequest};
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
    fn negative_border_points_default_off_and_allow_enabled_defaults() {
        let original = json!({
            "file_path": "/input.tif",
            "output_file": "/output.tif",
            "model_type": "sam3",
            "predictor_type": "VideoPredictor"
        });
        let request: ProcessRequest = serde_json::from_value(original.clone()).unwrap();
        assert_eq!(request.negative_border_points, None);
        for (value, expected) in [
            (json!(null), None),
            (json!({}), Some(NegativeBorderPoints::default())),
            (
                json!({ "count": 3 }),
                Some(NegativeBorderPoints {
                    count: 3,
                    inset_px: 12.0,
                }),
            ),
            (
                json!({ "inset_px": 0.5 }),
                Some(NegativeBorderPoints {
                    count: 15,
                    inset_px: 0.5,
                }),
            ),
            (
                json!({ "count": 20, "inset_px": 0 }),
                Some(NegativeBorderPoints {
                    count: 20,
                    inset_px: 0.0,
                }),
            ),
        ] {
            let mut body = original.clone();
            body["negative_border_points"] = value;
            let request: ProcessRequest = serde_json::from_value(body).unwrap();
            request.validate().unwrap();
            assert_eq!(request.negative_border_points, expected);
        }
    }

    #[test]
    fn invalid_negative_border_point_settings_are_rejected() {
        let original = json!({
            "file_path": "/input.tif",
            "output_file": "/output.tif",
            "model_type": "sam3",
            "predictor_type": "ImagePredictor"
        });
        for value in [
            json!({ "count": -1 }),
            json!({ "count": 1.5 }),
            json!({ "count": "15" }),
            json!({ "inset_px": "12" }),
        ] {
            let mut body = original.clone();
            body["negative_border_points"] = value;
            assert!(serde_json::from_value::<ProcessRequest>(body).is_err());
        }

        let mut request: ProcessRequest = serde_json::from_value(original).unwrap();
        request.negative_border_points = Some(NegativeBorderPoints {
            count: 0,
            inset_px: 12.0,
        });
        assert!(request
            .validate()
            .unwrap_err()
            .to_string()
            .contains("count must be a positive integer"));
        for inset_px in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            request.negative_border_points = Some(NegativeBorderPoints {
                count: 15,
                inset_px,
            });
            assert!(request
                .validate()
                .unwrap_err()
                .to_string()
                .contains("inset_px must be finite and nonnegative"));
        }
    }

    #[test]
    fn frame_repair_defaults_off_and_accepts_enabled_defaults_or_overrides() {
        for predictor in ["ImagePredictor", "VideoPredictor"] {
            let original = json!({
                "file_path": "/input.tif",
                "output_file": "/output.tif",
                "model_type": "sam3",
                "predictor_type": predictor
            });
            let request: ProcessRequest = serde_json::from_value(original.clone()).unwrap();
            assert_eq!(request.repair_inconsistent_frames, None);
            for (value, expected) in [
                (json!(null), None),
                (json!({}), Some(FrameRepairOptions::default())),
                (
                    json!({ "keep_unrepaired": true }),
                    Some(FrameRepairOptions {
                        keep_unrepaired: true,
                        ..FrameRepairOptions::default()
                    }),
                ),
                (
                    json!({ "max_area_ratio": 1, "min_iou": 0 }),
                    Some(FrameRepairOptions {
                        max_area_ratio: 1.0,
                        min_iou: 0.0,
                        keep_unrepaired: false,
                    }),
                ),
                (
                    json!({ "max_area_ratio": 2.25, "min_iou": 1 }),
                    Some(FrameRepairOptions {
                        max_area_ratio: 2.25,
                        min_iou: 1.0,
                        keep_unrepaired: false,
                    }),
                ),
            ] {
                let mut body = original.clone();
                body["repair_inconsistent_frames"] = value;
                let request: ProcessRequest = serde_json::from_value(body).unwrap();
                request.validate().unwrap();
                assert_eq!(request.repair_inconsistent_frames, expected);
            }
        }
    }

    #[test]
    fn frame_repair_settings_use_shared_validation_for_invalid_thresholds() {
        let original = json!({
            "file_path": "/input.tif",
            "output_file": "/output.tif",
            "model_type": "sam3",
            "predictor_type": "ImagePredictor"
        });
        for value in [
            json!({ "max_area_ratio": "1.5" }),
            json!({ "min_iou": "0.5" }),
            json!({ "keep_unrepaired": 1 }),
        ] {
            let mut body = original.clone();
            body["repair_inconsistent_frames"] = value;
            assert!(serde_json::from_value::<ProcessRequest>(body).is_err());
        }

        let mut request: ProcessRequest = serde_json::from_value(original).unwrap();
        for max_area_ratio in [0.0, 0.99, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let options = FrameRepairOptions {
                max_area_ratio,
                ..FrameRepairOptions::default()
            };
            let error = options.validate().unwrap_err().to_string();
            assert!(error.contains("max_area_ratio must be finite and at least 1"));
            request.repair_inconsistent_frames = Some(options);
            assert_eq!(request.validate().unwrap_err().to_string(), error);
        }
        for min_iou in [-0.01, 1.01, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let options = FrameRepairOptions {
                min_iou,
                ..FrameRepairOptions::default()
            };
            let error = options.validate().unwrap_err().to_string();
            assert!(error.contains("min_iou must be finite and between 0 and 1"));
            request.repair_inconsistent_frames = Some(options);
            assert_eq!(request.validate().unwrap_err().to_string(), error);
        }
    }
}
