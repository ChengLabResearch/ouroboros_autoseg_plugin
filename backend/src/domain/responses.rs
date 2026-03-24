use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::domain::jobs::unix_timestamp;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializationStep {
    pub name: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartupStatus {
    pub is_ready: bool,
    pub initialization_steps: Vec<InitializationStep>,
    pub start_time: Option<f64>,
    pub ready_time: Option<f64>,
}

impl StartupStatus {
    pub fn pending() -> Self {
        Self {
            is_ready: false,
            initialization_steps: vec![
                InitializationStep {
                    name: "Building Docker Image".to_string(),
                    status: "pending".to_string(),
                },
                InitializationStep {
                    name: "Connecting to Volume Server".to_string(),
                    status: "pending".to_string(),
                },
                InitializationStep {
                    name: "Initializing ML Models".to_string(),
                    status: "pending".to_string(),
                },
            ],
            start_time: None,
            ready_time: None,
        }
    }

    pub fn set_step_status(&mut self, step_name: &str, status: &str) {
        if self.start_time.is_none() {
            self.start_time = Some(unix_timestamp());
        }

        if let Some(step) = self
            .initialization_steps
            .iter_mut()
            .find(|step| step.name == step_name)
        {
            step.status = status.to_string();
        }
    }

    pub fn recalculate_readiness(&mut self) {
        let all_completed = self
            .initialization_steps
            .iter()
            .all(|step| step.status == "completed");
        self.is_ready = all_completed;

        if all_completed {
            if self.ready_time.is_none() {
                self.ready_time = Some(unix_timestamp());
            }
        } else {
            self.ready_time = None;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatusResponse {
    pub checkpoint_dir: String,
    pub models: HashMap<String, bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadModelResponse {
    pub status: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobSubmissionResponse {
    pub job_id: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatestJobResponse {
    pub job_id: Option<String>,
    pub status: String,
}
