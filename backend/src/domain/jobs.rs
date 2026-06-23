use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStep {
    pub name: String,
    pub progress: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub status: String,
    pub steps: Vec<JobStep>,
    pub created_at: f64,
    pub updated_at: f64,
}

impl JobRecord {
    pub fn new_running() -> Self {
        let now = unix_timestamp();
        Self {
            status: "running".to_string(),
            steps: vec![
                JobStep {
                    name: "Transferring".to_string(),
                    progress: 0,
                },
                JobStep {
                    name: "Inference".to_string(),
                    progress: 0,
                },
                JobStep {
                    name: "Saving".to_string(),
                    progress: 0,
                },
            ],
            created_at: now,
            updated_at: now,
        }
    }

    pub fn update_step(&mut self, step_index: usize, progress: u8) {
        if let Some(step) = self.steps.get_mut(step_index) {
            step.progress = progress.min(100);
            self.updated_at = unix_timestamp();
        }
    }

    pub fn set_status(&mut self, status: &str) {
        self.status = status.to_string();
        self.updated_at = unix_timestamp();
    }
}

pub fn unix_timestamp() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}
