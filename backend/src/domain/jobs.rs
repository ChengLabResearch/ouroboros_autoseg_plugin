use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobPhase {
    Transferring,
    Inference,
    Saving,
}

impl JobPhase {
    pub const ALL: [Self; 3] = [Self::Transferring, Self::Inference, Self::Saving];

    pub const fn index(self) -> usize {
        match self {
            Self::Transferring => 0,
            Self::Inference => 1,
            Self::Saving => 2,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Transferring => "Transferring",
            Self::Inference => "Inference",
            Self::Saving => "Saving",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStep {
    pub name: String,
    pub progress: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub status: String,
    pub steps: Vec<JobStep>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_phase: Option<JobPhase>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub created_at: f64,
    pub updated_at: f64,
}

impl JobRecord {
    pub fn new_running() -> Self {
        let now = unix_timestamp();
        Self {
            status: "running".to_string(),
            steps: JobPhase::ALL
                .into_iter()
                .map(|phase| JobStep {
                    name: phase.name().to_string(),
                    progress: 0,
                })
                .collect(),
            active_phase: Some(JobPhase::Transferring),
            error: None,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn update_phase(&mut self, phase: JobPhase, progress: u8) {
        let phase_index = phase.index();
        for step in self.steps.iter_mut().take(phase_index) {
            step.progress = 100;
        }
        if let Some(step) = self.steps.get_mut(phase_index) {
            step.progress = step.progress.max(progress.min(100));
        }
        self.active_phase = Some(phase);
        self.updated_at = unix_timestamp();
    }

    pub fn complete(&mut self) {
        for step in &mut self.steps {
            step.progress = 100;
        }
        self.status = "completed".to_string();
        self.active_phase = None;
        self.error = None;
        self.updated_at = unix_timestamp();
    }

    pub fn fail(&mut self, message: impl Into<String>) {
        self.status = "error".to_string();
        self.error = Some(message.into());
        self.updated_at = unix_timestamp();
    }
}

#[cfg(test)]
mod tests {
    use super::{JobPhase, JobRecord};

    #[test]
    fn phase_progress_is_monotonic_and_completes_prior_phases() {
        let mut job = JobRecord::new_running();
        job.update_phase(JobPhase::Transferring, 72);
        job.update_phase(JobPhase::Transferring, 40);
        assert_eq!(job.steps[0].progress, 72);

        job.update_phase(JobPhase::Inference, 1);
        assert_eq!(job.steps[0].progress, 100);
        assert_eq!(job.steps[1].progress, 1);
        assert_eq!(job.active_phase, Some(JobPhase::Inference));
    }

    #[test]
    fn completion_and_failure_have_coherent_terminal_state() {
        let mut completed = JobRecord::new_running();
        completed.update_phase(JobPhase::Saving, 25);
        completed.complete();
        assert!(completed.steps.iter().all(|step| step.progress == 100));
        assert_eq!(completed.active_phase, None);

        let mut failed = JobRecord::new_running();
        failed.update_phase(JobPhase::Inference, 37);
        failed.fail("Inference: device unavailable");
        assert_eq!(failed.status, "error");
        assert_eq!(failed.active_phase, Some(JobPhase::Inference));
        assert_eq!(
            failed.error.as_deref(),
            Some("Inference: device unavailable")
        );
    }
}

pub fn unix_timestamp() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}
