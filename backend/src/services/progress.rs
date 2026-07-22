use std::collections::HashSet;

use tokio::sync::mpsc;

use crate::{app_state::AppState, domain::jobs::JobPhase};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressEvent {
    PhaseStarted(JobPhase),
    Frame {
        phase: JobPhase,
        frame_index: usize,
        total_frames: usize,
    },
}

pub type ProgressSender = mpsc::UnboundedSender<ProgressEvent>;

pub fn channel() -> (ProgressSender, mpsc::UnboundedReceiver<ProgressEvent>) {
    mpsc::unbounded_channel()
}

pub async fn consume(
    state: AppState,
    job_id: String,
    mut receiver: mpsc::UnboundedReceiver<ProgressEvent>,
) {
    let mut seen: [HashSet<usize>; 3] = std::array::from_fn(|_| HashSet::new());
    let mut reported = [0u8; 3];

    while let Some(event) = receiver.recv().await {
        match event {
            ProgressEvent::PhaseStarted(phase) => {
                state.update_job_phase(&job_id, phase, 0).await;
            }
            ProgressEvent::Frame {
                phase,
                frame_index,
                total_frames,
            } => {
                if total_frames == 0 || frame_index >= total_frames {
                    continue;
                }
                let phase_index = phase.index();
                if !seen[phase_index].insert(frame_index) {
                    continue;
                }
                let progress = ((seen[phase_index].len() * 100) / total_frames).min(100) as u8;
                if progress > reported[phase_index] {
                    reported[phase_index] = progress;
                    state.update_job_phase(&job_id, phase, progress).await;
                }
            }
        }
    }
}

pub fn report_phase(sender: &ProgressSender, phase: JobPhase) {
    let _ = sender.send(ProgressEvent::PhaseStarted(phase));
}

pub fn report_frame(
    sender: &ProgressSender,
    phase: JobPhase,
    frame_index: usize,
    total_frames: usize,
) {
    // A closed receiver means the job has already reached a terminal state.
    let _ = sender.send(ProgressEvent::Frame {
        phase,
        frame_index,
        total_frames,
    });
}

#[cfg(test)]
mod tests {
    use super::{channel, consume, report_frame, report_phase};
    use crate::{app_state::AppState, config::AppConfig, domain::jobs::JobPhase};
    use std::path::PathBuf;

    fn state() -> AppState {
        AppState::new(AppConfig {
            plugin_name: "test".into(),
            volume_name: "test".into(),
            volume_server_url: "http://127.0.0.1:3001".into(),
            huggingface_base_url: "https://huggingface.co".into(),
            checkpoint_dir: PathBuf::from("/tmp/test"),
            internal_volume_path: PathBuf::from("/tmp/test"),
            fallback_annotation_interval: 1,
            bind_address: "127.0.0.1:8686".parse().unwrap(),
        })
        .unwrap()
    }

    #[tokio::test]
    async fn counts_unique_frames_and_ignores_bursts_and_regressions() {
        let state = state();
        state.create_job("job".into()).await;
        let (sender, receiver) = channel();
        let task = tokio::spawn(consume(state.clone(), "job".into(), receiver));
        report_phase(&sender, JobPhase::Inference);
        for frame in [0, 0, 2, 1, 4, 3] {
            report_frame(&sender, JobPhase::Inference, frame, 5);
        }
        drop(sender);
        task.await.unwrap();

        let job = state.get_job("job").await.unwrap();
        assert_eq!(job.steps[0].progress, 100);
        assert_eq!(job.steps[1].progress, 100);
    }
}
