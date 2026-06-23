use super::apply_probe_results;
use crate::domain::responses::StartupStatus;

#[test]
fn refresh_marks_volume_ready_and_ml_warning() {
    let mut status = StartupStatus::pending();

    apply_probe_results(&mut status, true, false);

    assert!(!status.is_ready);
    assert_eq!(status.initialization_steps[0].status, "completed");
    assert_eq!(status.initialization_steps[1].status, "completed");
    assert_eq!(status.initialization_steps[2].status, "warning");
}

#[test]
fn refresh_marks_volume_warning_when_ping_fails() {
    let mut status = StartupStatus::pending();

    apply_probe_results(&mut status, false, false);

    assert_eq!(status.initialization_steps[1].status, "warning");
    assert_eq!(status.initialization_steps[2].status, "warning");
}

#[test]
fn refresh_marks_ready_when_all_dependencies_are_ready() {
    let mut status = StartupStatus::pending();

    apply_probe_results(&mut status, true, true);

    assert!(status.is_ready);
    assert!(status.ready_time.is_some());
    assert_eq!(status.initialization_steps[0].status, "completed");
    assert_eq!(status.initialization_steps[1].status, "completed");
    assert_eq!(status.initialization_steps[2].status, "completed");
}
