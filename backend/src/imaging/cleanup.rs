use crate::{
    error::AppError,
    imaging::annotations::{interpolated_point_for_frame, AnnotationPoint},
    inference::image::FrameMask,
};

mod distance;
mod repair;

pub(crate) use repair::{
    repair_classified_frames, PreparedTraceClassification, TraceFrameClassifier,
};
pub use repair::{
    repair_inconsistent_frames, FrameFlagReason, FrameRepairAction, FrameRepairFrame,
    FrameRepairReport, FrameRepairRun, FrameRepairSummary,
};

const SEED_SEARCH_RADIUS: usize = 10;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct CleanupStats {
    pub frames_changed: usize,
    pub pixels_removed: usize,
    /// Frames whose clamped centre-line pixel was background (including recovered seeds).
    pub seed_misses: usize,
    /// Frames left unchanged because no seed could be found within the search radius.
    pub frames_without_seed: usize,
}

impl std::ops::AddAssign for CleanupStats {
    fn add_assign(&mut self, other: Self) {
        self.frames_changed += other.frames_changed;
        self.pixels_removed += other.pixels_removed;
        self.seed_misses += other.seed_misses;
        self.frames_without_seed += other.frames_without_seed;
    }
}

/// Keep the 8-connected foreground component at the interpolated centre line in
/// each 2D frame. A background seed uses the nearest foreground pixel within a
/// Euclidean radius of 10 pixels; ties use row-major order. If none exists, the
/// frame is unchanged. Coordinates are rounded and clamped to the frame edge.
pub fn keep_centerline_components(
    masks: &mut [FrameMask],
    annotation_points: &[AnnotationPoint],
) -> Result<CleanupStats, AppError> {
    let mut stats = CleanupStats::default();
    let mut selector = TraceComponentSelector::default();

    for (frame_index, mask) in masks.iter_mut().enumerate() {
        let selection = selector.select(mask, annotation_points, frame_index)?;
        stats += apply_trace_component(
            mask,
            selection.seed_status,
            selection.component.unwrap_or(&[]).iter().copied(),
        );
    }
    Ok(stats)
}

/// Apply a selected component with the centre-line cleanup's missing-seed policy.
/// Both newly selected and already classified components use this same path.
fn apply_trace_component(
    mask: &mut FrameMask,
    seed_status: TraceSeedStatus,
    component: impl IntoIterator<Item = bool>,
) -> CleanupStats {
    let mut stats = CleanupStats {
        seed_misses: usize::from(seed_status != TraceSeedStatus::Direct),
        ..CleanupStats::default()
    };
    if matches!(
        seed_status,
        TraceSeedStatus::MissingTrace | TraceSeedStatus::MissingForeground
    ) {
        stats.frames_without_seed = 1;
        return stats;
    }
    for (pixel, inside) in mask.pixels.iter_mut().zip(component) {
        if *pixel != 0 && !inside {
            *pixel = 0;
            stats.pixels_removed += 1;
        }
    }
    stats.frames_changed = usize::from(stats.pixels_removed > 0);
    stats
}

/// The single owner of trace interpolation, seed recovery, and component selection.
/// Its buffers are reused across frames; callers decide how to handle a missing seed.
#[derive(Default)]
struct TraceComponentSelector {
    reached: Vec<bool>,
    queue: Vec<usize>,
}

struct TraceComponentSelection<'a> {
    component: Option<&'a [bool]>,
    seed_status: TraceSeedStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TraceSeedStatus {
    MissingTrace,
    MissingForeground,
    Direct,
    Recovered,
}

fn validate_mask_geometry(mask: &FrameMask) -> Result<(), AppError> {
    // Selection indexes pixels before the TIFF writer can validate them.
    if mask.width == 0
        || mask.height == 0
        || mask.width.checked_mul(mask.height) != Some(mask.pixels.len())
    {
        return Err(AppError::bad_request(
            "Mask pixels do not match the declared frame geometry",
        ));
    }
    Ok(())
}

impl TraceComponentSelector {
    fn select(
        &mut self,
        mask: &FrameMask,
        annotation_points: &[AnnotationPoint],
        frame_index: usize,
    ) -> Result<TraceComponentSelection<'_>, AppError> {
        validate_mask_geometry(mask)?;
        let Some(point) = interpolated_point_for_frame(annotation_points, frame_index) else {
            return Ok(TraceComponentSelection {
                component: None,
                seed_status: TraceSeedStatus::MissingTrace,
            });
        };
        let x = point.x.round().clamp(0.0, (mask.width - 1) as f32) as usize;
        let y = point.y.round().clamp(0.0, (mask.height - 1) as f32) as usize;
        let mut seed = Some(y * mask.width + x);
        let seed_missed = mask.pixels[seed.unwrap()] == 0;
        if seed_missed {
            seed = None;
            let mut nearest_distance = SEED_SEARCH_RADIUS * SEED_SEARCH_RADIUS + 1;
            for yy in y.saturating_sub(SEED_SEARCH_RADIUS)
                ..=y.saturating_add(SEED_SEARCH_RADIUS).min(mask.height - 1)
            {
                for xx in x.saturating_sub(SEED_SEARCH_RADIUS)
                    ..=x.saturating_add(SEED_SEARCH_RADIUS).min(mask.width - 1)
                {
                    let distance = xx.abs_diff(x).pow(2) + yy.abs_diff(y).pow(2);
                    let index = yy * mask.width + xx;
                    if distance < nearest_distance && mask.pixels[index] != 0 {
                        nearest_distance = distance;
                        seed = Some(index);
                    }
                }
            }
        }
        let Some(seed) = seed else {
            return Ok(TraceComponentSelection {
                component: None,
                seed_status: TraceSeedStatus::MissingForeground,
            });
        };

        self.reached.clear();
        self.reached.resize(mask.pixels.len(), false);
        self.queue.clear();
        self.queue.push(seed);
        self.reached[seed] = true;
        let mut next = 0;
        while next < self.queue.len() {
            let index = self.queue[next];
            next += 1;
            let x = index % mask.width;
            let y = index / mask.width;
            for yy in y.saturating_sub(1)..=(y + 1).min(mask.height - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(mask.width - 1) {
                    let neighbor = yy * mask.width + xx;
                    if !self.reached[neighbor] && mask.pixels[neighbor] != 0 {
                        self.reached[neighbor] = true;
                        self.queue.push(neighbor);
                    }
                }
            }
        }
        Ok(TraceComponentSelection {
            component: Some(&self.reached),
            seed_status: if seed_missed {
                TraceSeedStatus::Recovered
            } else {
                TraceSeedStatus::Direct
            },
        })
    }
}

#[cfg(test)]
mod tests;
