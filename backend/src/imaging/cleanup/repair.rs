use serde::Serialize;

use crate::{error::AppError, imaging::annotations::AnnotationPoint, inference::image::FrameMask};

use super::{
    apply_trace_component,
    distance::{dilated_by_disc, signed_distances},
    validate_mask_geometry, CleanupStats, TraceComponentSelector, TraceSeedStatus,
};

const ANCHOR_RECOVERY_RADIUS: usize = 150;
const MAX_AREA_RATIO: f64 = 1.5;
const MIN_IOU: f64 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameFlagReason {
    MissingTrace,
    EmptyComponent,
    EdgeContact,
    AreaJump,
    LowIou,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameRepairAction {
    Kept,
    Clipped,
    Replaced,
}

impl FrameRepairAction {
    fn is_repaired(self) -> bool {
        matches!(self, Self::Clipped | Self::Replaced)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrameRepairFrame {
    /// Area of the selected raw component before recovery or repair. This one
    /// measurement supplies both predecessor comparisons and anchor recovery.
    pub component_area: usize,
    /// Initial detector reasons are retained even when anchor recovery accepts a frame.
    pub reasons: Vec<FrameFlagReason>,
    pub anchor_recovered: bool,
    pub repair: FrameRepairAction,
}

impl FrameRepairFrame {
    pub fn is_flagged(&self) -> bool {
        !self.reasons.is_empty() && !self.anchor_recovered
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FrameRepairReport {
    /// One authoritative classification and outcome per input frame, in stack order.
    #[serde(serialize_with = "crate::imaging::serialize_indexed_frames")]
    pub frames: Vec<FrameRepairFrame>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrameRepairRun {
    pub start_frame: usize,
    pub end_frame: usize,
    pub flagged_frames: usize,
    pub repaired_frames: usize,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize)]
pub struct FrameRepairSummary {
    pub total_frames: usize,
    pub initially_flagged_frames: usize,
    pub recovered_anchor_frames: usize,
    pub accepted_frames: usize,
    pub flagged_frames: usize,
    pub repaired_frames: usize,
    pub kept_frames: usize,
    pub clipped_frames: usize,
    pub replaced_frames: usize,
    pub unrepaired_frames: usize,
    pub runs: Vec<FrameRepairRun>,
}

impl FrameRepairReport {
    pub fn is_repaired(&self, frame_index: usize) -> bool {
        self.frames
            .get(frame_index)
            .is_some_and(|frame| frame.repair.is_repaired())
    }

    /// Derive serialization/logging aggregates from the per-frame records.
    pub fn summary(&self) -> FrameRepairSummary {
        let mut summary = FrameRepairSummary {
            total_frames: self.frames.len(),
            ..FrameRepairSummary::default()
        };
        for (frame_index, frame) in self.frames.iter().enumerate() {
            summary.initially_flagged_frames += usize::from(!frame.reasons.is_empty());
            summary.recovered_anchor_frames += usize::from(frame.anchor_recovered);
            let repaired = usize::from(frame.repair.is_repaired());
            summary.repaired_frames += repaired;
            match frame.repair {
                FrameRepairAction::Kept => summary.kept_frames += 1,
                FrameRepairAction::Clipped => summary.clipped_frames += 1,
                FrameRepairAction::Replaced => summary.replaced_frames += 1,
            }
            if !frame.is_flagged() {
                summary.accepted_frames += 1;
                continue;
            }
            summary.flagged_frames += 1;
            summary.unrepaired_frames += 1 - repaired;
            if let Some(run) = summary
                .runs
                .last_mut()
                .filter(|run| run.end_frame.checked_add(1) == Some(frame_index))
            {
                run.end_frame = frame_index;
                run.flagged_frames += 1;
                run.repaired_frames += repaired;
            } else {
                summary.runs.push(FrameRepairRun {
                    start_frame: frame_index,
                    end_frame: frame_index,
                    flagged_frames: 1,
                    repaired_frames: repaired,
                });
            }
        }
        summary
    }
}

/// Own trace-component selection and initial frame classification for both
/// online input windows and the offline repair pass.
#[derive(Default)]
pub(crate) struct TraceFrameClassifier {
    selector: TraceComponentSelector,
}

/// One owner of a raw trace component, its cleanup outcome, and its classification.
/// Component pixels and cleanup counts stay in memory; reports serialize the record.
#[derive(Debug, Serialize)]
pub(crate) struct PreparedTraceClassification {
    #[serde(skip)]
    pub(crate) component: FrameMask,
    #[serde(flatten)]
    pub(crate) record: FrameRepairFrame,
    #[serde(skip)]
    cleanup_stats: CleanupStats,
}

impl PreparedTraceClassification {
    /// Consume this selection without repeating seed recovery or classification.
    /// Missing seeds retain the raw mask, matching ordinary centre-line cleanup.
    pub(crate) fn keep_component(self, mask: &mut FrameMask) -> (FrameRepairFrame, CleanupStats) {
        if self.cleanup_stats.frames_without_seed == 0 {
            *mask = self.component;
        }
        (self.record, self.cleanup_stats)
    }
}

impl TraceFrameClassifier {
    /// Select the raw trace component and compare it with the preceding
    /// selected component, even when that predecessor was flagged. The caller
    /// supplies matching frame geometry; recovery and repair happen separately.
    pub(crate) fn prepare(
        &mut self,
        mut mask: FrameMask,
        annotation_points: &[AnnotationPoint],
        frame_index: usize,
        previous: Option<(&FrameMask, &FrameRepairFrame)>,
    ) -> Result<PreparedTraceClassification, AppError> {
        let selection = self
            .selector
            .select(&mask, annotation_points, frame_index)?;
        let seed_status = selection.seed_status;
        let mut reasons = Vec::new();
        if selection.seed_status == TraceSeedStatus::MissingTrace {
            reasons.push(FrameFlagReason::MissingTrace);
        }
        let cleanup_stats = apply_trace_component(
            &mut mask,
            seed_status,
            selection.component.unwrap_or(&[]).iter().copied(),
        );
        if selection.component.is_none() {
            mask.pixels.fill(0);
            reasons.push(FrameFlagReason::EmptyComponent);
        }
        let component_area = mask.pixels.iter().filter(|&&pixel| pixel != 0).count();
        if mask.pixels.iter().enumerate().any(|(index, &pixel)| {
            pixel != 0
                && (index % mask.width == 0
                    || index % mask.width == mask.width - 1
                    || index / mask.width == 0
                    || index / mask.width == mask.height - 1)
        }) {
            reasons.push(FrameFlagReason::EdgeContact);
        }
        if let Some((previous, previous_record)) = previous {
            let previous_area = previous_record.component_area;
            let minimum = previous_area.min(component_area);
            let maximum = previous_area.max(component_area);
            let area_ratio = if maximum == 0 {
                1.0
            } else if minimum == 0 {
                f64::INFINITY
            } else {
                maximum as f64 / minimum as f64
            };
            if area_ratio > MAX_AREA_RATIO {
                reasons.push(FrameFlagReason::AreaJump);
            }
            let intersection = previous
                .pixels
                .iter()
                .zip(&mask.pixels)
                .filter(|(left, right)| **left != 0 && **right != 0)
                .count();
            let union = previous_area + component_area - intersection;
            let iou = if union == 0 {
                0.0
            } else {
                intersection as f64 / union as f64
            };
            if iou < MIN_IOU {
                reasons.push(FrameFlagReason::LowIou);
            }
        }
        Ok(PreparedTraceClassification {
            component: mask,
            cleanup_stats,
            record: FrameRepairFrame {
                component_area,
                reasons,
                anchor_recovered: false,
                repair: FrameRepairAction::Kept,
            },
        })
    }
}

/// Select trace components and classify against the preceding raw component,
/// recover plausible anchors, then clip flagged frames to interpolated shape estimates.
pub fn repair_inconsistent_frames(
    masks: &mut [FrameMask],
    annotation_points: &[AnnotationPoint],
) -> Result<FrameRepairReport, AppError> {
    validate_repair_geometry(masks)?;
    let mut report = FrameRepairReport {
        frames: Vec::with_capacity(masks.len()),
    };
    let mut classifier = TraceFrameClassifier::default();
    for frame_index in 0..masks.len() {
        let (previous, current) = masks.split_at_mut(frame_index);
        let raw = FrameMask {
            width: current[0].width,
            height: current[0].height,
            pixels: std::mem::take(&mut current[0].pixels),
        };
        let classified = classifier.prepare(
            raw,
            annotation_points,
            frame_index,
            previous.last().zip(report.frames.last()),
        )?;
        current[0] = classified.component;
        report.frames.push(classified.record);
    }
    repair_classified_frames(masks, report)
}

fn validate_repair_geometry(masks: &[FrameMask]) -> Result<(), AppError> {
    for mask in masks.iter() {
        validate_mask_geometry(mask)?;
        if let Some(first) = masks.first() {
            if (mask.width, mask.height) != (first.width, first.height) {
                return Err(AppError::bad_request(
                    "Frame repair requires consistent mask dimensions across the stack",
                ));
            }
        }
    }
    Ok(())
}

/// Repair masks using their already measured classifications. Valid components
/// must already be applied; missing-seed masks are cleared here, after callers
/// have had the opportunity to preserve their unrepaired outputs.
pub(crate) fn repair_classified_frames(
    masks: &mut [FrameMask],
    mut report: FrameRepairReport,
) -> Result<FrameRepairReport, AppError> {
    validate_repair_geometry(masks)?;
    if masks.len() != report.frames.len() {
        return Err(AppError::bad_request(
            "Frame repair requires one classification per mask frame",
        ));
    }
    for (mask, frame) in masks.iter_mut().zip(&report.frames) {
        if frame.reasons.contains(&FrameFlagReason::EmptyComponent) {
            mask.pixels.fill(0);
        }
    }
    recover_anchors(&mut report);
    let mut next = 0;
    while next < masks.len() {
        if !report.frames[next].is_flagged() {
            next += 1;
            continue;
        }
        let start = next;
        while next < masks.len() && report.frames[next].is_flagged() {
            next += 1;
        }
        let previous = start.checked_sub(1);
        let following = (next < masks.len()).then_some(next);
        match (previous, following) {
            (Some(left), Some(right)) => {
                // Both fields stay in the native pixel coordinates; keep only this bracket.
                let left_field = signed_distances(&masks[left]);
                let right_field = signed_distances(&masks[right]);
                for frame_index in start..next {
                    if report.frames[frame_index]
                        .reasons
                        .contains(&FrameFlagReason::MissingTrace)
                    {
                        continue;
                    }
                    // Integer gap weights preserve exact zero ties without division.
                    let left_weight = (right - frame_index) as f64;
                    let right_weight = (frame_index - left) as f64;
                    let foreground = left_field
                        .iter()
                        .zip(&right_field)
                        .map(|(&left, &right)| left_weight * left + right_weight * right > 0.0)
                        .collect();
                    let estimate = Estimate::new(foreground, masks[left].width, masks[left].height);
                    report.frames[frame_index].repair = estimate.apply_to(&mut masks[frame_index]);
                }
            }
            (Some(anchor), None) | (None, Some(anchor)) => {
                let estimate = Estimate::new(
                    masks[anchor]
                        .pixels
                        .iter()
                        .map(|&pixel| pixel != 0)
                        .collect(),
                    masks[anchor].width,
                    masks[anchor].height,
                );
                for frame_index in start..next {
                    if report.frames[frame_index]
                        .reasons
                        .contains(&FrameFlagReason::MissingTrace)
                    {
                        continue;
                    }
                    report.frames[frame_index].repair = estimate.apply_to(&mut masks[frame_index]);
                }
            }
            (None, None) => {} // No accepted estimate exists; retain selected components and flags.
        }
    }
    Ok(report)
}

fn recover_anchors(report: &mut FrameRepairReport) {
    for index in 0..report.frames.len() {
        let frame = &report.frames[index];
        if frame.reasons.is_empty()
            || frame.component_area == 0
            || frame.reasons.contains(&FrameFlagReason::EdgeContact)
            || frame.reasons.contains(&FrameFlagReason::MissingTrace)
        {
            continue;
        }
        let first = index.saturating_sub(ANCHOR_RECOVERY_RADIUS);
        let end = index
            .saturating_add(ANCHOR_RECOVERY_RADIUS + 1)
            .min(report.frames.len());
        // Initial reasons never change. Recovered anchors therefore cannot alter
        // another candidate's median or make recovery depend on iteration order.
        let mut accepted = report.frames[first..end]
            .iter()
            .filter(|frame| frame.reasons.is_empty())
            .collect::<Vec<_>>();
        if accepted.is_empty() {
            continue;
        }
        accepted.sort_unstable_by_key(|frame| frame.component_area);
        let middle = accepted.len() / 2;
        let median = if accepted.len() % 2 == 0 {
            (accepted[middle - 1].component_area as f64 + accepted[middle].component_area as f64)
                / 2.0
        } else {
            accepted[middle].component_area as f64
        };
        // Integer-valued areas scaled in f64 preserve inclusive decimal cutoffs.
        let area = frame.component_area as f64 * 10.0;
        if area >= median * 7.0 && area <= median * 14.0 {
            report.frames[index].anchor_recovered = true;
        }
    }
}

/// Own the current expected foreground and its Euclidean clipping envelope.
/// One-sided runs reuse the same estimate; two-sided runs retain only one at a time.
struct Estimate {
    foreground: Vec<bool>,
    envelope: Vec<bool>,
    area: usize,
}

impl Estimate {
    fn new(foreground: Vec<bool>, width: usize, height: usize) -> Self {
        let area = foreground.iter().filter(|&&inside| inside).count();
        let envelope = dilated_by_disc(&foreground, width, height);
        Self {
            foreground,
            envelope,
            area,
        }
    }

    fn apply_to(&self, mask: &mut FrameMask) -> FrameRepairAction {
        let retained = mask
            .pixels
            .iter()
            .zip(&self.envelope)
            .filter(|(pixel, inside)| **pixel != 0 && **inside)
            .count();
        if (retained as f64) * 10.0 < (self.area as f64) * 3.0 {
            for (pixel, &inside) in mask.pixels.iter_mut().zip(&self.foreground) {
                *pixel = if inside { 255 } else { 0 };
            }
            return FrameRepairAction::Replaced;
        }
        let mut changed = false;
        for (pixel, &inside) in mask.pixels.iter_mut().zip(&self.envelope) {
            if *pixel != 0 && !inside {
                *pixel = 0;
                changed = true;
            }
        }
        if changed {
            FrameRepairAction::Clipped
        } else {
            FrameRepairAction::Kept
        }
    }
}

#[cfg(test)]
mod tests;
