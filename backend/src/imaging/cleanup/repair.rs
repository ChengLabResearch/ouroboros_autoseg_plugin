use serde::Serialize;

use crate::{
    domain::requests::FrameRepairOptions, error::AppError, imaging::annotations::AnnotationPoint,
    inference::image::FrameMask,
};

use super::{
    distance::signed_distances, validate_mask_geometry, TraceComponentSelector, TraceSeedStatus,
};

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
    Accepted,
    Interpolated,
    CopiedPrevious,
    CopiedNext,
    Unrepaired,
}

impl FrameRepairAction {
    fn is_repaired(self) -> bool {
        matches!(
            self,
            Self::Interpolated | Self::CopiedPrevious | Self::CopiedNext
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrameRepairFrame {
    pub frame_index: usize,
    pub reasons: Vec<FrameFlagReason>,
    pub repair: FrameRepairAction,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FrameRepairReport {
    pub options: FrameRepairOptions,
    /// One authoritative classification and outcome per input frame, in stack order.
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
    pub accepted_frames: usize,
    pub flagged_frames: usize,
    pub repaired_frames: usize,
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
        for frame in &self.frames {
            if frame.reasons.is_empty() {
                summary.accepted_frames += 1;
                continue;
            }
            summary.flagged_frames += 1;
            let repaired = usize::from(frame.repair.is_repaired());
            summary.repaired_frames += repaired;
            summary.unrepaired_frames += 1 - repaired;
            if let Some(run) = summary
                .runs
                .last_mut()
                .filter(|run| run.end_frame.checked_add(1) == Some(frame.frame_index))
            {
                run.end_frame = frame.frame_index;
                run.flagged_frames += 1;
                run.repaired_frames += repaired;
            } else {
                summary.runs.push(FrameRepairRun {
                    start_frame: frame.frame_index,
                    end_frame: frame.frame_index,
                    flagged_frames: 1,
                    repaired_frames: repaired,
                });
            }
        }
        summary
    }
}

/// Select trace components, classify each against the preceding raw selected
/// frame, then repair flagged runs from their nearest accepted endpoints.
pub fn repair_inconsistent_frames(
    masks: &mut [FrameMask],
    annotation_points: &[AnnotationPoint],
    options: &FrameRepairOptions,
) -> Result<FrameRepairReport, AppError> {
    options.validate()?;
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
    let mut report = FrameRepairReport {
        options: *options,
        frames: Vec::with_capacity(masks.len()),
    };
    let mut selector = TraceComponentSelector::default();
    for frame_index in 0..masks.len() {
        let (previous, current) = masks.split_at_mut(frame_index);
        let mask = &mut current[0];
        let selection = selector.select(mask, annotation_points, frame_index)?;
        let mut reasons = Vec::new();
        if selection.seed_status == TraceSeedStatus::MissingTrace {
            reasons.push(FrameFlagReason::MissingTrace);
        }
        if let Some(component) = selection.component {
            for (pixel, &inside) in mask.pixels.iter_mut().zip(component) {
                if !inside {
                    *pixel = 0;
                }
            }
        } else {
            mask.pixels.fill(0);
            reasons.push(FrameFlagReason::EmptyComponent);
        }

        if mask.pixels.iter().enumerate().any(|(index, &pixel)| {
            pixel != 0
                && (index % mask.width == 0
                    || index % mask.width == mask.width - 1
                    || index / mask.width == 0
                    || index / mask.width == mask.height - 1)
        }) {
            reasons.push(FrameFlagReason::EdgeContact);
        }
        if let Some(previous) = previous.last() {
            let mut previous_area = 0usize;
            let mut current_area = 0usize;
            let mut intersection = 0usize;
            for (&left, &right) in previous.pixels.iter().zip(&mask.pixels) {
                previous_area += usize::from(left != 0);
                current_area += usize::from(right != 0);
                intersection += usize::from(left != 0 && right != 0);
            }
            let minimum = previous_area.min(current_area);
            let maximum = previous_area.max(current_area);
            let area_ratio = if maximum == 0 {
                1.0
            } else if minimum == 0 {
                f64::INFINITY
            } else {
                maximum as f64 / minimum as f64
            };
            if area_ratio > report.options.max_area_ratio {
                reasons.push(FrameFlagReason::AreaJump);
            }
            let union = previous_area + current_area - intersection;
            let iou = if union == 0 {
                0.0
            } else {
                intersection as f64 / union as f64
            };
            if iou < report.options.min_iou {
                reasons.push(FrameFlagReason::LowIou);
            }
        }
        let repair = if reasons.is_empty() {
            FrameRepairAction::Accepted
        } else {
            FrameRepairAction::Unrepaired
        };
        report.frames.push(FrameRepairFrame {
            frame_index,
            reasons,
            repair,
        });
    }

    let mut next = 0;
    while next < masks.len() {
        if report.frames[next].reasons.is_empty() {
            next += 1;
            continue;
        }
        let start = next;
        while next < masks.len() && !report.frames[next].reasons.is_empty() {
            next += 1;
        }
        let previous = start.checked_sub(1);
        let following = (next < masks.len()).then_some(next);
        match (previous, following) {
            (Some(left), Some(right)) => {
                // Only the current bracket's two SDFs are retained, never a stack of SDFs.
                let left_distance = signed_distances(
                    &masks[left]
                        .pixels
                        .iter()
                        .map(|&pixel| pixel != 0)
                        .collect::<Vec<_>>(),
                    masks[left].width,
                    masks[left].height,
                );
                let right_distance = signed_distances(
                    &masks[right]
                        .pixels
                        .iter()
                        .map(|&pixel| pixel != 0)
                        .collect::<Vec<_>>(),
                    masks[right].width,
                    masks[right].height,
                );
                for frame_index in start..next {
                    if report.frames[frame_index]
                        .reasons
                        .contains(&FrameFlagReason::MissingTrace)
                    {
                        continue;
                    }
                    let fraction = (frame_index - left) as f64 / (right - left) as f64;
                    for ((pixel, &left), &right) in masks[frame_index]
                        .pixels
                        .iter_mut()
                        .zip(&left_distance)
                        .zip(&right_distance)
                    {
                        *pixel = if (1.0 - fraction) * left + fraction * right > 0.0 {
                            255
                        } else {
                            0
                        };
                    }
                    report.frames[frame_index].repair = FrameRepairAction::Interpolated;
                }
            }
            (Some(anchor), None) | (None, Some(anchor)) => {
                let action = if previous.is_some() {
                    FrameRepairAction::CopiedPrevious
                } else {
                    FrameRepairAction::CopiedNext
                };
                let (before, from_anchor) = masks.split_at_mut(anchor);
                let (source, after) = from_anchor
                    .split_first_mut()
                    .expect("accepted anchor exists");
                let targets = if previous.is_some() {
                    after
                } else {
                    &mut before[start..next]
                };
                for (offset, mask) in targets.iter_mut().enumerate() {
                    let frame = &mut report.frames[start + offset];
                    if frame.reasons.contains(&FrameFlagReason::MissingTrace) {
                        continue;
                    }
                    mask.pixels.copy_from_slice(&source.pixels);
                    frame.repair = action;
                }
            }
            (None, None) => {} // No accepted shape exists; retain selected components.
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests;
