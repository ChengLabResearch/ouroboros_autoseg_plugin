use serde::Serialize;

use crate::{
    domain::requests::FrameRepairOptions, error::AppError, imaging::annotations::AnnotationPoint,
    inference::image::FrameMask,
};

use super::{
    distance::{dilated_by_radius, PaddedSignedDistanceField},
    validate_mask_geometry, TraceComponentSelector, TraceSeedStatus,
};

const ANCHOR_RECOVERY_RADIUS: usize = 150;
const ESTIMATE_DILATION_RADIUS: usize = 10;

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
    pub options: FrameRepairOptions,
    /// One authoritative classification and outcome per input frame, in stack order.
    #[serde(serialize_with = "serialize_frames")]
    pub frames: Vec<FrameRepairFrame>,
}

fn serialize_frames<S: serde::Serializer>(
    frames: &[FrameRepairFrame],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    // Frame identity is vector position. The explicit index exists only at output.
    #[derive(Serialize)]
    struct IndexedFrame<'a> {
        frame_index: usize,
        #[serde(flatten)]
        frame: &'a FrameRepairFrame,
    }
    serializer.collect_seq(
        frames
            .iter()
            .enumerate()
            .map(|(frame_index, frame)| IndexedFrame { frame_index, frame }),
    )
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

/// Select trace components and classify against the preceding raw component,
/// recover plausible anchors, then clip flagged frames to aligned shape estimates.
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
        if let Some(previous) = previous.last() {
            let previous_area = report
                .frames
                .last()
                .expect("preceding frame classified")
                .component_area;
            let minimum = previous_area.min(component_area);
            let maximum = previous_area.max(component_area);
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
            if iou < report.options.min_iou {
                reasons.push(FrameFlagReason::LowIou);
            }
        }
        report.frames.push(FrameRepairFrame {
            component_area,
            reasons,
            anchor_recovered: false,
            repair: FrameRepairAction::Kept,
        });
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
                // Retain only this bracket's fields. Native source components are
                // sampled after centroid translation; no rounded binary shift occurs.
                let anchors = AlignedAnchors::new(
                    &masks[left],
                    &report.frames[left],
                    &masks[right],
                    &report.frames[right],
                );
                for frame_index in start..next {
                    if report.frames[frame_index]
                        .reasons
                        .contains(&FrameFlagReason::MissingTrace)
                    {
                        continue;
                    }
                    let fraction = (frame_index - left) as f64 / (right - left) as f64;
                    let estimate =
                        Estimate::new(anchors.estimate(fraction), anchors.width, anchors.height);
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

#[derive(Clone, Copy)]
struct Centroid {
    x: f64,
    y: f64,
}

fn centroid(mask: &FrameMask, record: &FrameRepairFrame) -> Centroid {
    let mut x = 0.0;
    let mut y = 0.0;
    for (index, &pixel) in mask.pixels.iter().enumerate() {
        if pixel != 0 {
            x += (index % mask.width) as f64;
            y += (index / mask.width) as f64;
        }
    }
    Centroid {
        x: x / record.component_area as f64,
        y: y / record.component_area as f64,
    }
}

struct AlignedAnchors {
    left: PaddedSignedDistanceField,
    right: PaddedSignedDistanceField,
    left_centroid: Centroid,
    right_centroid: Centroid,
    width: usize,
    height: usize,
}

impl AlignedAnchors {
    fn new(
        left: &FrameMask,
        left_record: &FrameRepairFrame,
        right: &FrameMask,
        right_record: &FrameRepairFrame,
    ) -> Self {
        let left_centroid = centroid(left, left_record);
        let right_centroid = centroid(right, right_record);
        let padding_x = (left_centroid.x - right_centroid.x).abs().ceil() as usize + 1;
        let padding_y = (left_centroid.y - right_centroid.y).abs().ceil() as usize + 1;
        Self {
            left: PaddedSignedDistanceField::new(
                &left
                    .pixels
                    .iter()
                    .map(|&pixel| pixel != 0)
                    .collect::<Vec<_>>(),
                left.width,
                left.height,
                padding_x,
                padding_y,
            ),
            right: PaddedSignedDistanceField::new(
                &right
                    .pixels
                    .iter()
                    .map(|&pixel| pixel != 0)
                    .collect::<Vec<_>>(),
                right.width,
                right.height,
                padding_x,
                padding_y,
            ),
            left_centroid,
            right_centroid,
            width: left.width,
            height: left.height,
        }
    }

    fn estimate(&self, fraction: f64) -> Vec<bool> {
        let target = Centroid {
            x: (1.0 - fraction) * self.left_centroid.x + fraction * self.right_centroid.x,
            y: (1.0 - fraction) * self.left_centroid.y + fraction * self.right_centroid.y,
        };
        (0..self.width * self.height)
            .map(|index| {
                let x = (index % self.width) as f64;
                let y = (index / self.width) as f64;
                let left = self.left.sample(
                    x + self.left_centroid.x - target.x,
                    y + self.left_centroid.y - target.y,
                );
                let right = self.right.sample(
                    x + self.right_centroid.x - target.x,
                    y + self.right_centroid.y - target.y,
                );
                (1.0 - fraction) * left + fraction * right > 0.0
            })
            .collect()
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
        let envelope = dilated_by_radius(&foreground, width, height, ESTIMATE_DILATION_RADIUS);
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
