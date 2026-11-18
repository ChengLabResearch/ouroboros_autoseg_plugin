use std::sync::{Arc, Mutex};

use serde::Serialize;

use crate::{
    domain::requests::InputWindowOptions,
    error::AppError,
    inference::{image::FrameMask, video::VideoFramePrompt},
};

use super::{
    annotations::{interpolated_point_for_frame, AnnotationPoint},
    cleanup::{PreparedTraceClassification, TraceFrameClassifier},
    tiff_io::ImageFrame,
};

pub type SharedInputWindow = Arc<Mutex<InputWindowController>>;

/// Native pixel bounds; right and bottom are exclusive. No coordinate rescaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PixelBounds {
    pub left: usize,
    pub top: usize,
    pub right: usize,
    pub bottom: usize,
}

impl PixelBounds {
    fn of_mask(mask: &FrameMask) -> Option<Self> {
        let mut bounds: Option<Self> = None;
        for (index, _) in mask.pixels.iter().enumerate().filter(|(_, p)| **p != 0) {
            let (x, y) = (index % mask.width, index / mask.width);
            match &mut bounds {
                Some(bounds) => {
                    bounds.left = bounds.left.min(x);
                    bounds.top = bounds.top.min(y);
                    bounds.right = bounds.right.max(x + 1);
                    bounds.bottom = bounds.bottom.max(y + 1);
                }
                None => {
                    bounds = Some(Self {
                        left: x,
                        top: y,
                        right: x + 1,
                        bottom: y + 1,
                    })
                }
            }
        }
        bounds
    }

    fn touches_border(self, width: usize, height: usize) -> bool {
        self.left == 0 || self.top == 0 || self.right == width || self.bottom == height
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowReset {
    Initial,
    Prompt,
    Flag,
}

#[derive(Debug, Serialize)]
pub struct InputWindowFrame {
    pub bounds: PixelBounds,
    pub background: Vec<u8>,
    pub reset: Option<WindowReset>,
    pub touches_border: bool,
    /// The shared detector's initial classification, before offline anchor recovery.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) raw_detection: Option<PreparedTraceClassification>,
}

#[derive(Debug, Serialize)]
pub struct InputWindowReport {
    #[serde(serialize_with = "super::serialize_indexed_frames")]
    pub frames: Vec<InputWindowFrame>,
}

#[derive(Debug, Serialize)]
pub struct InputWindowSummary {
    pub total_frames: usize,
    pub resets: usize,
    pub flag_resets: usize,
    pub prompt_resets: usize,
    pub border_frames: usize,
    pub border_fraction: f64,
}

impl InputWindowReport {
    pub fn summary(&self) -> InputWindowSummary {
        let flag_resets = self
            .frames
            .iter()
            .filter(|f| f.reset == Some(WindowReset::Flag))
            .count();
        let prompt_resets = self
            .frames
            .iter()
            .filter(|f| f.reset == Some(WindowReset::Prompt))
            .count();
        let border_frames = self.frames.iter().filter(|f| f.touches_border).count();
        InputWindowSummary {
            total_frames: self.frames.len(),
            resets: flag_resets + prompt_resets,
            flag_resets,
            prompt_resets,
            border_frames,
            border_fraction: if self.frames.is_empty() {
                0.0
            } else {
                border_frames as f64 / self.frames.len() as f64
            },
        }
    }
}

/// Owns immutable per-frame preprocessing decisions and the preceding raw mask.
/// The source and inference callback share this instance; historical loads replay
/// its recorded decision instead of consulting the latest mask.
pub struct InputWindowController {
    options: InputWindowOptions,
    annotations: Arc<[AnnotationPoint]>,
    prompts: Arc<[VideoFramePrompt]>,
    frames: Vec<InputWindowFrame>,
    previous_mask: Option<Arc<FrameMask>>,
    classifier: TraceFrameClassifier,
    automatic_size: Option<usize>,
    retain_components: bool,
}

impl InputWindowController {
    pub fn new(
        options: InputWindowOptions,
        annotations: Arc<[AnnotationPoint]>,
        prompts: Arc<[VideoFramePrompt]>,
        retain_components: bool,
    ) -> Result<Self, AppError> {
        options.validate()?;
        Ok(Self {
            options,
            annotations,
            prompts,
            frames: Vec::new(),
            previous_mask: None,
            classifier: TraceFrameClassifier::default(),
            automatic_size: None,
            retain_components,
        })
    }

    pub fn blank_frame(
        &mut self,
        frame_index: usize,
        frame: &mut ImageFrame,
    ) -> Result<(), AppError> {
        if frame.width == 0
            || frame.height == 0
            || frame.channels == 0
            || frame
                .width
                .checked_mul(frame.height)
                .and_then(|n| n.checked_mul(frame.channels))
                != Some(frame.pixels.len())
        {
            return Err(AppError::bad_request(
                "Input window pixels do not match frame geometry",
            ));
        }
        if frame_index == self.frames.len() {
            if frame_index > 0
                && self
                    .frames
                    .last()
                    .and_then(|f| f.raw_detection.as_ref())
                    .is_none()
            {
                return Err(AppError::internal(
                    "Input window needs the preceding raw mask before loading the next frame",
                ));
            }
            if let Some(previous) = &self.previous_mask {
                if (previous.width, previous.height) != (frame.width, frame.height) {
                    return Err(AppError::bad_request(
                        "Input window requires consistent frame dimensions",
                    ));
                }
            }
            let flagged = self
                .frames
                .last()
                .and_then(|f| f.raw_detection.as_ref())
                .is_some_and(|f| !f.record.reasons.is_empty());
            let prompted = self
                .prompts
                .binary_search_by_key(&frame_index, |p| p.frame_index)
                .is_ok();
            let reset = if frame_index == 0 {
                Some(WindowReset::Initial)
            } else if flagged {
                Some(WindowReset::Flag)
            } else if prompted {
                Some(WindowReset::Prompt)
            } else {
                None
            };
            let size = self.options.window_size.or(self.automatic_size);
            let bounds = if size.is_none() {
                // Automatic mode needs an unrestricted first observation. Empty
                // masks cannot establish a size, so keep the full frame until one can.
                PixelBounds {
                    left: 0,
                    top: 0,
                    right: frame.width,
                    bottom: frame.height,
                }
            } else if reset.is_some() {
                let point = interpolated_point_for_frame(&self.annotations, frame_index)
                    .ok_or_else(|| {
                        AppError::bad_request(format!(
                            "Input window needs a trace point for frame {frame_index}"
                        ))
                    })?;
                let size = size.expect("automatic size established or explicit size supplied");
                let x = point.x.round().clamp(0.0, (frame.width - 1) as f32) as usize;
                let y = point.y.round().clamp(0.0, (frame.height - 1) as f32) as usize;
                PixelBounds {
                    left: x.saturating_sub(size / 2),
                    top: y.saturating_sub(size / 2),
                    right: x.saturating_add(size - size / 2).min(frame.width),
                    bottom: y.saturating_add(size - size / 2).min(frame.height),
                }
            } else {
                let mask = self
                    .previous_mask
                    .as_ref()
                    .expect("previous frame observed");
                let bounds = PixelBounds::of_mask(mask).expect("empty component triggers a reset");
                let previous = self.frames.last().expect("previous window").bounds;
                PixelBounds {
                    left: bounds
                        .left
                        .saturating_sub(self.options.margin)
                        .max(previous.left.saturating_sub(self.options.max_growth)),
                    top: bounds
                        .top
                        .saturating_sub(self.options.margin)
                        .max(previous.top.saturating_sub(self.options.max_growth)),
                    right: bounds
                        .right
                        .saturating_add(self.options.margin)
                        .min(previous.right.saturating_add(self.options.max_growth))
                        .min(frame.width),
                    bottom: bounds
                        .bottom
                        .saturating_add(self.options.margin)
                        .min(previous.bottom.saturating_add(self.options.max_growth))
                        .min(frame.height),
                }
            };
            if bounds.left >= bounds.right || bounds.top >= bounds.bottom {
                return Err(AppError::bad_request(
                    "Previous mask lies outside the input window's permitted growth",
                ));
            }
            self.frames.push(InputWindowFrame {
                bounds,
                background: median_background(frame, self.previous_mask.as_deref()),
                reset,
                touches_border: bounds.touches_border(frame.width, frame.height),
                raw_detection: None,
            });
        }
        let decision = self.frames.get(frame_index).ok_or_else(|| {
            AppError::internal("Input window frames must first be loaded in forward order")
        })?;
        let bounds = decision.bounds;
        for (index, pixel) in frame.pixels.chunks_exact_mut(frame.channels).enumerate() {
            let (x, y) = (index % frame.width, index / frame.width);
            if x < bounds.left || x >= bounds.right || y < bounds.top || y >= bounds.bottom {
                pixel.copy_from_slice(&decision.background);
            }
        }
        Ok(())
    }

    pub fn observe(&mut self, frame_index: usize, mask: Arc<FrameMask>) -> Result<(), AppError> {
        if self.frames.len() != frame_index + 1 || self.frames[frame_index].raw_detection.is_some()
        {
            return Err(AppError::internal(
                "Input window raw masks must follow their input frames exactly once",
            ));
        }
        // The detector selects a component in its own working mask; the shared
        // raw output remains untouched for scoring, background, and following.
        let previous = frame_index
            .checked_sub(1)
            .and_then(|i| self.frames[i].raw_detection.as_ref())
            .map(|prepared| (&prepared.component, &prepared.record));
        let detection =
            self.classifier
                .prepare((*mask).clone(), &self.annotations, frame_index, previous)?;
        if self.automatic_size.is_none() || detection.record.reasons.is_empty() {
            if let Some(bounds) = PixelBounds::of_mask(&mask) {
                self.automatic_size = Some(
                    (bounds.right - bounds.left)
                        .max(bounds.bottom - bounds.top)
                        .saturating_mul(2)
                        .max(64),
                );
            }
        }
        if !self.retain_components && frame_index > 0 {
            // Only the preceding selection is needed for online detection.
            // A requested postpass instead consumes every retained selection.
            self.frames[frame_index - 1]
                .raw_detection
                .as_mut()
                .expect("preceding observation")
                .component
                .pixels = Vec::new();
        }
        self.frames[frame_index].raw_detection = Some(detection);
        self.previous_mask = Some(mask);
        Ok(())
    }

    pub fn finish(&mut self) -> Result<InputWindowReport, AppError> {
        if self
            .frames
            .iter()
            .any(|frame| frame.raw_detection.is_none())
        {
            return Err(AppError::internal(
                "Input window is missing a raw model output",
            ));
        }
        self.previous_mask = None;
        if !self.retain_components {
            if let Some(last) = self
                .frames
                .last_mut()
                .and_then(|f| f.raw_detection.as_mut())
            {
                last.component.pixels = Vec::new();
            }
        }
        Ok(InputWindowReport {
            frames: std::mem::take(&mut self.frames),
        })
    }
}

fn median_background(frame: &ImageFrame, previous: Option<&FrameMask>) -> Vec<u8> {
    let mut histograms = vec![[0usize; 256]; frame.channels];
    let mut count = 0;
    for (index, pixel) in frame.pixels.chunks_exact(frame.channels).enumerate() {
        if previous.is_some_and(|mask| mask.pixels[index] != 0) {
            continue;
        }
        for (histogram, &value) in histograms.iter_mut().zip(pixel) {
            histogram[value as usize] += 1;
        }
        count += 1;
    }
    // A full foreground mask leaves no background sample. Use the whole current
    // frame, as for frame zero, rather than inventing a black/white level.
    if count == 0 {
        return median_background(frame, None);
    }
    histograms
        .iter()
        .map(|histogram| {
            let mut cumulative = 0;
            let mut lower = None;
            for (value, &n) in histogram.iter().enumerate() {
                cumulative += n;
                if lower.is_none() && cumulative > (count - 1) / 2 {
                    lower = Some(value);
                }
                if cumulative > count / 2 {
                    return ((lower.unwrap() + value) / 2) as u8;
                }
            }
            unreachable!("histogram contains all background samples")
        })
        .collect()
}

#[cfg(test)]
mod tests;
