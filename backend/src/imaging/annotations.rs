use crate::{
    domain::requests::NegativeBorderPoints,
    error::AppError,
    inference::{image::PointPrompt, video::VideoFramePrompt},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnotationOrigin {
    Metadata,
    Fallback,
    GapFill,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnnotationPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub origin: AnnotationOrigin,
}

impl AnnotationPoint {
    pub fn video_frame_index(&self, frame_count: usize) -> Option<usize> {
        let frame_index = self.z.round() as isize;
        (frame_index >= 0 && (frame_index as usize) < frame_count).then_some(frame_index as usize)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VolumeShape {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
}

pub fn default_annotations(shape: VolumeShape, interval: usize) -> Vec<AnnotationPoint> {
    if shape.frames == 0 {
        return Vec::new();
    }

    let interval = interval.max(1);
    let center_x = (shape.width / 2) as f32;
    let center_y = (shape.height / 2) as f32;
    let mut points = (0..shape.frames)
        .step_by(interval)
        .map(|frame| AnnotationPoint {
            x: center_x,
            y: center_y,
            z: frame as f32,
            origin: AnnotationOrigin::Fallback,
        })
        .collect::<Vec<_>>();

    points.push(AnnotationPoint {
        x: center_x,
        y: center_y,
        z: (shape.frames - 1) as f32,
        origin: AnnotationOrigin::Fallback,
    });
    points
}

pub fn annotation_samples_for_video(
    points: &[AnnotationPoint],
    frame_count: usize,
) -> Vec<VideoFramePrompt> {
    let mut grouped: std::collections::BTreeMap<usize, Vec<PointPrompt>> =
        std::collections::BTreeMap::new();

    for point in points {
        let Some(frame_index) = point.video_frame_index(frame_count) else {
            continue;
        };

        grouped
            .entry(frame_index)
            .or_default()
            .push(PointPrompt::positive(point.x, point.y, point.origin));
    }

    grouped
        .into_iter()
        .map(|(frame_index, points)| VideoFramePrompt {
            frame_index,
            points,
        })
        .collect()
}

/// Add only internal video prompt gaps. Endpoints and existing points are unchanged.
pub fn fill_prompt_gaps(
    points: &mut Vec<AnnotationPoint>,
    frame_count: usize,
    max_gap: Option<usize>,
) -> usize {
    let Some(max_gap) = max_gap.filter(|gap| *gap > 0) else {
        return 0;
    };
    let frames = points
        .iter()
        .filter_map(|point| point.video_frame_index(frame_count))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut added = Vec::new();
    for pair in frames.windows(2) {
        let gap = pair[1] - pair[0];
        let segments = gap.div_ceil(max_gap);
        for step in 1..segments {
            // Integer round-to-nearest avoids floating-point frame collisions.
            let offset =
                ((gap as u128 * step as u128 + segments as u128 / 2) / segments as u128) as usize;
            let frame_index = pair[0] + offset;
            if let Some(point) = interpolated_point_for_frame(points, frame_index) {
                added.push(AnnotationPoint {
                    x: point.x,
                    y: point.y,
                    z: frame_index as f32,
                    origin: AnnotationOrigin::GapFill,
                });
            }
        }
    }
    let count = added.len();
    points.extend(added);
    count
}

pub fn interpolated_point_for_frame(
    points: &[AnnotationPoint],
    frame_index: usize,
) -> Option<PointPrompt> {
    if points.is_empty() {
        return None;
    }

    let mut sorted = points.to_vec();
    sorted.sort_by(|left, right| {
        left.z
            .partial_cmp(&right.z)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if sorted.len() == 1 {
        return Some(PointPrompt::positive(
            sorted[0].x,
            sorted[0].y,
            sorted[0].origin,
        ));
    }

    let frame = frame_index as f32;
    let mut right = 0usize;
    while right < sorted.len() && sorted[right].z < frame {
        right += 1;
    }

    if right == 0 {
        return Some(PointPrompt::positive(
            sorted[0].x,
            sorted[0].y,
            sorted[0].origin,
        ));
    }
    if right >= sorted.len() {
        let point = sorted.last()?;
        return Some(PointPrompt::positive(point.x, point.y, point.origin));
    }

    let left = sorted[right - 1];
    let right_point = sorted[right];
    let dz = right_point.z - left.z;
    if dz.abs() < f32::EPSILON {
        return Some(PointPrompt::positive(
            right_point.x,
            right_point.y,
            right_point.origin,
        ));
    }

    let t = (frame - left.z) / dz;
    Some(PointPrompt::positive(
        left.x + t * (right_point.x - left.x),
        left.y + t * (right_point.y - left.y),
        left.origin,
    ))
}

/// Generate the image path's existing interpolated positive prompt on each frame.
pub fn annotation_samples_for_image(
    points: &[AnnotationPoint],
    frame_count: usize,
) -> Vec<VideoFramePrompt> {
    (0..frame_count)
        .filter_map(|frame_index| {
            interpolated_point_for_frame(points, frame_index).map(|point| VideoFramePrompt {
                frame_index,
                points: vec![point],
            })
        })
        .collect()
}

/// Append fixed negatives to the prepared prompt frames without consulting masks.
/// Distances are measured clockwise from the inset rectangle's top-left corner.
pub fn add_negative_border_points(
    prompts: &mut [VideoFramePrompt],
    shape: VolumeShape,
    options: &NegativeBorderPoints,
) -> Result<(), AppError> {
    let inset = options.inset_px;
    let horizontal = shape.width.saturating_sub(1) as f32 - 2.0 * inset;
    let vertical = shape.height.saturating_sub(1) as f32 - 2.0 * inset;
    if horizontal <= 0.0 || vertical <= 0.0 {
        return Err(AppError::bad_request(format!(
            "negative_border_points.inset_px ({inset}) leaves no border rectangle in a {}x{} frame",
            shape.width, shape.height
        )));
    }
    let perimeter = 2.0 * (horizontal + vertical);
    let negatives = (0..options.count)
        .map(|index| {
            let distance = perimeter * (index as f32 / options.count as f32);
            let (x, y) = if distance < horizontal {
                (inset + distance, inset)
            } else if distance < horizontal + vertical {
                (inset + horizontal, inset + distance - horizontal)
            } else if distance < 2.0 * horizontal + vertical {
                (
                    inset + 2.0 * horizontal + vertical - distance,
                    inset + vertical,
                )
            } else {
                (inset, inset + perimeter - distance)
            };
            PointPrompt::negative(x, y)
        })
        .collect::<Vec<_>>();
    for prompt in prompts {
        prompt.points.extend_from_slice(&negatives);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
