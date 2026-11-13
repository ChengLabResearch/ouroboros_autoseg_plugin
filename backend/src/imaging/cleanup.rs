use crate::{
    error::AppError,
    imaging::annotations::{interpolated_point_for_frame, AnnotationPoint},
    inference::image::FrameMask,
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

/// Keep the 8-connected foreground component at the interpolated centre line in
/// each 2D frame. A background seed uses the nearest foreground pixel within a
/// Euclidean radius of 10 pixels; ties use row-major order. If none exists, the
/// frame is unchanged. Coordinates are rounded and clamped to the frame edge.
pub fn keep_centerline_components(
    masks: &mut [FrameMask],
    annotation_points: &[AnnotationPoint],
) -> Result<CleanupStats, AppError> {
    let mut stats = CleanupStats::default();
    let mut reached = Vec::new();
    let mut queue = Vec::new();

    for (frame_index, mask) in masks.iter_mut().enumerate() {
        // Cleanup indexes pixels before the TIFF writer can validate them.
        if mask.width == 0
            || mask.height == 0
            || mask.width.checked_mul(mask.height) != Some(mask.pixels.len())
        {
            return Err(AppError::bad_request(
                "Mask pixels do not match the declared frame geometry",
            ));
        }
        let Some(point) = interpolated_point_for_frame(annotation_points, frame_index) else {
            stats.seed_misses += 1;
            stats.frames_without_seed += 1;
            continue;
        };
        let x = point.x.round().clamp(0.0, (mask.width - 1) as f32) as usize;
        let y = point.y.round().clamp(0.0, (mask.height - 1) as f32) as usize;
        let mut seed = Some(y * mask.width + x);
        if mask.pixels[seed.unwrap()] == 0 {
            stats.seed_misses += 1;
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
            stats.frames_without_seed += 1;
            continue;
        };

        reached.clear();
        reached.resize(mask.pixels.len(), false);
        queue.clear();
        queue.push(seed);
        reached[seed] = true;
        let mut next = 0;
        while next < queue.len() {
            let index = queue[next];
            next += 1;
            let x = index % mask.width;
            let y = index / mask.width;
            for yy in y.saturating_sub(1)..=(y + 1).min(mask.height - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(mask.width - 1) {
                    let neighbor = yy * mask.width + xx;
                    if !reached[neighbor] && mask.pixels[neighbor] != 0 {
                        reached[neighbor] = true;
                        queue.push(neighbor);
                    }
                }
            }
        }
        let mut removed = 0;
        for (pixel, reached) in mask.pixels.iter_mut().zip(&reached) {
            if *pixel != 0 && !reached {
                *pixel = 0;
                removed += 1;
            }
        }
        stats.frames_changed += usize::from(removed > 0);
        stats.pixels_removed += removed;
    }
    Ok(stats)
}

#[cfg(test)]
mod tests;
