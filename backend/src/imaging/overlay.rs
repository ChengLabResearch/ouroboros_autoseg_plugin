use crate::{imaging::annotations::AnnotationOrigin, inference::image::PointPromptKind};

pub enum OverlayMarker {
    Prompt(PointPromptKind),
    RepairedFrame,
}

pub fn draw_overlay_marker(
    mask_slice: &mut [u8],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    intensity: u8,
    marker: OverlayMarker,
) {
    let x = x.round() as isize;
    let y = y.round() as isize;
    let offsets: &[(isize, isize)] = match marker {
        OverlayMarker::RepairedFrame => &[
            (0, 0),
            (-1, -1),
            (1, 1),
            (-1, 1),
            (1, -1),
            (-2, -2),
            (2, 2),
            (-2, 2),
            (2, -2),
        ],
        OverlayMarker::Prompt(PointPromptKind::Negative) => {
            &[(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        }
        OverlayMarker::Prompt(PointPromptKind::Positive(AnnotationOrigin::GapFill)) => &[
            (0, -3),
            (1, -2),
            (2, -1),
            (3, 0),
            (2, 1),
            (1, 2),
            (0, 3),
            (-1, 2),
            (-2, 1),
            (-3, 0),
            (-2, -1),
            (-1, -2),
        ],
        OverlayMarker::Prompt(PointPromptKind::Positive(_)) => &[
            (0, 0),
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
            (2, 0),
            (-2, 0),
            (0, 2),
            (0, -2),
        ],
    };

    for (dx, dy) in offsets {
        let xx = x + dx;
        let yy = y + dy;
        if xx < 0 || yy < 0 {
            continue;
        }
        let xx = xx as usize;
        let yy = yy as usize;
        if xx >= width || yy >= height {
            continue;
        }
        let index = yy * width + xx;
        if let Some(pixel) = mask_slice.get_mut(index) {
            *pixel = intensity;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{draw_overlay_marker, OverlayMarker};

    #[test]
    fn repaired_frame_marker_uses_existing_image_edge_clipping() {
        let mut pixels = vec![0; 4];
        draw_overlay_marker(
            &mut pixels,
            2,
            2,
            2.0,
            2.0,
            192,
            OverlayMarker::RepairedFrame,
        );
        assert_eq!(pixels, [192, 0, 0, 192]);
    }
}
