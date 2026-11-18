use crate::{
    imaging::{annotations::AnnotationOrigin, input_window::PixelBounds},
    inference::image::PointPromptKind,
};

pub enum OverlayMarker {
    Prompt(PointPromptKind),
    RepairedFrame,
    InputWindow(PixelBounds),
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
    let mut set_pixel = |x: usize, y: usize| {
        if x < width && y < height {
            if let Some(pixel) = mask_slice.get_mut(y * width + x) {
                *pixel = intensity;
            }
        }
    };
    let x = x.round() as isize;
    let y = y.round() as isize;
    let offsets: &[(isize, isize)] = match marker {
        OverlayMarker::InputWindow(bounds) => {
            let left = bounds.left.min(width);
            let right = bounds.right.min(width);
            let top = bounds.top.min(height);
            let bottom = bounds.bottom.min(height);
            if left >= right || top >= bottom {
                return;
            }
            for x in left..right {
                set_pixel(x, top);
                set_pixel(x, bottom - 1);
            }
            for y in top..bottom {
                set_pixel(left, y);
                set_pixel(right - 1, y);
            }
            return;
        }
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
        set_pixel(xx as usize, yy as usize);
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
