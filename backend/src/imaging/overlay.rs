use crate::imaging::annotations::AnnotationOrigin;

pub fn draw_annotation_marker(
    mask_slice: &mut [u8],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    intensity: u8,
    origin: AnnotationOrigin,
) {
    let x = x.round() as isize;
    let y = y.round() as isize;
    let offsets: &[(isize, isize)] = if origin == AnnotationOrigin::GapFill {
        &[
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
        ]
    } else {
        &[
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
        ]
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
