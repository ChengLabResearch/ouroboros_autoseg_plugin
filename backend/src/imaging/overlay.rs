pub fn draw_annotation_star(
    mask_slice: &mut [u8],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    intensity: u8,
) {
    let x = x.round() as isize;
    let y = y.round() as isize;
    let offsets = [
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
    ];

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
