use crate::inference::image::FrameMask;

/// Pixel-centre chessboard distance, positive inside the selected component.
/// An absent opposite class has infinite distance and cannot be an accepted anchor.
pub(super) fn signed_distances(mask: &FrameMask) -> Vec<f64> {
    let mut signed = chessboard_distances(mask, false);
    let outside = chessboard_distances(mask, true);
    for ((distance, outside), &pixel) in signed.iter_mut().zip(outside).zip(&mask.pixels) {
        if pixel == 0 {
            *distance = -outside;
        }
    }
    signed
}

fn chessboard_distances(mask: &FrameMask, foreground: bool) -> Vec<f64> {
    let mut distances = mask
        .pixels
        .iter()
        .map(|&pixel| {
            if (pixel != 0) == foreground {
                0.0
            } else {
                f64::INFINITY
            }
        })
        .collect::<Vec<_>>();
    // Unit weights on all eight neighbours give the exact Chebyshev metric.
    for direction in [-1isize, 1] {
        for position in 0..distances.len() {
            let index = if direction < 0 {
                position
            } else {
                distances.len() - 1 - position
            };
            let x = index % mask.width;
            let y = index / mask.width;
            for (dx, dy) in [
                (direction, 0),
                (-1, direction),
                (0, direction),
                (1, direction),
            ] {
                if let (Some(xx), Some(yy)) = (x.checked_add_signed(dx), y.checked_add_signed(dy)) {
                    if xx < mask.width && yy < mask.height {
                        distances[index] =
                            distances[index].min(distances[yy * mask.width + xx] + 1.0);
                    }
                }
            }
        }
    }
    distances
}

/// The fixed 21x21 Euclidean disc, represented by each row's horizontal radius.
pub(super) fn dilated_by_disc(component: &[bool], width: usize, height: usize) -> Vec<bool> {
    const ROW_RADII: [usize; 21] = [
        0, 4, 6, 7, 8, 8, 9, 9, 9, 9, 10, 9, 9, 9, 9, 8, 8, 7, 6, 4, 0,
    ];
    let mut dilated = vec![false; component.len()];
    for (index, &inside) in component.iter().enumerate() {
        if !inside {
            continue;
        }
        let x = index % width;
        let y = index / width;
        for (row, radius) in ROW_RADII.into_iter().enumerate() {
            if let Some(yy) = y
                .checked_add_signed(row as isize - 10)
                .filter(|&yy| yy < height)
            {
                let start = yy * width + x.saturating_sub(radius);
                let end = yy * width + (x + radius).min(width - 1);
                dilated[start..=end].fill(true);
            }
        }
    }
    dilated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chessboard_signed_distances_match_brute_force_for_small_rectangles() {
        for (width, height) in [(1, 5), (5, 1), (2, 3), (3, 2), (3, 3)] {
            for bits in 0..(1usize << (width * height)) {
                let mask = FrameMask {
                    width,
                    height,
                    pixels: (0..width * height)
                        .map(|index| if bits & (1 << index) != 0 { 255 } else { 0 })
                        .collect(),
                };
                for (index, distance) in signed_distances(&mask).into_iter().enumerate() {
                    let nearest = mask
                        .pixels
                        .iter()
                        .enumerate()
                        .filter(|(_, pixel)| **pixel != mask.pixels[index])
                        .map(|(other, _)| {
                            (index % width)
                                .abs_diff(other % width)
                                .max((index / width).abs_diff(other / width))
                                as f64
                        })
                        .fold(f64::INFINITY, f64::min);
                    let expected = if mask.pixels[index] != 0 {
                        nearest
                    } else {
                        -nearest
                    };
                    assert_eq!(
                        distance, expected,
                        "{width}x{height}, mask {bits}, pixel {index}"
                    );
                }
            }
        }
        let mut mask = FrameMask {
            width: 300,
            height: 1,
            pixels: vec![0; 300],
        };
        mask.pixels[0] = 255;
        assert_eq!(
            signed_distances(&mask)[299],
            -299.0,
            "distances do not saturate at255"
        );
    }

    #[test]
    fn ten_pixel_dilation_is_an_inclusive_euclidean_disc_at_centre_and_edges() {
        for (x, y) in [(12usize, 12usize), (0, 0), (24, 24), (0, 12), (12, 24)] {
            let mut component = vec![false; 25 * 25];
            component[y * 25 + x] = true;
            let dilated = dilated_by_disc(&component, 25, 25);
            for (index, inside) in dilated.into_iter().enumerate() {
                assert_eq!(
                    inside,
                    x.abs_diff(index % 25).pow(2) + y.abs_diff(index / 25).pow(2) <= 100
                );
            }
        }
        assert!(dilated_by_disc(&vec![false; 25 * 25], 25, 25)
            .iter()
            .all(|&inside| !inside));
    }
}
