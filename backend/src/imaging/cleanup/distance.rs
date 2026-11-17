/// Exact pixel-centre Euclidean signed distance, positive inside the component.
/// A missing opposite class has infinite distance; callers own whether such a
/// component is an eligible interpolation endpoint.
pub(super) fn signed_distances(component: &[bool], width: usize, height: usize) -> Vec<f64> {
    let mut signed = squared_distances(component, width, height, false);
    let outside = squared_distances(component, width, height, true);
    for ((distance, outside), inside) in signed.iter_mut().zip(outside).zip(component) {
        *distance = if *inside {
            distance.sqrt()
        } else {
            -outside.sqrt()
        };
    }
    signed
}

/// A signed field extended with actual background pixels. The padding covers
/// the active anchor pair's centroid displacement, including bilinear neighbours.
pub(super) struct PaddedSignedDistanceField {
    values: Vec<f64>,
    width: usize,
    height: usize,
    padding_x: usize,
    padding_y: usize,
}

impl PaddedSignedDistanceField {
    pub(super) fn new(
        component: &[bool],
        width: usize,
        height: usize,
        padding_x: usize,
        padding_y: usize,
    ) -> Self {
        let padded_width = width + 2 * padding_x;
        let padded_height = height + 2 * padding_y;
        let mut padded = vec![false; padded_width * padded_height];
        for (y, row) in component.chunks_exact(width).enumerate() {
            let start = (y + padding_y) * padded_width + padding_x;
            padded[start..start + width].copy_from_slice(row);
        }
        Self {
            values: signed_distances(&padded, padded_width, padded_height),
            width: padded_width,
            height: padded_height,
            padding_x,
            padding_y,
        }
    }

    /// Sample in the original image's floating-point pixel-centre coordinates.
    /// The caller's centroid-derived padding must cover the requested location.
    pub(super) fn sample(&self, x: f64, y: f64) -> f64 {
        let x = x + self.padding_x as f64;
        let y = y + self.padding_y as f64;
        debug_assert!((0.0..=(self.width - 1) as f64).contains(&x));
        debug_assert!((0.0..=(self.height - 1) as f64).contains(&y));
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = x - x0 as f64;
        let fy = y - y0 as f64;
        let top =
            (1.0 - fx) * self.values[y0 * self.width + x0] + fx * self.values[y0 * self.width + x1];
        let bottom =
            (1.0 - fx) * self.values[y1 * self.width + x0] + fx * self.values[y1 * self.width + x1];
        (1.0 - fy) * top + fy * bottom
    }
}

/// Euclidean disk dilation, using the same exact transform as signed distances.
pub(super) fn dilated_by_radius(
    component: &[bool],
    width: usize,
    height: usize,
    radius: usize,
) -> Vec<bool> {
    let limit = (radius as f64).powi(2);
    squared_distances(component, width, height, true)
        .into_iter()
        .map(|distance| distance <= limit)
        .collect()
}

fn squared_distances(
    component: &[bool],
    width: usize,
    height: usize,
    feature_value: bool,
) -> Vec<f64> {
    let mut distances = component
        .iter()
        .map(|&value| {
            if value == feature_value {
                0.0
            } else {
                f64::INFINITY
            }
        })
        .collect::<Vec<_>>();
    let length = width.max(height);
    let mut source = vec![0.0; length];
    let mut transformed = vec![0.0; length];
    let mut sites = Vec::with_capacity(length);
    let mut boundaries = Vec::with_capacity(length + 1);

    for row in distances.chunks_exact_mut(width) {
        source[..width].copy_from_slice(row);
        transform_line(&source[..width], row, &mut sites, &mut boundaries);
    }
    for x in 0..width {
        for y in 0..height {
            source[y] = distances[y * width + x];
        }
        transform_line(
            &source[..height],
            &mut transformed[..height],
            &mut sites,
            &mut boundaries,
        );
        for y in 0..height {
            distances[y * width + x] = transformed[y];
        }
    }
    distances
}

/// Felzenszwalb/Huttenlocher's lower envelope of parabolas (Algorithm 1):
/// https://cs.brown.edu/people/pfelzens/papers/dt-final.pdf
/// Infinite sites are omitted so rows without features cannot produce inf-inf NaN.
fn transform_line(
    source: &[f64],
    output: &mut [f64],
    sites: &mut Vec<usize>,
    boundaries: &mut Vec<f64>,
) {
    sites.clear();
    boundaries.clear();
    for (q, value) in source
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
    {
        let mut crossing = f64::NEG_INFINITY;
        while let Some(&previous) = sites.last() {
            let qf = q as f64;
            let pf = previous as f64;
            crossing = ((value + qf * qf) - (source[previous] + pf * pf)) / (2.0 * (qf - pf));
            if crossing > *boundaries.last().unwrap() {
                break;
            }
            sites.pop();
            boundaries.pop();
        }
        sites.push(q);
        boundaries.push(crossing);
    }
    if sites.is_empty() {
        output.fill(f64::INFINITY);
        return;
    }
    boundaries.push(f64::INFINITY);
    let mut envelope_index = 0;
    for (q, distance) in output.iter_mut().enumerate() {
        while boundaries[envelope_index + 1] < q as f64 {
            envelope_index += 1;
        }
        let offset = q as f64 - sites[envelope_index] as f64;
        *distance = offset * offset + source[sites[envelope_index]];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_signed_distances_match_brute_force_for_small_rectangles() {
        for (width, height) in [(1, 5), (5, 1), (2, 3), (3, 2), (3, 3)] {
            // Exhaust every binary mask, including featureless rows and columns.
            for bits in 0..(1usize << (width * height)) {
                let component = (0..width * height)
                    .map(|index| bits & (1 << index) != 0)
                    .collect::<Vec<_>>();
                let actual = signed_distances(&component, width, height);
                for (index, &distance) in actual.iter().enumerate() {
                    let x = index % width;
                    let y = index / width;
                    let squared = component
                        .iter()
                        .enumerate()
                        .filter(|(_, inside)| **inside != component[index])
                        .map(|(other, _)| {
                            (x.abs_diff(other % width).pow(2) + y.abs_diff(other / width).pow(2))
                                as f64
                        })
                        .fold(f64::INFINITY, f64::min);
                    let expected = if component[index] {
                        squared.sqrt()
                    } else {
                        -squared.sqrt()
                    };
                    assert_eq!(
                        distance, expected,
                        "{width}x{height}, mask {bits}, pixel {index}"
                    );
                }
            }
        }
    }

    #[test]
    fn padded_fields_sample_fractional_native_and_off_image_coordinates() {
        let mut component = vec![false; 5 * 3];
        component[5 + 2] = true;
        let field = PaddedSignedDistanceField::new(&component, 5, 3, 4, 2);
        assert_eq!(field.sample(2.0, 1.0), 1.0);
        assert_eq!(field.sample(1.5, 1.0), 0.0);
        assert_eq!(field.sample(3.5, 1.0), -1.5);
        assert_eq!(field.sample(-0.5, 1.0), -2.5);
        let expected = (-2.0f64.sqrt() - 5.0f64.sqrt() - 5.0f64.sqrt() - 8.0f64.sqrt()) / 4.0;
        // The independently expanded four-sample mean has a different addition order.
        assert!((field.sample(3.5, 2.5) - expected).abs() <= f64::EPSILON * expected.abs());
    }

    #[test]
    fn ten_pixel_dilation_is_an_inclusive_euclidean_disk() {
        let mut component = vec![false; 25 * 25];
        component[12 * 25 + 12] = true;
        let dilated = dilated_by_radius(&component, 25, 25, 10);
        assert!(dilated[12 * 25 + 22]);
        assert!(dilated[20 * 25 + 18]);
        assert!(!dilated[13 * 25 + 22]);
        assert!(
            !dilated[20 * 25 + 20],
            "diagonal Chebyshev neighbours are outside the disk"
        );
        assert!(dilated_by_radius(&vec![false; 25 * 25], 25, 25, 10)
            .iter()
            .all(|&inside| !inside));
    }
}
