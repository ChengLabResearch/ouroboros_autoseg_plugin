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
}
