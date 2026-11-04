//! The eight capability extensions, implemented for [`Dense`] with real semantics.
//!
//! Two conventions are worth stating, because the traits leave them to the implementation:
//!
//! - **Indices** are ordinary `Dense` tensors holding integral values, since this is a
//!   single-dtype carrier.
//! - **Complex values** (needed only by [`TensorFft`]) are carried *interleaved* along the last
//!   axis as `re, im, re, im, …`. A real-only element type cannot otherwise represent a
//!   spectrum, and this keeps `fft` shape-preserving so `ifft(fft(x)) ≈ x` is checkable.
//!
//! One module per capability trait, mirroring
//! [`super::capabilities`], which holds the tests for them.

use super::dense::Dense;
use euclid_numeric::Real;

/// Mean of a slice, `0.0` for an empty one.
pub(super) fn mean_of(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Stable `ln Σ exp` via the max-shift trick.
pub(super) fn logsumexp_of(xs: &[f64]) -> f64 {
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + xs.iter().map(|&x| (x - m).exp()).sum::<f64>().ln()
}

/// A general Einstein summation over one or two operands.
///
/// Handles repeated letters within an operand (`"ii->i"` takes a diagonal), letters dropped
/// from the output (summed over), and permutation (`"ij->ji"`).
pub(super) fn einsum(operands: &[&Dense], config: &str) -> Dense {
    let Some((inputs, output)) = config.split_once("->") else {
        return Dense::point(f64::NAN);
    };
    let input_specs: Vec<Vec<char>> = inputs
        .split(',')
        .map(|s| s.trim().chars().collect())
        .collect();
    let out_spec: Vec<char> = output.trim().chars().collect();

    if input_specs.len() != operands.len() {
        return Dense::point(f64::NAN);
    }

    // Letter -> extent, read off whichever operand axis mentions it.
    let mut letters: Vec<char> = Vec::new();
    let mut extents: Vec<usize> = Vec::new();
    for (spec, operand) in input_specs.iter().zip(operands) {
        for (axis, &letter) in spec.iter().enumerate() {
            let extent = operand.dims().get(axis).copied().unwrap_or(0);
            match letters.iter().position(|&l| l == letter) {
                Some(i) => {
                    if extents[i] != extent {
                        return Dense::point(f64::NAN);
                    }
                }
                None => {
                    letters.push(letter);
                    extents.push(extent);
                }
            }
        }
    }

    let out_shape: Vec<usize> = out_spec
        .iter()
        .map(|l| {
            letters
                .iter()
                .position(|x| x == l)
                .map(|i| extents[i])
                .unwrap_or(0)
        })
        .collect();
    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let total: usize = extents.iter().product::<usize>().max(1);

    let mut data = vec![0.0; out_len];
    // Walk every assignment of letters to index values, accumulating into the output slot.
    for flat in 0..total {
        let assignment = Dense::unravel(flat, &extents);
        let value: f64 = input_specs
            .iter()
            .zip(operands)
            .map(|(spec, operand)| {
                let coords: Vec<usize> = spec
                    .iter()
                    .map(|l| {
                        letters
                            .iter()
                            .position(|x| x == l)
                            .map(|i| assignment[i])
                            .unwrap_or(0)
                    })
                    .collect();
                operand
                    .values()
                    .get(Dense::ravel(&coords, operand.dims()))
                    .copied()
                    .unwrap_or(0.0)
            })
            .product();

        let out_coords: Vec<usize> = out_spec
            .iter()
            .map(|l| {
                letters
                    .iter()
                    .position(|x| x == l)
                    .map(|i| assignment[i])
                    .unwrap_or(0)
            })
            .collect();
        data[Dense::ravel(&out_coords, &out_shape)] += value;
    }

    Dense::new(&out_shape, data)
}

/// Order a lane's indices, breaking ties by position so the sort is deterministic.
pub(super) fn ordered_indices(lane: &[f64], descending: bool) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..lane.len()).collect();
    idx.sort_by(|&a, &b| {
        let ord = lane[a].total_cmp(&lane[b]);
        let ord = if descending { ord.reverse() } else { ord };
        ord.then(a.cmp(&b))
    });
    idx
}

/// Walk every window position: yields the output shape and, per output cell, the (possibly
/// padded) values the window covers.
pub(super) fn windows(
    input: &Dense,
    window_dimensions: &[usize],
    window_strides: &[usize],
    padding: &[(usize, usize)],
) -> (Vec<usize>, Vec<Vec<f64>>) {
    let shape = input.dims();
    let rank = shape.len();
    let dim = |v: &[usize], a: usize| v.get(a).copied().unwrap_or(1);
    let pad = |a: usize| padding.get(a).copied().unwrap_or((0, 0));

    let out_shape: Vec<usize> = (0..rank)
        .map(|a| {
            let (lo, hi) = pad(a);
            let padded = shape[a] + lo + hi;
            let w = dim(window_dimensions, a);
            if padded < w {
                0
            } else {
                (padded - w) / dim(window_strides, a) + 1
            }
        })
        .collect();

    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let window_len: usize = (0..rank).map(|a| dim(window_dimensions, a)).product();

    let mut cells = Vec::with_capacity(out_len);
    for slot in 0..out_len {
        let out_coords = Dense::unravel(slot, &out_shape);
        let mut covered = Vec::with_capacity(window_len);
        let window_shape: Vec<usize> = (0..rank).map(|a| dim(window_dimensions, a)).collect();

        for w in 0..window_len {
            let w_coords = Dense::unravel(w, &window_shape);
            let mut in_coords = vec![0usize; rank];
            let mut inside = true;
            for a in 0..rank {
                let (lo, _) = pad(a);
                let start = out_coords[a] * dim(window_strides, a);
                let padded_idx = start + w_coords[a];
                if padded_idx < lo || padded_idx - lo >= shape[a] {
                    inside = false;
                    break;
                }
                in_coords[a] = padded_idx - lo;
            }
            if inside {
                covered.push(input.values()[Dense::ravel(&in_coords, shape)]);
            }
        }
        cells.push(covered);
    }

    (out_shape, cells)
}

mod contract;
mod conv;
mod fft;
mod indexing;
mod reduce;
mod sort;
mod transform;
mod window;
