//! `TensorSort`: order statistics and the index-returning family.

use super::super::dense::Dense;
use crate::{Tensor, TensorSort};
use euclid_utils::is_close;

/// Hand-computed expected values in this suite are exact decimals, so the only error is
/// floating-point rounding through a few operations.
const TOL: f64 = 1e-9;

/// [`is_close`] at this suite's tolerance.
fn close(a: f64, b: f64) -> bool {
    is_close(a, b, TOL)
}

/// Elementwise [`close`] over a tensor's values against an expected slice.
fn all_close(got: &Dense, want: &[f64]) -> bool {
    got.values().len() == want.len() && got.values().iter().zip(want).all(|(&g, &w)| close(g, w))
}

// --- TensorSort ---------------------------------------------------------

#[test]
fn sort_orders_both_directions() {
    let x = Dense::vector(&[3.0, 1.0, 2.0]);
    assert!(all_close(&x.clone().sort(0, false), &[1.0, 2.0, 3.0]));
    assert!(all_close(&x.sort(0, true), &[3.0, 2.0, 1.0]));
}

#[test]
fn argsort_returns_the_ordering_permutation() {
    let x = Dense::vector(&[3.0, 1.0, 2.0]);
    assert!(all_close(&x.argsort(0, false), &[1.0, 2.0, 0.0]));
}

#[test]
fn argsort_composes_with_take_to_reproduce_sort() {
    // The documented relationship between the index and value families.
    let x = Dense::vector(&[5.0, -2.0, 9.0, 0.0]);
    let order = x.argsort(0, false);
    assert_eq!(x.take(&order, 0), x.clone().sort(0, false));
}

#[test]
fn argmax_and_argmin_find_the_extremes() {
    let x = Dense::vector(&[2.0, 9.0, 4.0, -1.0]);
    assert!(close(x.argmax(0).item(), 1.0));
    assert!(close(x.argmin(0).item(), 3.0));
}

#[test]
fn argmax_breaks_ties_toward_the_first_index() {
    let x = Dense::vector(&[1.0, 5.0, 5.0]);
    assert!(close(x.argmax(0).item(), 1.0));
}

#[test]
fn argmax_runs_per_row_on_a_matrix() {
    let x = Dense::matrix(2, 3, &[1.0, 7.0, 2.0, 9.0, 0.0, 3.0]);
    let got = x.argmax(1);
    assert_eq!(got.dims(), &[2]);
    assert!(all_close(&got, &[1.0, 0.0]));
}

#[test]
fn top_k_returns_values_and_their_indices() {
    let x = Dense::vector(&[4.0, 1.0, 9.0, 7.0]);
    let (values, indices) = x.top_k(2, true);
    assert!(all_close(&values, &[9.0, 7.0]));
    assert!(all_close(&indices, &[2.0, 3.0]));

    let (smallest, idx) = x.top_k(2, false);
    assert!(all_close(&smallest, &[1.0, 4.0]));
    assert!(all_close(&idx, &[1.0, 0.0]));
}

#[test]
fn top_k_clamps_k_to_the_axis_length() {
    let x = Dense::vector(&[1.0, 2.0]);
    let (values, _) = x.top_k(5, true);
    assert_eq!(values.dims(), &[2]);
}

#[test]
fn top_k_works_per_row_on_the_last_axis() {
    let x = Dense::matrix(2, 3, &[1.0, 5.0, 3.0, 9.0, 2.0, 4.0]);
    let (values, indices) = x.top_k(2, true);
    assert_eq!(values.dims(), &[2, 2]);
    assert!(all_close(&values, &[5.0, 3.0, 9.0, 4.0]));
    assert!(all_close(&indices, &[1.0, 2.0, 0.0, 2.0]));
}
