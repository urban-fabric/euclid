//! `TensorIndexing`: ranges and categorical encoding.

use super::super::dense::Dense;
use crate::{Tensor, TensorIndexing, TensorSort};
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

// --- TensorIndexing -----------------------------------------------------

#[test]
fn iota_is_the_index_range() {
    assert!(all_close(&Dense::iota(4), &[0.0, 1.0, 2.0, 3.0]));
}

#[test]
fn one_hot_encodes_each_index_as_a_row() {
    let idx = Dense::vector(&[0.0, 2.0, 1.0]);
    let got = Dense::one_hot(&idx, 3);
    assert_eq!(got.dims(), &[3, 3]);
    assert!(all_close(
        &got,
        &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    ));
}

#[test]
fn one_hot_leaves_out_of_range_indices_empty() {
    let got = Dense::one_hot(&Dense::vector(&[5.0]), 3);
    assert!(all_close(&got, &[0.0, 0.0, 0.0]));
}

#[test]
fn one_hot_rows_sum_to_one_and_recover_the_index() {
    let idx = Dense::vector(&[0.0, 2.0, 1.0]);
    let encoded = Dense::one_hot(&idx, 3);
    // Summing the one-hot columns weighted by iota recovers the original index.
    assert_eq!(encoded.argmax(1), idx);
}

#[test]
fn bincount_histograms_the_indices() {
    let idx = Dense::vector(&[0.0, 1.0, 1.0, 3.0, 1.0]);
    assert!(all_close(
        &Dense::bincount(&idx, 5),
        &[1.0, 3.0, 0.0, 1.0, 0.0]
    ));
}

#[test]
fn bincount_totals_the_input_length() {
    let idx = Dense::vector(&[0.0, 1.0, 2.0, 2.0]);
    let counts = Dense::bincount(&idx, 3);
    assert!(close(counts.sum(&[]).item(), 4.0));
}
