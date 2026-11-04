//! `TensorWindow`: sliding-window reductions and pooling.

use super::super::dense::Dense;
use crate::TensorWindow;
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

// --- TensorWindow -------------------------------------------------------

#[test]
fn reduce_window_sum_slides_a_window() {
    let x = Dense::vector(&[1.0, 2.0, 3.0, 4.0]);
    // Width 2, stride 1, no padding -> 3 windows.
    let got = x.reduce_window_sum(&[2], &[1], &[(0, 0)]);
    assert_eq!(got.dims(), &[3]);
    assert!(all_close(&got, &[3.0, 5.0, 7.0]));
}

#[test]
fn reduce_window_respects_stride_and_padding() {
    let x = Dense::vector(&[1.0, 2.0, 3.0, 4.0]);
    // Stride 2 halves the output.
    assert!(all_close(
        &x.reduce_window_sum(&[2], &[2], &[(0, 0)]),
        &[3.0, 7.0]
    ));
    // One cell of padding on each side widens it again.
    let padded = x.reduce_window_sum(&[2], &[1], &[(1, 1)]);
    assert_eq!(padded.dims(), &[5]);
    assert!(all_close(&padded, &[1.0, 3.0, 5.0, 7.0, 4.0]));
}

#[test]
fn reduce_window_max_and_mean() {
    let x = Dense::vector(&[1.0, 5.0, 2.0, 8.0]);
    assert!(all_close(
        &x.reduce_window_max(&[2], &[1], &[(0, 0)]),
        &[5.0, 5.0, 8.0]
    ));
    assert!(all_close(
        &x.reduce_window_mean(&[2], &[1], &[(0, 0)]),
        &[3.0, 3.5, 5.0]
    ));
}

#[test]
fn max_pool2d_halves_an_nchw_image() {
    // 1x1x4x4 ramp, 2x2 pool, stride 2.
    let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let x = Dense::new(&[1, 1, 4, 4], data);
    let got = x.max_pool2d_nchw((2, 2), (2, 2), [(0, 0), (0, 0)]);
    assert_eq!(got.dims(), &[1, 1, 2, 2]);
    // Each 2x2 block's max is its bottom-right corner.
    assert!(all_close(&got, &[5.0, 7.0, 13.0, 15.0]));
}

#[test]
fn avg_pool2d_averages_each_block() {
    let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let x = Dense::new(&[1, 1, 4, 4], data);
    let got = x.avg_pool2d_nchw((2, 2), (2, 2), [(0, 0), (0, 0)]);
    assert!(all_close(&got, &[2.5, 4.5, 10.5, 12.5]));
}
