//! `TensorConv`: batched, grouped, dilated convolution.

use super::super::dense::Dense;
use crate::TensorConv;
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

// --- TensorConv ---------------------------------------------------------

#[test]
fn conv2d_with_an_identity_kernel_is_the_identity() {
    let x = Dense::new(&[1, 1, 3, 3], (1..=9).map(|i| i as f64).collect());
    let kernel = Dense::new(&[1, 1, 1, 1], vec![1.0]);
    let got = x.conv2d_nchw(&kernel, (1, 1), [(0, 0), (0, 0)], (1, 1), (1, 1), 1);
    assert_eq!(got.dims(), &[1, 1, 3, 3]);
    assert!(all_close(
        &got,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    ));
}

#[test]
fn conv2d_sums_a_box_kernel() {
    let x = Dense::new(&[1, 1, 3, 3], (1..=9).map(|i| i as f64).collect());
    let kernel = Dense::new(&[1, 1, 2, 2], vec![1.0; 4]);
    let got = x.conv2d_nchw(&kernel, (1, 1), [(0, 0), (0, 0)], (1, 1), (1, 1), 1);
    assert_eq!(got.dims(), &[1, 1, 2, 2]);
    // 2x2 sliding sums over [[1,2,3],[4,5,6],[7,8,9]]
    assert!(all_close(&got, &[12.0, 16.0, 24.0, 28.0]));
}

#[test]
fn conv2d_same_padding_preserves_the_spatial_shape() {
    let x = Dense::new(&[1, 1, 3, 3], (1..=9).map(|i| i as f64).collect());
    let kernel = Dense::new(&[1, 1, 3, 3], vec![1.0; 9]);
    let got = x.conv2d_nchw(&kernel, (1, 1), [(1, 1), (1, 1)], (1, 1), (1, 1), 1);
    assert_eq!(got.dims(), &[1, 1, 3, 3]);
    // Centre tap covers the whole 3x3, summing 1..9.
    assert!(close(got.values()[4], 45.0));
}

#[test]
fn conv2d_stride_subsamples_the_output() {
    let x = Dense::new(&[1, 1, 4, 4], (0..16).map(|i| i as f64).collect());
    let kernel = Dense::new(&[1, 1, 2, 2], vec![1.0; 4]);
    let got = x.conv2d_nchw(&kernel, (2, 2), [(0, 0), (0, 0)], (1, 1), (1, 1), 1);
    assert_eq!(got.dims(), &[1, 1, 2, 2]);
    assert!(all_close(&got, &[10.0, 18.0, 42.0, 50.0]));
}

#[test]
fn conv2d_kernel_dilation_widens_the_receptive_field() {
    let x = Dense::new(&[1, 1, 3, 3], (1..=9).map(|i| i as f64).collect());
    // A dilated 2x2 kernel of ones reaches the four corners of the 3x3 input.
    let kernel = Dense::new(&[1, 1, 2, 2], vec![1.0; 4]);
    let got = x.conv2d_nchw(&kernel, (1, 1), [(0, 0), (0, 0)], (1, 1), (2, 2), 1);
    assert_eq!(got.dims(), &[1, 1, 1, 1]);
    assert!(close(got.item(), 1.0 + 3.0 + 7.0 + 9.0));
}

#[test]
fn conv2d_feature_groups_keep_channels_separate() {
    // Two input channels, two groups, one output channel each: no cross-channel mixing.
    let x = Dense::new(&[1, 2, 1, 2], vec![1.0, 2.0, 10.0, 20.0]);
    let kernel = Dense::new(&[2, 1, 1, 1], vec![1.0, 100.0]);
    let got = x.conv2d_nchw(&kernel, (1, 1), [(0, 0), (0, 0)], (1, 1), (1, 1), 2);
    assert_eq!(got.dims(), &[1, 2, 1, 2]);
    assert!(all_close(&got, &[1.0, 2.0, 1000.0, 2000.0]));
}

#[test]
fn conv_rejects_an_unsupported_dimension_numbering() {
    // This reference only implements NCHW/OIHW; anything else must not silently compute.
    let x = Dense::new(&[1, 1, 2, 2], vec![1.0; 4]);
    let k = Dense::new(&[1, 1, 1, 1], vec![1.0]);
    let got = x.conv_general_dilated(
        &k,
        &[1, 1],
        &[(0, 0), (0, 0)],
        &[1, 1],
        &[1, 1],
        3, // batch axis is not 0
        1,
        &[2, 3],
        1,
        0,
        &[2, 3],
        1,
        1,
    );
    assert!(got.values().is_empty());
}
