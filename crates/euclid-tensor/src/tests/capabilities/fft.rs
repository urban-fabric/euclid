//! `TensorFft`: the discrete Fourier transform.

use super::super::dense::Dense;
use crate::{FftKind, TensorFft};
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

// --- TensorFft ----------------------------------------------------------

/// Interleave real values as `re, im` pairs with zero imaginary part.
fn as_complex(re: &[f64]) -> Dense {
    let mut data = Vec::with_capacity(re.len() * 2);
    for &x in re {
        data.push(x);
        data.push(0.0);
    }
    Dense::new(&[data.len()], data)
}

#[test]
fn fft_of_a_constant_signal_is_a_single_spike() {
    // DFT of [1,1,1,1] is [4,0,0,0].
    let x = as_complex(&[1.0, 1.0, 1.0, 1.0]);
    let spectrum = x.fft_forward(&[4]);
    assert_eq!(spectrum.dims(), &[8]);
    assert!(close(spectrum.values()[0], 4.0));
    assert!(spectrum.values()[2..].iter().all(|&v| close(v, 0.0)));
}

#[test]
fn fft_matches_a_hand_computed_spectrum() {
    // DFT of [1,0,0,0] is flat: every bin is 1.
    let spectrum = as_complex(&[1.0, 0.0, 0.0, 0.0]).fft_forward(&[4]);
    for k in 0..4 {
        assert!(close(spectrum.values()[2 * k], 1.0));
        assert!(close(spectrum.values()[2 * k + 1], 0.0));
    }
}

#[test]
fn ifft_inverts_fft() {
    let x = as_complex(&[1.0, 2.0, -3.0, 0.5, 7.0, -1.0, 0.0, 4.0]);
    let round_trip = x.fft_forward(&[8]).ifft(&[8]);
    assert!(all_close(&round_trip, x.values()));
}

#[test]
fn fft_is_linear() {
    let a = as_complex(&[1.0, 2.0, 3.0, 4.0]);
    let b = as_complex(&[0.5, -1.0, 2.0, 0.0]);
    let sum_then_fft = (a.clone() + b.clone()).fft_forward(&[4]);
    let fft_then_sum = a.fft_forward(&[4]) + b.fft_forward(&[4]);
    assert!(all_close(&sum_then_fft, fft_then_sum.values()));
}

#[test]
fn fft_rejects_a_multi_axis_request() {
    // Only single-axis transforms are implemented; two lengths must not fabricate a result.
    let x = as_complex(&[1.0, 2.0, 3.0, 4.0]);
    assert!(x.fft(FftKind::Forward, &[2, 2]).values().is_empty());
}
