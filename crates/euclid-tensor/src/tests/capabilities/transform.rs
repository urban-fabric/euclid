//! `TensorTransform`: roll, tile, clip.

use super::super::dense::Dense;
use crate::TensorTransform;
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

// --- TensorTransform ----------------------------------------------------

#[test]
fn roll_shifts_circularly_in_both_directions() {
    let x = Dense::vector(&[1.0, 2.0, 3.0, 4.0]);
    assert!(all_close(&x.roll(1, 0), &[4.0, 1.0, 2.0, 3.0]));
    assert!(all_close(&x.roll(-1, 0), &[2.0, 3.0, 4.0, 1.0]));
}

#[test]
fn roll_by_the_full_length_is_the_identity() {
    let x = Dense::vector(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(x.roll(4, 0), x);
    assert_eq!(x.roll(-8, 0), x);
}

#[test]
fn tile_repeats_along_the_axis() {
    let x = Dense::vector(&[1.0, 2.0]);
    let got = x.tile(3, 0);
    assert_eq!(got.dims(), &[6]);
    assert!(all_close(&got, &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]));
}

#[test]
fn clip_bounds_every_element() {
    let x = Dense::vector(&[-5.0, 0.5, 9.0]);
    let got = x.clip(Dense::point(0.0), Dense::point(1.0));
    assert!(all_close(&got, &[0.0, 0.5, 1.0]));
}
