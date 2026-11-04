//! `TensorReduce`: scans and statistical reductions.

use super::super::dense::Dense;
use crate::TensorReduce;
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

// --- TensorReduce -------------------------------------------------------

#[test]
fn cumsum_accumulates_along_the_axis() {
    let x = Dense::vector(&[1.0, 2.0, 3.0, 4.0]);
    assert!(all_close(&x.cumsum(0), &[1.0, 3.0, 6.0, 10.0]));
}

#[test]
fn cumsum_runs_per_lane_on_a_matrix() {
    let x = Dense::matrix(2, 3, &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
    // Along axis 1, each row accumulates independently.
    assert!(all_close(
        &x.clone().cumsum(1),
        &[1.0, 3.0, 6.0, 10.0, 30.0, 60.0]
    ));
    // Along axis 0, each column does.
    assert!(all_close(&x.cumsum(0), &[1.0, 2.0, 3.0, 11.0, 22.0, 33.0]));
}

#[test]
fn var_and_std_are_the_population_moments() {
    // mean 4, deviations -3,-1,1,3 -> var = (9+1+1+9)/4 = 5
    let x = Dense::vector(&[1.0, 3.0, 5.0, 7.0]);
    assert!(close(x.clone().var(&[]).item(), 5.0));
    assert!(close(x.std(&[]).item(), 5.0_f64.sqrt()));
}

#[test]
fn var_over_one_axis_removes_only_that_axis() {
    let x = Dense::matrix(2, 2, &[1.0, 3.0, 10.0, 14.0]);
    let v = x.var(&[1]);
    assert_eq!(v.dims(), &[2]);
    assert!(all_close(&v, &[1.0, 4.0]));
}

#[test]
fn logsumexp_is_stable_against_large_inputs() {
    let x = Dense::vector(&[1000.0, 1000.0]);
    // Naive exp() would overflow to inf here; the max-shift keeps it finite.
    let got = x.logsumexp(&[]).item();
    assert!(got.is_finite());
    assert!(close(got, 1000.0 + 2.0_f64.ln()));
}

#[test]
fn logsumexp_matches_the_direct_formula_for_small_inputs() {
    let xs = [0.5, -1.0, 2.0, 0.0];
    let want = xs.iter().map(|&x: &f64| x.exp()).sum::<f64>().ln();
    assert!(close(Dense::vector(&xs).logsumexp(&[]).item(), want));
}

#[test]
fn softmax_normalizes_each_lane_to_one() {
    let x = Dense::matrix(2, 3, &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
    let s = x.softmax(1);
    let rows: Vec<f64> = s.values().chunks(3).map(|r| r.iter().sum()).collect();
    assert!(rows.iter().all(|&r| close(r, 1.0)));
    // A uniform lane is exactly uniform.
    assert!(s.values()[3..].iter().all(|&v| close(v, 1.0 / 3.0)));
}

#[test]
fn log_softmax_is_the_log_of_softmax() {
    let x = Dense::vector(&[0.3, -1.2, 2.5]);
    let direct = x.clone().softmax(0).map_values(f64::ln);
    let stable = x.log_softmax(0);
    assert!(all_close(&stable, direct.values()));
}
