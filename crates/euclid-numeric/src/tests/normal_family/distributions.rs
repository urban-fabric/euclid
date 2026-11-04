//! The normal CDF, its log, and the two inverses.
//!
//! Part of the [`normal_family`](super) suite.

#![allow(clippy::excessive_precision)]

// Reference digits are written out in full as provenance; see the note in `accuracy`.

use crate::Real;
use euclid_utils::{
    assert_close, assert_exact, assert_nondecreasing, assert_probability, log_sweep, rel, sweep,
};

// --- ndtr: reference values and shape -----------------------------------

#[test]
fn ndtr_matches_reference_values() {
    // Reference: Python's statistics.NormalDist().cdf, an independent implementation.
    let table = [
        (-3.0f64, 1.349_898_031_630_095e-3),
        (-1.959_963_984_540_054, 0.025),
        (-1.0, 0.158_655_253_931_457_05),
        (0.0, 0.5),
        (1.0, 0.841_344_746_068_542_9),
        (1.959_963_984_540_054, 0.975),
        (3.0, 0.998_650_101_968_369_9),
        (5.0, 0.999_999_713_348_428_2),
    ];
    for (x, want) in table {
        assert_close(Real::ndtr(x), want, 1e-12, &format!("ndtr({x})"));
    }
}

#[test]
fn ndtr_is_a_cdf() {
    let xs: Vec<f64> = sweep(-10.0, 10.0, 400).collect();
    let values: Vec<f64> = xs.iter().map(|&x| Real::ndtr(x)).collect();
    assert_probability(values.iter().copied(), "ndtr stays in [0,1]");
    assert_nondecreasing(values.iter().copied(), 0.0, "ndtr is monotone");
    assert_exact(Real::ndtr(f64::NEG_INFINITY), 0.0, "ndtr(-inf) = 0");
    assert_exact(Real::ndtr(f64::INFINITY), 1.0, "ndtr(+inf) = 1");
    assert!(Real::ndtr(f64::NAN).is_nan());
}

#[test]
fn ndtr_is_symmetric_about_a_half() {
    // Phi(-x) = 1 - Phi(x).
    for x in sweep(0.0, 8.0, 80) {
        assert_close(
            Real::ndtr(-x),
            1.0 - Real::ndtr(x),
            1e-12,
            "Phi(-x) = 1 - Phi(x)",
        );
    }
}

// --- log_ndtr: the reason it exists -------------------------------------

#[test]
fn log_ndtr_stays_finite_where_the_naive_composition_dies() {
    // Phi underflows to exactly zero around x = -38, so ln(ndtr(x)) is -inf from there on
    // while the true value is a perfectly ordinary number.
    assert_exact(Real::ndtr(-40.0f64), 0.0, "Phi underflows");
    assert!(Real::ndtr(-40.0f64).ln().is_infinite(), "so its log dies");

    for x in [-40.0f64, -100.0, -1e3, -1e6, -1e8] {
        let v = Real::log_ndtr(x);
        assert!(v.is_finite() && v < 0.0, "log_ndtr({x}) = {v}");
        // ln Phi(x) ~ -x^2/2 - ln|x| - ln sqrt(2 pi) in the far left tail.
        let asym = -0.5 * x * x - (-x).ln() - <f64 as Real>::half_ln_tau();
        assert!(
            rel(v, asym) < 1e-6,
            "log_ndtr({x}) = {v}, asymptotic {asym}"
        );
    }
}

#[test]
fn log_ndtr_agrees_with_the_direct_log_where_both_work() {
    for x in sweep(-30.0, 8.0, 200) {
        let direct = Real::ndtr(x).ln();
        if !direct.is_finite() {
            continue;
        }
        assert_close(Real::log_ndtr(x), direct, 1e-11, &format!("log_ndtr({x})"));
    }
}

#[test]
fn log_ndtr_is_monotone_and_crosses_its_branch_seamlessly() {
    let values: Vec<f64> = sweep(-50.0, 10.0, 600).map(Real::log_ndtr).collect();
    assert_nondecreasing(values.iter().copied(), 0.0, "log_ndtr is monotone");
    // The implementation switches formula at x = -1; a mismatched join shows up as a jump.
    let left = Real::log_ndtr(-1.0f64 - 1e-9);
    let right = Real::log_ndtr(-1.0f64 + 1e-9);
    assert!(
        (left - right).abs() < 1e-8,
        "discontinuity at the branch: {left} vs {right}"
    );
    assert!(Real::log_ndtr(0.0f64) < 0.0);
    assert_close(
        Real::log_ndtr(0.0f64),
        -<f64 as Real>::ln_2(),
        1e-14,
        "ln Phi(0) = -ln 2",
    );
}

#[test]
fn a_truncated_normal_log_likelihood_stays_finite() {
    // The motivating use. A censored observation contributes ln Phi((c - mu)/sigma); built
    // from ln(ndtr(..)) it becomes -inf for any observation far enough out, silently turning
    // the whole log-likelihood into -inf.
    let (mu, sigma) = (0.0f64, 1.0f64);
    let censoring_points = [-5.0f64, -20.0, -50.0, -200.0];
    let total: f64 = censoring_points
        .iter()
        .map(|&c| Real::log_ndtr((c - mu) / sigma))
        .sum();
    assert!(total.is_finite(), "log-likelihood = {total}");
    // The naive version is -inf as soon as any point is past about -38.
    let naive: f64 = censoring_points
        .iter()
        .map(|&c| Real::ndtr((c - mu) / sigma).ln())
        .sum();
    assert!(
        naive.is_infinite(),
        "the naive form is expected to collapse, got {naive}"
    );
}

// --- ndtri: the quantile ------------------------------------------------

#[test]
fn ndtri_matches_reference_values() {
    let table = [
        (0.025f64, -1.959_963_984_540_054),
        (0.05, -1.644_853_626_951_472_7),
        (0.1, -1.281_551_565_544_600_5),
        (0.5, 0.0),
        (0.9, 1.281_551_565_544_600_5),
        (0.975, 1.959_963_984_540_054),
        (0.995, 2.575_829_303_548_900_7),
    ];
    for (p, want) in table {
        assert_close(Real::ndtri(p), want, 1e-11, &format!("ndtri({p})"));
    }
}

#[test]
fn ndtri_round_trips_through_ndtr() {
    for p in sweep(0.001, 0.999, 500) {
        assert_close(
            Real::ndtr(Real::ndtri(p)),
            p,
            1e-11,
            &format!("ndtr(ndtri({p}))"),
        );
    }
}

#[test]
fn ndtri_round_trips_deep_in_the_tail_through_the_log_cdf() {
    // Below about p = 1e-16 the probability itself is the only usable handle, so the
    // round-trip has to be checked in the log domain.
    for p in log_sweep(1e-300, 1e-16, 60) {
        let x = Real::ndtri(p);
        assert!(x.is_finite() && x < 0.0, "ndtri({p}) = {x}");
        assert_close(
            Real::log_ndtr(x),
            p.ln(),
            1e-10,
            &format!("tail round trip at {p}"),
        );
    }
}

#[test]
fn ndtri_is_monotone_antisymmetric_and_handles_its_endpoints() {
    let values: Vec<f64> = sweep(1e-6, 1.0 - 1e-6, 500).map(Real::ndtri).collect();
    assert_nondecreasing(values.iter().copied(), 0.0, "ndtri is monotone");
    for p in sweep(0.01, 0.49, 40) {
        assert_close(
            Real::ndtri(1.0 - p),
            -Real::ndtri(p),
            1e-10,
            "ndtri is antisymmetric",
        );
    }
    assert_exact(Real::ndtri(0.5f64), 0.0, "ndtri(1/2) = 0");
    assert_exact(Real::ndtri(0.0f64), f64::NEG_INFINITY, "ndtri(0) = -inf");
    assert_exact(Real::ndtri(1.0f64), f64::INFINITY, "ndtri(1) = +inf");
    assert!(Real::ndtri(1.5f64).is_nan(), "out of domain");
    assert!(Real::ndtri(-0.5f64).is_nan(), "out of domain");
    assert!(Real::ndtri(f64::NAN).is_nan());
}

// --- erfc_inv -----------------------------------------------------------

#[test]
fn erfc_inv_inverts_erfc_across_its_domain() {
    for y in sweep(0.01, 1.99, 200) {
        assert_close(
            Real::erfc(Real::erfc_inv(y)),
            y,
            1e-11,
            &format!("erfc_inv({y})"),
        );
    }
    assert_exact(Real::erfc_inv(1.0f64), 0.0, "erfc_inv(1) = 0");
    assert_exact(Real::erfc_inv(0.0f64), f64::INFINITY, "erfc_inv(0) = +inf");
    assert_exact(
        Real::erfc_inv(2.0f64),
        f64::NEG_INFINITY,
        "erfc_inv(2) = -inf",
    );
    assert!(Real::erfc_inv(2.5f64).is_nan());
    assert!(Real::erfc_inv(-0.1f64).is_nan());
}

#[test]
fn erfc_inv_reaches_tails_that_erf_inv_cannot() {
    // `erf_inv(1 - y)` is the obvious composition and it is useless for small y: 1 - y rounds
    // to 1 and erf_inv returns +inf.
    let y = 1e-20f64;
    assert_exact(1.0 - y, 1.0, "1 - y rounds away");
    assert!(
        Real::erf_inv(1.0f64 - y).is_infinite(),
        "so erf_inv gives inf"
    );
    let x = Real::erfc_inv(y);
    assert!(x.is_finite() && x > 0.0, "erfc_inv({y}) = {x}");
    // Verify through the scaled form, since erfc(x) itself has underflowed.
    assert_close(
        Real::erfcx(x).ln() - x * x,
        y.ln(),
        1e-9,
        "erfc_inv tail check via erfcx",
    );
    for y in log_sweep(1e-280, 1e-20, 40) {
        let x = Real::erfc_inv(y);
        assert!(x.is_finite() && x > 0.0, "erfc_inv({y}) = {x}");
        assert_close(Real::erfcx(x).ln() - x * x, y.ln(), 1e-9, "erfc_inv tail");
    }
}
