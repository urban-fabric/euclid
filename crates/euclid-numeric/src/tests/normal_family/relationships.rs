//! Edges tying the normal family into the relationship web.
//!
//! Part of the [`normal_family`](super) suite.

#![allow(clippy::excessive_precision)]

// Reference digits are written out in full as provenance; see the note in `accuracy`.

use crate::Real;
use euclid_utils::{assert_close, derivative, integrate, sweep};

// --- relationship-web edges for the new functions -----------------------

#[test]
fn ndtr_differentiates_to_the_normal_density() {
    // Phi'(x) = phi(x). Ties ndtr to exp and half_ln_tau.
    for x in sweep(-5.0, 5.0, 60) {
        let want = (-0.5 * x * x - <f64 as Real>::half_ln_tau()).exp();
        assert_close(
            derivative(Real::ndtr, x),
            want,
            1e-8,
            &format!("ndtr'({x})"),
        );
    }
}

#[test]
fn log_ndtr_differentiates_to_the_inverse_mills_ratio() {
    // d/dx ln Phi(x) = phi(x)/Phi(x). Ties log_ndtr to both ndtr and the density.
    for x in sweep(-6.0, 4.0, 60) {
        let ln_pdf = -0.5 * x * x - <f64 as Real>::half_ln_tau();
        let want = (ln_pdf - Real::log_ndtr(x)).exp();
        // As with `ln_bessel_i0`, differentiation amplifies the underlying function's error
        // by 1/h: `log_ndtr` runs through `erfcx` in the left tail, and the step is ~5e-4, so
        // an erfcx error of 1e-9 becomes microscopically visible here.
        assert_close(
            derivative(Real::log_ndtr, x),
            want,
            1e-5,
            &format!("log_ndtr'({x})"),
        );
    }
}

#[test]
fn the_normal_density_integrates_to_ndtr() {
    // Quadrature of the density reproduces the CDF, pinning the normalizer independently.
    let pdf = |t: f64| (-0.5 * t * t - <f64 as Real>::half_ln_tau()).exp();
    for x in sweep(-4.0, 4.0, 20) {
        let area = integrate(pdf, -40.0, x, 20_000);
        assert_close(area, Real::ndtr(x), 1e-10, &format!("int phi to {x}"));
    }
}

#[test]
fn ndtr_agrees_with_the_erf_and_incomplete_gamma_routes() {
    // Three independent routes to the same quantity, closing a triangle with the existing
    // web: ndtr, the erf form, and the chi-squared-with-1-degree-of-freedom form.
    for x in sweep(0.05, 6.0, 80) {
        let via_erf = 0.5 * (1.0 + Real::erf(x * <f64 as Real>::frac_1_sqrt_2()));
        let via_gamma = 0.5 * (1.0 + Real::gamma_lr(x * x / 2.0, 0.5));
        assert_close(
            Real::ndtr(x),
            via_erf,
            1e-12,
            &format!("ndtr vs erf at {x}"),
        );
        assert_close(
            Real::ndtr(x),
            via_gamma,
            1e-11,
            &format!("ndtr vs gamma at {x}"),
        );
    }
}

#[test]
fn ndtri_and_erfc_inv_are_consistent() {
    // erfc(x) = 2 Phi(-x sqrt 2), so erfc_inv(y) = -ndtri(y/2)/sqrt 2.
    for y in sweep(0.01, 1.9, 60) {
        assert_close(
            Real::erfc_inv(y),
            -Real::ndtri(0.5 * y) * <f64 as Real>::frac_1_sqrt_2(),
            1e-12,
            "erfc_inv via ndtri",
        );
    }
    // And erf_inv agrees where it is usable.
    for p in sweep(0.02, 0.98, 60) {
        assert_close(
            Real::ndtri(p),
            Real::erf_inv(2.0 * p - 1.0) / <f64 as Real>::frac_1_sqrt_2(),
            1e-10,
            "ndtri via erf_inv",
        );
    }
}

#[test]
fn log_ndtr_keeps_its_relative_accuracy_in_the_upper_tail() {
    // The mirror of the left-tail problem, and the easier one to miss because nothing
    // overflows. As `x` grows, `Phi(x) -> 1` and `ln Phi(x) -> 0` from below, so the answer is
    // a small negative number — but `Phi(x)` rounds to exactly `1.0` past about `x = 8.3`, and
    // `ln(1.0)` is `0`. That is accurate to `8e-17` absolute and carries no relative
    // information at all, which is the sense a log-likelihood needs.
    //
    // `ln Phi(x) = ln1p(-Phi(-x))` recovers it, because the complement is the small tail.
    //
    // The check is against `-Phi(-x)`: for large `x`, `ln1p(-Q) -> -Q`, and the two must agree
    // to the accuracy of `ln1p` itself once `Q` is small. At `x = 1` they legitimately differ
    // by 9% (`Q` is 0.16 there, nowhere near the linear regime), so the sweep starts where the
    // asymptote actually holds.
    for &x in &[3.0f64, 5.0, 8.0, 8.25, 10.0, 20.0, 37.0] {
        let got = Real::log_ndtr(x);
        let complement = -Real::ndtr(-x);
        assert!(
            got < 0.0,
            "ln Phi({x}) must be strictly negative, got {got:e}"
        );
        assert_close(
            got,
            complement,
            1e-3,
            &format!("log_ndtr({x}) against -Phi(-{x})"),
        );
    }
    // The specific regression: a bare `ln(ndtr(x))` returns exactly zero from here up.
    assert_eq!(Real::ndtr(8.25f64), 1.0, "Phi(8.25) rounds to 1 in f64");
    assert!(
        Real::log_ndtr(8.25f64) < 0.0,
        "log_ndtr(8.25) must not collapse to 0 just because Phi did"
    );
    // And it keeps going as far as the exponent range allows, on both carriers.
    assert!(Real::log_ndtr(37.0f64) < 0.0);
    assert!(Real::log_ndtr(10.0f32) < 0.0);
    // The left tail is untouched by that branch.
    assert_close(
        Real::log_ndtr(-100.0f64),
        -5.005_524_208_694_205e3,
        1e-14,
        "the left tail must be unchanged",
    );
}
