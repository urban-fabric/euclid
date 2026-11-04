//! Cancellation near zero, and the absorbing-zero conventions.

use crate::Real;
use euclid_utils::rel;

// --- cancellation near zero ---------------------------------------------

#[test]
fn ln_1p_keeps_digits_where_ln_one_plus_x_loses_them() {
    // For tiny x, `1.0 + x` rounds to exactly 1.0, so `ln(1+x)` returns 0 — total loss.
    let x = 1e-18f64;
    assert_eq!(
        (1.0 + x).ln(),
        0.0,
        "the naive form is expected to collapse"
    );
    // ln(1+x) ≈ x − x²/2, so the true value is x to full precision.
    assert!(rel(Real::ln_1p(&x), x) < 1e-15);

    // Through the moderate range it still beats the naive form.
    for i in 1..40 {
        let x = 10f64.powi(-i);
        let want = Real::ln_1p(&x);
        assert!(want.is_finite() && want > 0.0, "ln_1p({x}) = {want}");
    }
}

#[test]
fn exp_m1_keeps_digits_where_exp_minus_one_loses_them() {
    let x = 1e-18f64;
    assert_eq!(x.exp() - 1.0, 0.0, "the naive form is expected to collapse");
    assert!(rel(Real::exp_m1(&x), x) < 1e-15);
}

#[test]
fn ln_1m_exp_survives_both_ends_of_its_domain() {
    // ln(1 − e^x) for x < 0. Near x = 0⁻ the inner `1 − e^x` cancels; far from 0 it saturates.
    let near_zero = -1e-17f64;
    assert_eq!(
        (1.0 - near_zero.exp()).ln(),
        f64::NEG_INFINITY,
        "the naive form is expected to collapse to -inf"
    );
    let stable = Real::ln_1m_exp(&near_zero);
    assert!(stable.is_finite(), "ln_1m_exp({near_zero}) = {stable}");
    // 1 − e^x ≈ −x for tiny x, so ln(1 − e^x) ≈ ln(−x).
    assert!(rel(stable, (-near_zero).ln()) < 1e-12);

    // Deep in the tail, 1 − e^x → 1, so the result → 0⁻.
    let far = -800.0f64;
    assert!(Real::ln_1m_exp(&far).abs() < 1e-300);
}

// --- absorbing zeros ----------------------------------------------------

#[test]
fn xlogy_absorbs_zero_where_the_product_would_be_nan() {
    // 0 · ln 0 = 0 · (−∞) = NaN naively; the convention says 0.
    assert!((0.0f64 * 0.0f64.ln()).is_nan(), "naive product is NaN");
    assert_eq!(Real::xlogy(0.0, 0.0), 0.0);
    assert_eq!(Real::xlogy(0.0, 5.0), 0.0);
    // Non-zero x behaves like the ordinary product.
    assert!(rel(Real::xlogy(3.0, 2.0), 3.0 * 2.0f64.ln()) < 1e-15);
}

#[test]
fn xlog1py_absorbs_zero_at_the_logarithm_singularity() {
    // y = −1 puts ln(1+y) at −∞.
    assert!(
        (0.0f64 * (1.0 + -1.0f64).ln()).is_nan(),
        "naive product is NaN"
    );
    assert_eq!(Real::xlog1py(0.0, -1.0), 0.0);
    assert_eq!(Real::xlog1py(0.0, 7.0), 0.0);
    assert!(rel(Real::xlog1py(2.0, 0.5), 2.0 * 1.5f64.ln()) < 1e-15);
}

#[test]
fn entropy_sum_stays_finite_with_zero_probabilities() {
    // The motivating use: a distribution with exact zeros must not poison the sum.
    let p = [0.0f64, 0.25, 0.0, 0.5, 0.25];
    let entropy: f64 = p.iter().map(|&pi| -Real::xlogy(pi, pi)).sum();
    assert!(entropy.is_finite() && entropy > 0.0, "entropy = {entropy}");
    // Equals the entropy of the non-zero support alone.
    let want = -(0.25f64 * 0.25f64.ln() + 0.5 * 0.5f64.ln() + 0.25 * 0.25f64.ln());
    assert!(rel(entropy, want) < 1e-15);
}
