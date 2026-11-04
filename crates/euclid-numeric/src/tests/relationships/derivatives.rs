//! Derivative edges: numerical differentiation ties a function to its derivative.

use super::support::{beta_pdf, gamma_pdf};
use crate::Real;
use euclid_utils::{assert_close, derivative};

// --- derivative edges ----------------------------------------------------

#[test]
fn softplus_differentiates_to_sigmoid() {
    // d/dx ln(1 + e^x) = sigma(x). Ties `ln_1p_exp` to `sigmoid`, two separately-stabilized
    // implementations that must nevertheless be exact derivatives of each other.
    for i in -60..60 {
        let x = f64::from(i) * 0.25;
        let d = derivative(|t| Real::ln_1p_exp(&t), x);
        assert_close(
            d,
            Real::sigmoid(&x),
            1e-8,
            &format!("softplus'({x}) = sigmoid"),
        );
    }
}

#[test]
fn sigmoid_differentiates_to_its_own_product_form() {
    // sigma' = sigma(1 - sigma). Ties `sigmoid` to `ln_1m_sigmoid` via the product.
    for i in -40..40 {
        let x = f64::from(i) * 0.25;
        let s = Real::sigmoid(&x);
        let d = derivative(|t| Real::sigmoid(&t), x);
        assert_close(d, s * (1.0 - s), 1e-8, &format!("sigmoid'({x})"));
        // And the log form agrees: d/dx ln sigma(x) = 1 - sigma(x).
        let dlog = derivative(|t| Real::ln_sigmoid(&t), x);
        assert_close(dlog, 1.0 - s, 1e-8, &format!("ln_sigmoid'({x})"));
    }
}

#[test]
fn ln_gamma_differentiates_to_digamma_which_differentiates_to_trigamma() {
    // The polygamma ladder. Three functions, two edges — an error in any one of them shows up.
    for i in 2..60 {
        let x = f64::from(i) * 0.25;
        let d = derivative(Real::ln_gamma, x);
        assert_close(
            d,
            Real::digamma(x),
            1e-7,
            &format!("ln_gamma'({x}) = digamma"),
        );
        let dd = derivative(Real::digamma, x);
        assert_close(
            dd,
            Real::trigamma(x),
            1e-6,
            &format!("digamma'({x}) = trigamma"),
        );
    }
}

#[test]
fn erf_differentiates_to_the_gaussian_kernel() {
    // d/dx erf(x) = 2/sqrt(pi) e^{-x^2}. Ties `erf` to `exp` and `pi`.
    for i in -30..30 {
        let x = f64::from(i) * 0.1;
        let d = derivative(Real::erf, x);
        let want = 2.0 / <f64 as Real>::pi().sqrt() * (-x * x).exp();
        assert_close(d, want, 1e-8, &format!("erf'({x})"));
        // erfc is its complement, so its derivative is the negation.
        let dc = derivative(Real::erfc, x);
        assert_close(dc, -want, 1e-7, &format!("erfc'({x})"));
    }
}

#[test]
fn incomplete_gamma_differentiates_to_the_gamma_density() {
    // d/dx P(a, x) = x^{a-1} e^{-x} / Gamma(a). Ties `gamma_lr` to `ln_gamma` through the
    // normalizer: if either were wrong by a constant factor, this would fail.
    for &a in &[0.75f64, 1.0, 2.5, 6.0] {
        for i in 1..40 {
            let x = f64::from(i) * 0.25;
            let d = derivative(|t| Real::gamma_lr(t, a), x);
            assert_close(d, gamma_pdf(x, a), 1e-7, &format!("P'({a}, {x}) = pdf"));
        }
    }
}

#[test]
fn incomplete_beta_differentiates_to_the_beta_density() {
    // d/dx I_x(a,b) = x^{a-1}(1-x)^{b-1} / B(a,b). Ties `beta_reg` to `ln_beta`, and through
    // it to `ln_gamma`.
    for &(a, b) in &[(2.0f64, 3.0f64), (1.5, 1.5), (5.0, 2.0)] {
        for i in 1..20 {
            let x = f64::from(i) * 0.05;
            let d = derivative(|t| Real::beta_reg(t, a, b), x);
            assert_close(
                d,
                beta_pdf(x, a, b),
                1e-6,
                &format!("I'({x}; {a},{b}) = pdf"),
            );
        }
    }
}

#[test]
fn ln_bessel_i0_differentiates_to_the_bessel_ratio() {
    // d/dx ln I0(x) = I1(x)/I0(x) — the von Mises mean resultant length. Ties `ln_bessel_i0`
    // to `bessel_i1_over_i0`, which are computed by different routes.
    //
    // Both sides now reach machine precision, so this edge is checked ~1000x tighter than it
    // was when they shared the A&S polynomials' 1e-7 error.
    // The wide step this test used to need existed only to suppress the old 1e-7 function
    // error, which differentiation amplifies by 1/h. With `ln I0` now accurate to machine
    // precision the standard step is strictly better: its O(h^4) truncation is negligible,
    // where h = 0.1 left ~1e-6 of truncation behind.
    for i in 2..40 {
        let x = f64::from(i) * 0.5;
        let d = derivative(Real::ln_bessel_i0, x);
        assert_close(
            d,
            Real::bessel_i1_over_i0(x),
            1e-9,
            &format!("ln_bessel_i0'({x}) = I1/I0"),
        );
    }
}

#[test]
fn logit_differentiates_to_the_reciprocal_variance() {
    // d/dp logit(p) = 1/(p(1-p)).
    for i in 1..20 {
        let p = f64::from(i) * 0.05;
        let d = derivative(|t| Real::logit(&t), p);
        assert_close(d, 1.0 / (p * (1.0 - p)), 1e-7, &format!("logit'({p})"));
    }
}
