//! Integral edges: quadrature ties a density to its CDF, pinning the normalizer.

use super::support::{beta_pdf, gamma_pdf, normal_pdf};
use crate::Real;
use euclid_utils::{assert_close, integrate};

// --- integral edges ------------------------------------------------------

#[test]
fn the_normal_density_integrates_to_the_erf_cdf() {
    // integral_{-x}^{x} phi = erf(x/sqrt 2). Ties `erf` to `exp` and to `half_ln_tau` — the
    // density's normalizing constant is part of what is being checked.
    for i in 1..20 {
        let x = f64::from(i) * 0.2;
        let area = integrate(normal_pdf, -x, x, 2000);
        let want = Real::erf(x * <f64 as Real>::frac_1_sqrt_2());
        assert_close(area, want, 1e-10, &format!("int phi over [-{x},{x}] = erf"));
    }
    // And the total mass is 1, which pins the normalizer on its own.
    let total = integrate(normal_pdf, -40.0, 40.0, 20000);
    assert_close(total, 1.0, 1e-10, "normal density integrates to 1");
}

#[test]
fn the_gamma_density_integrates_to_the_incomplete_gamma() {
    // integral_0^x gamma pdf = P(a, x). An independent check on both `gamma_lr` and the
    // `ln_gamma` normalizer, by a route that shares no code with either.
    // Integer `a >= 2` only: the integrand goes like x^(a-1) near 0, so for non-integer `a`
    // some higher derivative is unbounded at the endpoint and Simpson's O(h^4) convergence
    // degrades. Small and fractional `a` are covered by the derivative edge above and the
    // `a = 1/2` closed form below, neither of which needs endpoint smoothness — one of the
    // reasons the web keeps several kinds of edge.
    for &a in &[2.0f64, 3.0, 7.0] {
        for i in 1..12 {
            let x = f64::from(i) * 0.75;
            let area = integrate(|t| gamma_pdf(t, a), 0.0, x, 4000);
            assert_close(
                area,
                Real::gamma_lr(x, a),
                1e-8,
                &format!("int gamma pdf to {x} = P({a}, {x})"),
            );
        }
    }
}

#[test]
fn the_beta_density_integrates_to_the_incomplete_beta() {
    // integral_0^x beta pdf = I_x(a,b). Ties `beta_reg` to `ln_beta` to `ln_gamma`.
    for &(a, b) in &[(2.0f64, 3.0f64), (4.0, 2.0), (2.5, 2.5)] {
        for i in 1..10 {
            let x = f64::from(i) * 0.1;
            let area = integrate(|t| beta_pdf(t, a, b), 0.0, x, 4000);
            assert_close(
                area,
                Real::beta_reg(x, a, b),
                1e-7,
                &format!("int beta pdf to {x} = I({a},{b})"),
            );
        }
    }
}

#[test]
fn the_gamma_integral_recovers_the_gamma_function() {
    // Gamma(a) = integral_0^inf x^{a-1} e^{-x} dx, the defining integral. Ties `gamma` and
    // `ln_gamma` to `exp`/`powf` with no incomplete-gamma machinery involved.
    // Integer `a >= 2`, for the endpoint-smoothness reason given above. Note the `a = 1` case
    // is specifically *not* usable here: the integrand is 1 at x = 0, not 0, and Simpson
    // weights that endpoint by h/3 — returning 0 from the guard biases the result by ~3e-4.
    for &a in &[2.0f64, 3.0, 5.0] {
        let f = |x: f64| {
            if x <= 0.0 {
                0.0
            } else {
                Real::powf(&x, a - 1.0) * (-x).exp()
            }
        };
        let area = integrate(f, 0.0, 60.0, 60000);
        assert_close(area, Real::gamma(a), 1e-7, &format!("int = Gamma({a})"));
        assert_close(
            area.ln(),
            Real::ln_gamma(a),
            1e-7,
            &format!("ln int = ln_gamma({a})"),
        );
    }
}
