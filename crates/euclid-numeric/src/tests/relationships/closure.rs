//! Summary tests: the same quantity by several independent routes, and a check that the web has teeth.

use super::support::gamma_pdf;
use crate::Real;
use euclid_utils::{assert_close, derivative, integrate, rel};

// --- the web closes ------------------------------------------------------

#[test]
fn the_normal_cdf_agrees_through_four_independent_routes() {
    // The summary test. Phi(x) computed four ways that share almost no code:
    //   1. via erf
    //   2. via erfc (the stable-tail route)
    //   3. via erfcx (the scaled route, used for extreme tails)
    //   4. via gamma_lr at half shape (the incomplete-gamma route)
    // All four must agree wherever all four are representable. Any single wrong
    // implementation shows up as a disagreement rather than as a plausible number.
    for i in 1..60 {
        let x = f64::from(i) * 0.1;
        let z = x * <f64 as Real>::frac_1_sqrt_2();

        let via_erf = 0.5 * (1.0 + Real::erf(z));
        let via_erfc = 0.5 * Real::erfc(-z);
        let via_erfcx = 0.5 * Real::erfcx(-z) * (-z * z).exp();
        let via_gamma = 0.5 * (1.0 + Real::gamma_lr(x * x / 2.0, 0.5));

        assert_close(via_erfc, via_erf, 1e-12, &format!("Phi({x}): erfc vs erf"));
        assert_close(
            via_erfcx,
            via_erf,
            1e-10,
            &format!("Phi({x}): erfcx vs erf"),
        );
        assert_close(
            via_gamma,
            via_erf,
            1e-11,
            &format!("Phi({x}): gamma vs erf"),
        );
    }
}

#[test]
fn a_log_density_round_trips_through_its_cdf_and_quantile() {
    // End-to-end: build a Gamma log-density from `ln_gamma`, integrate it with quadrature to
    // get a probability, invert the CDF by bisection using `gamma_lr`, and check the quantile
    // reproduces the probability. Exercises ln_gamma, exp, ln, gamma_lr and the comparison
    // surface in one chain.
    let a = 3.0f64;
    for &p in &[0.1f64, 0.25, 0.5, 0.75, 0.9] {
        // Bisection on P(a, x) = p.
        let (mut lo, mut hi) = (0.0f64, 50.0f64);
        for _ in 0..200 {
            let mid = Real::min(Real::max((lo + hi) * 0.5, lo), hi);
            if Real::gamma_lr(mid, a) < p {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let quantile = (lo + hi) * 0.5;
        // The CDF of the quantile is the probability we asked for.
        assert_close(
            Real::gamma_lr(quantile, a),
            p,
            1e-9,
            &format!("quantile({p})"),
        );
        // And quadrature of the density up to that point agrees.
        let area = integrate(|t| gamma_pdf(t, a), 0.0, quantile, 4000);
        assert_close(area, p, 1e-8, &format!("int pdf to quantile({p}) = {p}"));
    }
}

#[test]
fn the_relationship_web_is_sensitive_to_a_perturbation() {
    // A consistency web is only worth having if it would actually catch an error, so this
    // checks the web has teeth: perturbing one function by a *single part in 10^6* — far
    // smaller than a real bug, and invisible to any eyeball check — must break the relations
    // that mention it.
    let perturbed_ln_gamma = |x: f64| Real::ln_gamma(x) * (1.0 + 1e-6);

    // The duplication formula notices.
    let x = 2.25f64;
    let lhs = perturbed_ln_gamma(2.0 * x);
    let rhs = (2.0 * x - 1.0) * <f64 as Real>::ln_2() - 0.5 * <f64 as Real>::ln_pi()
        + perturbed_ln_gamma(x)
        + perturbed_ln_gamma(x + 0.5);
    assert!(
        rel(lhs, rhs) > 1e-9,
        "duplication formula failed to notice a 1e-6 perturbation"
    );

    // So does the derivative edge against digamma.
    let d = derivative(perturbed_ln_gamma, 4.0);
    assert!(
        rel(d, Real::digamma(4.0)) > 1e-8,
        "ln_gamma' = digamma failed to notice a 1e-6 perturbation"
    );

    // And so does the gamma-density integral, through the normalizer.
    let perturbed_pdf = |t: f64, a: f64| {
        if t <= 0.0 {
            0.0
        } else {
            ((a - 1.0) * t.ln() - t - perturbed_ln_gamma(a)).exp()
        }
    };
    let area = integrate(|t| perturbed_pdf(t, 3.0), 0.0, 4.0, 4000);
    assert!(
        rel(area, Real::gamma_lr(4.0, 3.0)) > 1e-8,
        "pdf integral failed to notice a 1e-6 perturbation"
    );
}
