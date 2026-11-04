//! Functional equations: duplication and reflection formulas.

use crate::Real;
use euclid_utils::assert_close;

// --- functional equations -----------------------------------------------

#[test]
fn log_gamma_satisfies_the_duplication_formula() {
    // Gamma(2x) = 2^{2x-1} / sqrt(pi) * Gamma(x) Gamma(x + 1/2), i.e.
    // lnG(2x) = (2x-1) ln 2 - (1/2) ln pi + lnG(x) + lnG(x + 1/2).
    //
    // One relation constraining `ln_gamma` at three different arguments at once, plus `ln_2`
    // and `ln_pi`. A systematic error in the Lanczos coefficients would not survive this.
    for i in 1..80 {
        let x = f64::from(i) * 0.25;
        let lhs = Real::ln_gamma(2.0 * x);
        let rhs = (2.0 * x - 1.0) * <f64 as Real>::ln_2() - 0.5 * <f64 as Real>::ln_pi()
            + Real::ln_gamma(x)
            + Real::ln_gamma(x + 0.5);
        assert_close(lhs, rhs, 1e-12, &format!("duplication at {x}"));
    }
}

#[test]
fn digamma_satisfies_the_reflection_formula() {
    // psi(1-x) - psi(x) = pi cot(pi x). Ties `digamma` to `tan` and `pi`.
    for i in 1..20 {
        let x = f64::from(i) * 0.05; // (0, 1)
        let lhs = Real::digamma(1.0 - x) - Real::digamma(x);
        let rhs = <f64 as Real>::pi() / Real::tan(<f64 as Real>::pi() * x);
        assert_close(lhs, rhs, 1e-9, &format!("digamma reflection at {x}"));
    }
}

#[test]
fn digamma_satisfies_the_duplication_formula() {
    // psi(2x) = (1/2) psi(x) + (1/2) psi(x + 1/2) + ln 2.
    for i in 1..60 {
        let x = f64::from(i) * 0.25;
        let lhs = Real::digamma(2.0 * x);
        let rhs = 0.5 * Real::digamma(x) + 0.5 * Real::digamma(x + 0.5) + <f64 as Real>::ln_2();
        assert_close(lhs, rhs, 1e-9, &format!("digamma duplication at {x}"));
    }
}

#[test]
fn trigamma_satisfies_the_reflection_formula() {
    // psi'(x) + psi'(1-x) = pi^2 / sin^2(pi x).
    for i in 1..20 {
        let x = f64::from(i) * 0.05;
        let lhs = Real::trigamma(x) + Real::trigamma(1.0 - x);
        let s = Real::sin(<f64 as Real>::pi() * x);
        let rhs = <f64 as Real>::pi() * <f64 as Real>::pi() / (s * s);
        assert_close(lhs, rhs, 1e-9, &format!("trigamma reflection at {x}"));
    }
}
