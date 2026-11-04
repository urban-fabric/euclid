//! Unit tests for the gamma family.
//!
//! A child module in its own file so implementation plus tests stay inside the crate's
//! 400-line budget, while `use super::*` still reaches the private Lanczos coefficients,
//! the Stirling remainder and the Edgeworth crossover.

use super::*;

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

#[test]
fn gamma_matches_factorials() {
    close(gamma(1.0), 1.0, 1e-12);
    close(gamma(5.0), 24.0, 1e-10);
    close(gamma(6.0), 120.0, 1e-10);
    close(gamma(0.5), core::f64::consts::PI.sqrt(), 1e-12);
}

#[test]
fn ln_gamma_matches_gamma() {
    for &x in &[0.3, 0.7, 1.5, 3.2, 10.0, 50.0] {
        close(ln_gamma(x).exp(), gamma(x), 1e-9);
    }
}

#[test]
fn regularized_incomplete_gamma_bounds_and_complement() {
    for &(a, x) in &[(1.0, 0.5), (2.5, 3.0), (5.0, 2.0), (10.0, 12.0)] {
        let p = gamma_lr(a, x);
        let q = gamma_ur(a, x);
        assert!((0.0..=1.0).contains(&p), "P out of range: {p}");
        close(p + q, 1.0, 1e-10);
    }
    close(gamma_lr(1.0, 2.0), 1.0 - (-2.0f64).exp(), 1e-10);
}

#[test]
fn digamma_known_values() {
    // Tightened to 1e-11 after adding the 1/(240·x⁸) asymptotic term: the prior
    // truncation reached only ~1e-10 (vs the ~15-digit intent of the coefficients).
    close(digamma(1.0), -crate::constants::f64::EULER_GAMMA, 1e-11);
    close(
        digamma(2.0),
        1.0 - crate::constants::f64::EULER_GAMMA,
        1e-11,
    );
    // ψ(3) = 3/2 − γ.
    close(
        digamma(3.0),
        1.5 - crate::constants::f64::EULER_GAMMA,
        1e-11,
    );
}

#[test]
fn trigamma_known_values() {
    let pi2 = core::f64::consts::PI * core::f64::consts::PI;
    close(trigamma(1.0), pi2 / 6.0, 1e-10);
    close(trigamma(2.0), pi2 / 6.0 - 1.0, 1e-10);
    close(trigamma(0.5), pi2 / 2.0, 1e-10);
    close(trigamma(3.3) - trigamma(4.3), 1.0 / (3.3 * 3.3), 1e-10);
}
