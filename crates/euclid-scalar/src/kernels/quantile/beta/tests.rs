//! Unit tests for the beta quantile.
//!
//! Beside [`super`] rather than with the shared solver. Shorter than its gamma sibling for the
//! same reason the implementation is: the support is bounded, so there is no seed and no bracket
//! search to get wrong — only the route table and the closed forms.

use super::*;
use crate::kernels::beta::beta_reg;

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(1.0),
        "{a:e} vs {b:e} (tol {tol:e})"
    );
}

#[test]
fn beta_quantile_round_trips_over_a_wide_parameter_range() {
    for &(a, b) in &[
        (0.5f64, 0.5f64),
        (1.0, 1.0),
        (2.0, 5.0),
        (5.0, 2.0),
        (0.1, 0.1),
        (100.0, 50.0),
    ] {
        // A U-shaped beta (`a` or `b` below 1) piles its mass into the corners, so the
        // density at the endpoints is enormous — around 1e10 for (0.1, 0.1). The root
        // itself then sits within ~1e-13 of 0 or 1, where one ulp of `x` is worth about
        // 1e-6 of CDF. That is a conditioning limit of `f64`, not of the solver: no
        // root-finder can resolve the probability more finely than the representable
        // spacing of `x` allows. Measured worst case 4.4e-7 at (0.1, 0.1), which matches
        // that estimate; the tolerance sits just above it.
        let tol = if a < 1.0 || b < 1.0 { 1e-5 } else { 1e-10 };
        for i in 1..40 {
            let p = f64::from(i) * 0.025;
            let x = beta_reg_inv(p, a, b);
            assert!((0.0..=1.0).contains(&x), "beta_reg_inv({p},{a},{b}) = {x}");
            close(beta_reg(a, b, x), p, tol);
        }
    }
}

#[test]
fn beta_quantile_matches_its_closed_forms() {
    // I_x(1,1) = x, so the quantile is p.
    for i in 1..50 {
        let p = f64::from(i) * 0.02;
        close(beta_reg_inv(p, 1.0, 1.0), p, 1e-12);
        // I_x(a,1) = x^a  =>  quantile is p^(1/a).
        close(beta_reg_inv(p, 3.0, 1.0), p.powf(1.0 / 3.0), 1e-10);
        // Arcsine law: I_x(1/2,1/2) = (2/pi) asin(sqrt x)  =>  x = sin^2(pi p / 2).
        let want = (core::f64::consts::PI * p / 2.0).sin().powi(2);
        close(beta_reg_inv(p, 0.5, 0.5), want, 1e-9);
    }
}

/// The beta quantile's four routes. Its upper endpoint is attained, unlike the gamma's.
#[test]
fn the_beta_quantile_routes_attain_both_endpoints() {
    use super::BetaQuantileRoute::{AtOne, AtZero, Solve, Undefined};

    for (p, a, b) in [
        (0.5, 0.0, 1.0),
        (0.5, 1.0, 0.0),
        (0.5, -1.0, 1.0),
        (0.5, 1.0, -1.0),
        (0.5, f64::NAN, 1.0),
        (0.5, 1.0, f64::NAN),
        (0.5, f64::INFINITY, 1.0),
        (0.5, 1.0, f64::INFINITY),
        (f64::NAN, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (2.0, 1.0, 1.0),
    ] {
        assert_eq!(
            super::BetaQuantileRoute::for_arguments(p, a, b),
            Undefined,
            "at p = {p:e}, a = {a:e}, b = {b:e}"
        );
    }
    assert_eq!(
        super::BetaQuantileRoute::for_arguments(0.0, 2.0, 3.0),
        AtZero
    );
    assert_eq!(
        super::BetaQuantileRoute::for_arguments(1.0, 2.0, 3.0),
        AtOne
    );
    // `I_x(a, b) = 1` at `x = 1` for every shape pair, which is the arm the gamma lacks.
    assert_eq!(super::beta_reg_inv(1.0, 0.1, 0.01), 1.0);
    for p in [f64::MIN_POSITIVE, 1e-12, 0.5, 1.0 - 1e-12] {
        assert_eq!(super::BetaQuantileRoute::for_arguments(p, 2.0, 3.0), Solve);
        // Including the U-shaped densities, where the mass piles into both corners.
        assert_eq!(super::BetaQuantileRoute::for_arguments(p, 0.1, 0.01), Solve);
    }
}
