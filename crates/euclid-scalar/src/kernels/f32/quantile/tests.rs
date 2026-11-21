//! Unit tests for the single-precision gamma and beta quantiles.
//!
//! A child module so implementation plus tests stay inside the crate 400-line budget, while
//! `use super::*` still reaches `GammaQuantileRoute`, `GammaTail`, `BetaQuantileRoute` and the
//! iteration caps. The `f64` kernel is used as the reference throughout — it is four orders
//! more accurate than `f32` can represent — which is also how the small-shape conditioning
//! limit in `gamma_lr_inv` own documentation is measured rather than asserted away.

use super::*;
use crate::kernels::f32::beta::beta_reg;
use crate::kernels::f32::gamma::gamma_lr;

fn close(a: f32, b: f32, tol: f32) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(1.0),
        "{a:e} vs {b:e} (tol {tol:e})"
    );
}

#[test]
fn gamma_quantile_round_trips_in_f32() {
    for &a in &[0.5f32, 1.0, 2.5, 10.0] {
        for i in 1..20 {
            let p = (i as f32) * 0.05;
            let x = gamma_lr_inv(p, a);
            assert!(x.is_finite() && x >= 0.0, "gamma_lr_inv({p},{a}) = {x}");
            close(gamma_lr(a, x), p, 1e-4);
        }
    }
}

#[test]
fn beta_quantile_round_trips_in_f32() {
    for &(a, b) in &[(1.0f32, 1.0f32), (2.0, 5.0), (5.0, 2.0)] {
        for i in 1..20 {
            let p = (i as f32) * 0.05;
            let x = beta_reg_inv(p, a, b);
            assert!((0.0..=1.0).contains(&x), "beta_reg_inv({p},{a},{b}) = {x}");
            close(beta_reg(a, b, x), p, 1e-4);
        }
    }
}

/// A **small** root has to be resolved relative to itself, not to `1`.
///
/// The regression for a defect this file's other tests could not see. `beta_reg_inv` stopped
/// bisecting once `hi - lo` fell below a bare `4·ε = 4.8e-7`, an *absolute* bound — so for
/// any root below that it returned the bracket midpoint. `I_x(0.1, 0.1) = 0.05` came back as
/// `2.38e-7`, which is `2⁻²²` and visibly the bisection halting, against a true root of
/// `8.66e-11`: **2,752 times too large**. Its sibling `gamma_lr_inv` and the f64 peer both
/// scaled by the bracket; this one did not.
///
/// Nothing above reaches it because the three shapes tested are `(1,1)`, `(2,5)` and `(5,2)`,
/// whose roots all sit mid-interval where an absolute `4.8e-7` is harmless. A U-shaped beta
/// — either parameter below 1 — piles its mass into the corners and pushes the root toward
/// an endpoint, which is the same reason the f64 suite has a `quantiles::extremes` file.
///
/// The reference is the `f64` kernel, four orders finer than `f32` can represent. The bound
/// is `1e-5` against a measured worst of `2.98e-7` (about 2.5 f32 ulp), leaving room for the
/// CDF's own narrowing error without admitting anything like the defect.
#[test]
fn the_beta_quantile_resolves_a_root_near_an_endpoint() {
    for &(a, b) in &[(0.1f32, 0.1f32), (0.5, 0.5), (0.2, 3.0), (3.0, 0.2)] {
        for i in 1..20 {
            let p = (i as f32) * 0.05;
            let got = beta_reg_inv(p, a, b);
            let truth =
                crate::kernels::quantile::beta_reg_inv(f64::from(p), f64::from(a), f64::from(b));
            // Skip where the true root is not a normal f32; there the format answers.
            if truth < f64::from(f32::MIN_POSITIVE) * 16.0 || truth > 1.0 - 1e-6 {
                continue;
            }
            let relative = ((f64::from(got) - truth) / truth).abs();
            assert!(
                relative < 1e-5,
                "beta_reg_inv({p}, {a}, {b}) = {got:e}, f64 says {truth:e} \
                 (relative {relative:.3e}) — the bisection stopped on an absolute bound"
            );
        }
    }
}

/// The gamma quantile's five routes, asserted as decisions.
#[test]
fn the_gamma_quantile_routes_cover_the_probability_range() {
    use GammaQuantileRoute::{AtInfinity, AtZero, RoundsToShape, Solve, Undefined};

    for (p, a) in [
        (0.5, 0.0),
        (0.5, -1.0),
        (0.5, f32::NAN),
        (0.5, f32::INFINITY),
        (f32::NAN, 2.0),
        ((-0.0_f32).next_down(), 2.0),
        (1.0_f32.next_up(), 2.0),
        (-1.0, 2.0),
        (2.0, 2.0),
    ] {
        assert_eq!(
            GammaQuantileRoute::for_arguments(p, a),
            Undefined,
            "at p = {p:e}, a = {a:e}"
        );
    }
    assert_eq!(GammaQuantileRoute::for_arguments(0.0, 2.0), AtZero);
    assert_eq!(GammaQuantileRoute::for_arguments(1.0, 2.0), AtInfinity);
    // The snap needs `ulp(a) ≥ 128√a`, which for `a = 2^k` in this format is
    // `k − 23 ≥ 7 + k/2`, so `k ≥ 60` — against `k ≥ 118` in `f64`, the 29-bit mantissa
    // difference showing up directly in where the distribution stops being resolvable.
    for p in [f32::MIN_POSITIVE, 1e-12, 0.25, 0.5, 0.75, 1.0 - 1e-6] {
        assert_eq!(GammaQuantileRoute::for_arguments(p, 2.0), Solve);
        assert_eq!(GammaQuantileRoute::for_arguments(p, 2f32.powi(50)), Solve);
        assert_eq!(
            GammaQuantileRoute::for_arguments(p, 2f32.powi(64)),
            RoundsToShape
        );
    }
}

/// The two tails, and the fact that they solve the same root through different equations.
#[test]
fn the_gamma_tail_switches_at_the_median_and_agrees_with_itself() {
    use GammaTail::{Lower, Upper};

    // Half itself is the lower tail: the comparison is strict.
    assert_eq!(GammaTail::for_probability(0.5), Lower);
    assert_eq!(GammaTail::for_probability(0.5_f32.next_up()), Upper);
    assert_eq!(GammaTail::for_probability(0.5_f32.next_down()), Lower);
    assert_eq!(GammaTail::for_probability(0.0), Lower);
    assert_eq!(GammaTail::for_probability(1.0), Upper);

    // The target moves with the arm, and only for the upper one.
    assert_eq!(Lower.target(0.3), 0.3);
    assert_eq!(Upper.target(0.75), -0.25);

    // `P = 1 − Q`, so the two residuals agree at every `x`. That identity is what makes the
    // switch free, and it is the only thing tying the two arms together.
    let a = 3.0;
    for x in [0.5_f32, 1.0, 3.0, 7.0, 20.0] {
        let p = 0.75;
        let lower = Lower.cdf(a, x) - Lower.target(p);
        let upper = Upper.cdf(a, x) - Upper.target(p);
        assert!(
            (lower - upper).abs() <= 1e-6,
            "at x = {x}: {lower} vs {upper}"
        );
    }
    // And the arms are genuinely different functions of `x`.
    assert!(Lower.cdf(a, 3.0) > 0.0);
    assert!(Upper.cdf(a, 3.0) < 0.0);
}

/// The beta quantile's four routes. Its upper endpoint is attained, unlike the gamma's.
#[test]
fn the_beta_quantile_routes_attain_both_endpoints() {
    use BetaQuantileRoute::{AtOne, AtZero, Solve, Undefined};

    for (p, a, b) in [
        (0.5, 0.0, 1.0),
        (0.5, 1.0, 0.0),
        (0.5, -1.0, 1.0),
        (0.5, 1.0, -1.0),
        (0.5, f32::NAN, 1.0),
        (0.5, 1.0, f32::NAN),
        (0.5, f32::INFINITY, 1.0),
        (0.5, 1.0, f32::INFINITY),
        (f32::NAN, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (2.0, 1.0, 1.0),
    ] {
        assert_eq!(
            BetaQuantileRoute::for_arguments(p, a, b),
            Undefined,
            "at p = {p:e}, a = {a:e}, b = {b:e}"
        );
    }
    assert_eq!(BetaQuantileRoute::for_arguments(0.0, 2.0, 3.0), AtZero);
    assert_eq!(BetaQuantileRoute::for_arguments(1.0, 2.0, 3.0), AtOne);
    // `I_x(a, b) = 1` at `x = 1` for every shape pair, which is the arm the gamma lacks.
    assert_eq!(beta_reg_inv(1.0, 0.1, 0.01), 1.0);
    for p in [f32::MIN_POSITIVE, 1e-12, 0.5, 1.0 - 1e-6] {
        assert_eq!(BetaQuantileRoute::for_arguments(p, 2.0, 3.0), Solve);
        // Including the U-shaped densities, where the mass piles into both corners.
        assert_eq!(BetaQuantileRoute::for_arguments(p, 0.1, 0.01), Solve);
    }
}
