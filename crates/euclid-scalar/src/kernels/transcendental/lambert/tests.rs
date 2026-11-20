//! Unit tests for the Lambert `W`, its three seeds and its Halley step.
//!
//! A grandchild module so implementation plus tests stay inside the crate 400-line budget,
//! while `use super::*` still reaches `LambertDomain`, `LambertSeed`, `LambertResidual`,
//! `lambert_step` and the two regime constants.
//!
//! What is asserted here is the *decomposition* rather than the root. Halley converges to the
//! same `W` from any seed and after any step, so the converged value is blind to every quantity
//! that decided how it got there — which is why the seed, the residual form and the step are
//! each pinned as values, and why the seed series is evaluated at arguments where its `p` is
//! nonzero. Both of those lessons came from mutation testing.

use super::*;
use crate::constants::f64::OMEGA;

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(1.0),
        "{a:e} vs {b:e} (tol {tol:e})"
    );
}

#[test]
fn lambert_w_satisfies_its_defining_equation() {
    // w e^w = x, checked across the whole real domain.
    for i in 0..200 {
        let x = NEG_INV_E + f64::from(i) * 0.5;
        let w = lambert_w(x);
        assert!(w.is_finite(), "W({x}) = {w}");
        close(w * w.exp(), x, 1e-11);
    }
}

#[test]
fn lambert_w_hits_its_known_values() {
    assert_eq!(lambert_w(0.0), 0.0);
    close(lambert_w(1.0), OMEGA, 1e-14);
    close(lambert_w(core::f64::consts::E), 1.0, 1e-14);
    close(lambert_w(NEG_INV_E), -1.0, 1e-7);
    // W(x ln x) = ln x for x >= 1/e.
    for i in 1..40 {
        let x = f64::from(i) * 0.5 + 0.5;
        let l = x.ln();
        close(lambert_w(l * x), l, 1e-10);
    }
}

#[test]
fn lambert_w_is_accurate_near_the_branch_point_and_far_out() {
    // The branch point is where Newton-type iteration stalls, so it gets its own series.
    for &d in &[0.0f64, 1e-12, 1e-9, 1e-6, 1e-3] {
        let x = NEG_INV_E + d;
        let w = lambert_w(x);
        assert!(w >= -1.0 - 1e-6, "W({x}) = {w} must be >= -1");
        close(w * w.exp(), x, 1e-6);
    }
    // And for very large arguments, where the start is ln x - ln ln x.
    for &x in &[1e3f64, 1e10, 1e100, 1e300] {
        let w = lambert_w(x);
        assert!(w.is_finite() && w > 0.0, "W({x}) = {w}");
        // w + ln w = ln x
        close(w + w.ln(), x.ln(), 1e-12);
    }
}

#[test]
fn lambert_w_is_monotone_and_rejects_its_domain_edge() {
    let mut prev = f64::NEG_INFINITY;
    for i in 0..500 {
        let x = NEG_INV_E + f64::from(i) * 0.1;
        let w = lambert_w(x);
        assert!(w >= prev - 1e-9, "W must increase: {prev} then {w}");
        prev = w;
    }
    assert!(
        lambert_w(NEG_INV_E - 1e-6).is_nan(),
        "below the branch point"
    );
    assert!(lambert_w(-1.0).is_nan());
    assert!(lambert_w(f64::NAN).is_nan());
    assert!(lambert_w(f64::INFINITY).is_infinite());
}

/// The domain arms, including both ends of the branch and both infinities.
///
/// `W(−1/e) = −1` exactly, so the comparison that rejects arguments below the branch point is
/// strict and the endpoint belongs to the domain: widening it to `<=` would return `NaN` for
/// the one argument where the answer is the most exactly known value the function has.
#[test]
fn the_domain_arms_cover_the_branch_and_both_infinities() {
    use LambertDomain::{AtInfinity, AtOrigin, Iterate, Undefined};
    for (x, want) in [
        (f64::NAN, Undefined),
        (f64::NEG_INFINITY, Undefined),
        (-1.0, Undefined),
        (NEG_INV_E.next_down(), Undefined),
        (NEG_INV_E, Iterate),
        (NEG_INV_E.next_up(), Iterate),
        (-0.1, Iterate),
        (0.0, AtOrigin),
        (-0.0, AtOrigin),
        (f64::MIN_POSITIVE, Iterate),
        (1.0, Iterate),
        (f64::MAX, Iterate),
        (f64::INFINITY, AtInfinity),
    ] {
        assert_eq!(LambertDomain::for_argument(x), want, "at {x:e}");
    }
    // The origin returns the argument, so a negative zero survives.
    assert!(lambert_w(-0.0).is_sign_negative());
    assert_eq!(lambert_w(-0.0), 0.0);
}

/// The seed regimes, at and either side of both boundaries.
///
/// Every one of these seeds lands in Halley's basin, so the routing is free to be wrong
/// without the root moving — which is why the decision is asserted here directly rather than
/// through a value.
#[test]
fn the_seed_regimes_are_half_open_at_both_boundaries() {
    use LambertSeed::{BranchPoint, Logarithmic, Taylor};
    let window = NEG_INV_E + W_BRANCH_POINT_WINDOW;
    assert_eq!(LambertSeed::for_argument(window.next_down()), BranchPoint);
    assert_eq!(LambertSeed::for_argument(window), Taylor);
    assert_eq!(LambertSeed::for_argument(window.next_up()), Taylor);
    assert_eq!(LambertSeed::for_argument(NEG_INV_E), BranchPoint);
    assert_eq!(LambertSeed::for_argument(W_LOGARITHMIC_MIN), Taylor);
    assert_eq!(
        LambertSeed::for_argument(W_LOGARITHMIC_MIN.next_up()),
        Logarithmic
    );
    assert_eq!(
        LambertSeed::for_argument(W_LOGARITHMIC_MIN.next_down()),
        Taylor
    );
    assert_eq!(LambertSeed::for_argument(f64::MAX), Logarithmic);

    // And each one converges from its own regime, which is the whole reason for three.
    for (x, want) in [
        (NEG_INV_E + 1e-4, BranchPoint),
        (-0.2, Taylor),
        (0.5, Taylor),
        (3.0, Taylor),
        (3.5, Logarithmic),
        (1e20, Logarithmic),
    ] {
        assert_eq!(LambertSeed::for_argument(x), want, "regime at {x:e}");
        let w = lambert_w(x);
        assert!(w.is_finite() && w >= -1.0, "W({x:e}) = {w} left the branch");
        close(w * w.exp(), x, 1e-10);
    }
}

/// Each seed as a *value*, which is the only way its arithmetic is reachable.
#[test]
fn each_seed_is_its_own_series() {
    use LambertSeed::{BranchPoint, Logarithmic, Taylor};
    // `x − x² + 3x³/2`, written Horner-wise as `x(1 − x + 1.5x²)`. Exact at these points.
    assert_eq!(Taylor.evaluate(0.5), 0.4375);
    assert_eq!(Taylor.evaluate(1.0), 1.5);
    assert_eq!(Taylor.evaluate(0.0), 0.0);
    // The top of the Taylor regime, where the seed is `34.5` against a root of `1.05`. This
    // is the number `W_STEPS` is sized to absorb.
    assert_eq!(Taylor.evaluate(3.0), 34.5);
    close(Taylor.evaluate(-0.3), -0.4305, 1e-12);

    // `ln x − ln ln x`, from `w + ln w = ln x`.
    for (x, want) in [
        (2.0_f64, 1.059_660_101_141_609_6),
        (10.0, 1.468_552_647_746_09),
        (1e30, 64.842_322_962_911_25),
    ] {
        close(Logarithmic.evaluate(x), want, 1e-13);
    }

    // The branch-point series in `p = √(2(1 + ex))`, which is where every Newton-type step
    // stalls. At `−1/e` itself `p` is zero to rounding and the seed is `−1`, the branch's end.
    close(BranchPoint.evaluate(NEG_INV_E), -1.0, 1e-15);
    // Below `−1/e` the radicand goes negative; the `max(0)` keeps the seed at the endpoint
    // rather than returning a `NaN` from the square root.
    assert_eq!(BranchPoint.evaluate(-0.4), -1.0);
    // **At arguments where `p` is nonzero**, which is what actually exercises the series.
    // Both assertions above sit at `p = 0` — one at the branch point, one clamped — so every
    // term past the leading `−1` drops out and the polynomial is untested by them. Mutation
    // testing found exactly that: eight operators in this one expression survived, and these
    // two rows are what kill them. References from the series evaluated independently.
    for (x, want) in [
        (-0.36_f64, -0.805_953_279_255_721),
        (-0.365, -0.879_801_849_454_059_6),
    ] {
        close(BranchPoint.evaluate(x), want, 1e-13);
    }
    // Every seed is inside the branch, which is what makes it a seed at all.
    for x in [-0.367_8_f64, -0.2, 0.5, 1.0, 2.0, 1e10, f64::MAX] {
        let seed = LambertSeed::for_argument(x).evaluate(x);
        assert!(seed >= -1.0 && seed.is_finite(), "seed at {x:e} is {seed}");
    }
}

#[test]
fn the_residual_form_follows_the_sign_of_the_argument() {
    use LambertResidual::{Scaled, Unscaled};
    assert_eq!(LambertResidual::for_argument(1.0), Scaled);
    assert_eq!(LambertResidual::for_argument(f64::MIN_POSITIVE), Scaled);
    assert_eq!(LambertResidual::for_argument(0.0), Unscaled);
    assert_eq!(LambertResidual::for_argument(-0.0), Unscaled);
    assert_eq!(LambertResidual::for_argument(-0.2), Unscaled);

    // Both forms vanish at the root, which is what makes them the same equation: at `x = 1`
    // the root is `OMEGA`, so the scaled residual is zero there to rounding.
    let (w, x) = (OMEGA, 1.0_f64);
    let ew = w.exp();
    let (scaled, scaled_denominator) = lambert_residual(w, x, ew);
    let (unscaled, unscaled_denominator) = lambert_residual(w, -x, ew);
    assert!(scaled.abs() < 1e-15, "{scaled}");
    assert!((unscaled - (w * ew + x)).abs() < 1e-15);

    // The denominators differ by exactly `e^w`, which is the whole content of the scaling.
    assert_eq!(scaled_denominator, 1.0 + w);
    assert_eq!(unscaled_denominator, ew * (1.0 + w));
}

#[test]
fn the_halley_step_falls_back_to_newton_where_its_denominator_degenerates() {
    // Ordinary case: `f / (denom_newton − f(w+2)/(2w+2))`, rebuilt here independently.
    for (w, f, denom_newton) in [
        (0.5_f64, 0.25_f64, 1.5_f64),
        (-0.5, -0.1, 0.5),
        (2.0, 1.0, 3.0),
    ] {
        let correction = f * (w + 2.0) / (2.0 * w + 2.0);
        assert_eq!(
            lambert_step(w, f, denom_newton),
            f / (denom_newton - correction)
        );
    }

    // Exactly degenerate: at `w = 0` the correction is `f`, so `denom_newton = 1` with
    // `f = 1` cancels Halley's denominator to zero and the step must become Newton's `f/1`.
    assert_eq!(1.0 * (0.0 + 2.0) / (2.0 * 0.0 + 2.0), 1.0);
    assert_eq!(lambert_step(0.0, 1.0, 1.0), 1.0);

    // Carried out of the format: a huge residual makes the correction infinite, and the
    // fallback keeps the step finite where `f/denom` would be a zero or a `NaN`.
    let step = lambert_step(1.0, f64::MAX, 2.0);
    assert!(!(f64::MAX * 3.0 / 4.0).is_finite());
    assert_eq!(step, f64::MAX / 2.0);
    assert!(step.is_finite());
}

/// The scaled residual is what carries the iteration to the top of the format.
#[test]
fn the_scaled_residual_reaches_the_top_of_the_format() {
    // `w·e^w` at `W(MAX)` is `MAX` itself, so the unscaled residual overflows after any
    // rounding-sized overshoot. This is the argument the scaled form exists for.
    let w = lambert_w(f64::MAX);
    assert!(w.is_finite(), "W(MAX) = {w}");
    assert_eq!(
        LambertSeed::for_argument(f64::MAX),
        LambertSeed::Logarithmic
    );
    // The root inverts, formed in the log domain because `w·e^w` is at the format's ceiling.
    close(w.ln() + w, f64::MAX.ln(), 1e-14);
}
