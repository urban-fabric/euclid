//! Unit tests for the single-precision Lambert `W`.
//!
//! A grandchild module so implementation plus tests stay inside the crate 400-line budget,
//! while `use super::*` still reaches `LambertDomain`, `LambertSeed`, `LambertResidual`,
//! `lambert_step`, `lambert_w_steps` and this format rederived constants.
//!
//! Two things are asserted here that the `f64` peer cannot be: the **step count**, which is the
//! only observable the period-two cycle check changes, and the seed policy cost, which is what
//! sets `W_LOGARITHMIC_MIN` at `1.5` rather than the `f64` peer `3.0`.

use super::*;

#[test]
fn lambert_w_satisfies_its_defining_equation_in_f32() {
    // w e^w = x, to f32 resolution. Checked as a *relative* comparison because W spans
    // many decades across this sweep.
    for i in 0..40 {
        let x = -0.36f32 + (i as f32) * 0.5;
        let w = lambert_w(x);
        assert!(w.is_finite(), "W({x}) = {w}");
        let back = w * w.exp();
        assert!(
            (back - x).abs() <= 1e-5 * x.abs().max(1.0),
            "W({x}) = {w} does not invert: {back}"
        );
    }
}

/// The domain arms, including both ends of the branch and both infinities.
#[test]
fn the_domain_arms_cover_the_branch_and_both_infinities() {
    use LambertDomain::{AtInfinity, AtOrigin, Iterate, Undefined};
    for (x, want) in [
        (f32::NAN, Undefined),
        (f32::NEG_INFINITY, Undefined),
        (-1.0, Undefined),
        (NEG_INV_E.next_down(), Undefined),
        (NEG_INV_E, Iterate),
        (NEG_INV_E.next_up(), Iterate),
        (-0.1, Iterate),
        (0.0, AtOrigin),
        (-0.0, AtOrigin),
        (f32::MIN_POSITIVE, Iterate),
        (1.0, Iterate),
        (f32::MAX, Iterate),
        (f32::INFINITY, AtInfinity),
    ] {
        assert_eq!(LambertDomain::for_argument(x), want, "at {x:e}");
    }
    // The origin returns the argument, so a negative zero survives.
    assert!(lambert_w(-0.0).is_sign_negative());
    assert_eq!(lambert_w(-0.0), 0.0);
}

/// Every seed has to land in Halley's basin, which is the whole reason for three of them.
#[test]
fn each_lambert_seed_converges_from_its_own_regime() {
    for (x, expected) in [
        (NEG_INV_E + 1e-4, LambertSeed::BranchPoint),
        (NEG_INV_E + 1e-3, LambertSeed::BranchPoint),
        (-0.2, LambertSeed::Taylor),
        (0.5, LambertSeed::Taylor),
        // The boundary itself, and either side of it.
        (1.5, LambertSeed::Taylor),
        (1.6, LambertSeed::Logarithmic),
        (3.0, LambertSeed::Logarithmic),
        (1e20, LambertSeed::Logarithmic),
    ] {
        assert_eq!(
            LambertSeed::for_argument(x),
            expected,
            "regime at x = {x:e}"
        );
        let w = lambert_w(x);
        assert!(w.is_finite() && w >= -1.0, "W({x:e}) = {w} left the branch");
        let back = w * w.exp();
        assert!(
            (back - x).abs() <= 1e-5 * x.abs().max(1.0),
            "W({x:e}) = {w} does not invert: {back}"
        );
    }
}

/// The seed regimes, at and either side of both boundaries.
///
/// The sibling test checks that each seed *converges*; this one checks which seed is chosen,
/// which convergence cannot see — every one of them lands in Halley's basin, so the routing
/// is free to be wrong without the root moving.
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
    assert_eq!(LambertSeed::for_argument(f32::MAX), Logarithmic);
}

/// Each seed as a *value*, which is the only way its arithmetic is reachable.
#[test]
fn each_seed_is_its_own_series() {
    use LambertSeed::{BranchPoint, Logarithmic, Taylor};
    // `x − x² + 3x³/2`, written Horner-wise as `x(1 − x + 1.5x²)`. Exact at these points.
    assert_eq!(Taylor.evaluate(0.5), 0.4375);
    assert_eq!(Taylor.evaluate(1.0), 1.5);
    assert_eq!(Taylor.evaluate(1.5), 4.3125);
    assert_eq!(Taylor.evaluate(0.0), 0.0);
    assert!((Taylor.evaluate(-0.3) - -0.430_5).abs() <= 1e-6);

    // `ln x − ln ln x`, from `w + ln w = ln x`.
    for (x, want) in [
        (2.0_f32, 1.059_660_1_f32),
        (10.0, 1.468_552_6),
        (1e30, 64.842_32),
    ] {
        let got = Logarithmic.evaluate(x);
        assert!(
            (got - want).abs() <= 1e-5 * want.abs(),
            "at {x}: {got} vs {want}"
        );
    }

    // The branch-point series in `p = √(2(1 + ex))`, which is where every Newton-type step
    // stalls. At `−1/e` itself `p` is zero and the seed is exactly `−1`, the branch's end.
    assert_eq!(BranchPoint.evaluate(NEG_INV_E), -1.0);
    // Below `−1/e` the radicand goes negative; the `max(0)` keeps it at the endpoint rather
    // than returning a `NaN` from the square root.
    assert_eq!(BranchPoint.evaluate(-0.4), -1.0);
    for (x, want) in [(-0.36_f32, -0.805_953_4_f32), (-0.365, -0.879_801_6)] {
        let got = BranchPoint.evaluate(x);
        assert!((got - want).abs() <= 1e-5, "at {x}: {got} vs {want}");
    }
    // Every seed is inside the branch, which is what makes it a seed at all.
    for x in [-0.36_f32, -0.2, 0.5, 1.0, 2.0, 1e10, f32::MAX] {
        let seed = LambertSeed::for_argument(x).evaluate(x);
        assert!(seed >= -1.0 && seed.is_finite(), "seed at {x} is {seed}");
    }
}

#[test]
fn the_residual_form_follows_the_sign_of_the_argument() {
    use LambertResidual::{Scaled, Unscaled};
    assert_eq!(LambertResidual::for_argument(1.0), Scaled);
    assert_eq!(LambertResidual::for_argument(f32::MIN_POSITIVE), Scaled);
    assert_eq!(LambertResidual::for_argument(0.0), Unscaled);
    assert_eq!(LambertResidual::for_argument(-0.0), Unscaled);
    assert_eq!(LambertResidual::for_argument(-0.2), Unscaled);

    // Both forms vanish at the root, which is what makes them the same equation: at `x = 1`
    // the root is `W(1) = 0.567143`, so each residual is zero there to rounding.
    let (w, x) = (0.567_143_3_f32, 1.0_f32);
    let ew = w.exp();
    let (scaled, _) = lambert_residual(w, x, ew);
    let (unscaled, _) = lambert_residual(w, -x, ew);
    assert!(scaled.abs() < 1e-6, "{scaled}");
    assert!((unscaled - (w * ew + x)).abs() < 1e-6);

    // The denominators differ by exactly `e^w`, which is the whole content of the scaling.
    let (_, scaled_denominator) = lambert_residual(w, x, ew);
    let (_, unscaled_denominator) = lambert_residual(w, -x, ew);
    assert_eq!(scaled_denominator, 1.0 + w);
    assert_eq!(unscaled_denominator, ew * (1.0 + w));
}

#[test]
fn the_halley_step_falls_back_to_newton_where_its_denominator_degenerates() {
    // Ordinary case: `f / (denom_newton − f(w+2)/(2w+2))`, rebuilt here independently.
    for (w, f, denom_newton) in [
        (0.5_f32, 0.25_f32, 1.5_f32),
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
    let correction = 1.0 * (0.0 + 2.0) / (2.0 * 0.0 + 2.0);
    assert_eq!(correction, 1.0);
    assert_eq!(lambert_step(0.0, 1.0, 1.0), 1.0);

    // Carried out of the format: a huge residual makes the correction infinite, and the
    // fallback keeps the step finite where `f/denom` would be a zero or a `NaN`.
    let step = lambert_step(1.0, f32::MAX, 2.0);
    assert!(!(f32::MAX * 3.0 / 4.0).is_finite());
    assert_eq!(step, f32::MAX / 2.0);
    assert!(step.is_finite());
}

#[test]
fn the_scaled_residual_reaches_the_top_of_the_format() {
    // `W(MAX)` is `84.237`, and `w·e^w` at that root is `MAX` itself — so the unscaled
    // residual overflows after any rounding-sized overshoot. This is the argument the scaled
    // form exists for, and the only one that distinguishes the two.
    let w = lambert_w(f32::MAX);
    assert!(w.is_finite(), "W(MAX) = {w}");
    assert!((w - 84.288_59).abs() < 1e-3, "{w}");
    // The root inverts, formed in the log domain because `w·e^w` is at the format's ceiling.
    assert!((w.ln() + w - f32::MAX.ln()).abs() < 5e-3, "{w}");
    // The seed is `84.2373`, so the iteration moves the root by `0.05` — about 6,700 ulp at
    // this magnitude. The scaled residual is what lets those steps be taken at all.
    let seed = LambertSeed::for_argument(f32::MAX).evaluate(f32::MAX);
    assert!((seed - 84.237_32).abs() < 1e-3, "{seed}");
    assert!(
        (w - seed).abs() > 0.04,
        "the iteration did not move: {w} from {seed}"
    );
    assert_eq!(
        LambertSeed::for_argument(f32::MAX),
        LambertSeed::Logarithmic
    );
}

#[test]
fn the_branch_point_itself_is_on_the_branch() {
    // `W(−1/e) = −1` exactly: the comparison that rejects arguments below the branch point is
    // strict, and the endpoint belongs to the domain. Widening it to `<=` returns `NaN` for
    // the one argument where the answer is the most exactly known value the function has.
    assert_eq!(lambert_w(NEG_INV_E), -1.0);
    assert!(lambert_w(NEG_INV_E.next_down()).is_nan());
    assert!(lambert_w(-0.4).is_nan());
    // It is reached through the branch-point seed, which is already exact there, so the
    // residual is zero on the first step and the iteration stops immediately.
    assert_eq!(
        LambertSeed::for_argument(NEG_INV_E),
        LambertSeed::BranchPoint
    );
    assert_eq!(lambert_w_steps(NEG_INV_E), (-1.0, 1));
}

/// The iteration's **cost**, which is the only thing the cycle check changes.
///
/// Asserted as a step count because the root is identical either way: weakening
/// `next == w || next == previous` to `&&` leaves every value in this crate unchanged and
/// costs the remaining iterations on every argument that lands in a period-two cycle. That is
/// a performance regression with no accuracy signature, and this is the instrument for it.
#[test]
fn no_argument_runs_the_halley_loop_to_its_cap() {
    let mut worst = 0;
    let mut worst_at = 0.0;
    for i in 0..8000 {
        let x = NEG_INV_E + (i as f32) * 1e-3;
        let (w, steps) = lambert_w_steps(x);
        assert!(w.is_finite(), "W({x}) = {w}");
        assert!(
            steps < W_STEPS,
            "W({x}) ran the loop to its cap of {W_STEPS}"
        );
        if steps > worst {
            worst = steps;
            worst_at = x;
        }
    }
    // Measured: **7**, at `x = 1.389`, against the cap of twelve. `W_STEPS`' own
    // documentation quotes 6 for this seed policy; the difference is the counting
    // convention — this counts iterations *entered*, so an argument whose residual is
    // already zero counts one. The figure is pinned rather than bounded because a change to
    // it is exactly what a regression in the seed policy or the cycle check looks like.
    assert_eq!(worst, 7, "worst case moved to {worst} at x = {worst_at}");
}
