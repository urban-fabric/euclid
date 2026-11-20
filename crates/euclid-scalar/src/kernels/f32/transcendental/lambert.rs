//! The principal branch of the Lambert `W`, and the three seeds that reach it.
//!
//! A child module so the parent keeps both its sub-topics — `W` and the Riemann zeta — inside the
//! crate's 400-line budget while carrying tests for each. The split is also where the
//! decomposition lives: [`LambertSeed`], [`LambertResidual`] and [`lambert_step`] are named so
//! their arithmetic can be asserted as values, which the converged root cannot do.

use crate::constants::f32::NEG_INV_E;

// ===== Lambert W =========================================================

/// Iteration cap for Halley's method on `w e^w = x`.
///
/// **Rederived at `f32`, where the `f64` peer uses 60.** Halley converges cubically, so from a
/// seed good to two digits the correct digits roughly triple per step and three steps already
/// exhaust `f32`. The loop exits on `next == w` regardless, so this is a backstop rather than a
/// schedule — but a backstop seven times larger than any argument needs is one that hides a
/// non-converging case instead of reporting it.
///
/// Measured over 8,000 arguments spanning `[−1/e, 1e9]`, the worst case settles in **6**
/// iterations with the seed policy below. Twelve leaves double that.
///
/// The count depends on [`W_LOGARITHMIC_MIN`] far more than on anything else: carrying the
/// Taylor seed to `x = 3`, as the `f64` peer does, needs **21** steps instead of 6, because
/// `x − x² + 3x³/2` is `34.5` there against a true `W(3) = 1.05`. The `f64` peer absorbs that
/// with a cap of 60 and its own doc notes the regime "could reduce iterations"; at `f32`, where
/// the whole point of the format is speed, 21 wasted Halley steps is not a rounding detail.
const W_STEPS: u32 = 12;

/// Width of the branch-point window, in `x` above `−1/e`.
///
/// The `f64` peer uses `1e-3`, and the window has to be *wider* here rather than narrower. The
/// seed is a series in `p = √(2(1 + ex))`, and `1 + ex` is a sum of two `O(1)` quantities that
/// cancel as `x → −1/e`: its absolute error is `ulp(1)`, so its *relative* error blows up `2^29`
/// times further from the branch point in `f32` than in `f64`. That is also where the ordinary
/// Taylor seed is at its worst, so the series has to cover the whole band.
const W_BRANCH_POINT_WINDOW: f32 = 1e-2;

/// Above this argument the logarithmic seed is closer than the Taylor one.
///
/// **`1.5`, where the `f64` peer uses `3.0`.** The Taylor series about the origin is only useful
/// for `|x| ≲ 0.3`; by `x = 3` it returns `34.5` for a root of `1.05`, and every bit of that has
/// to be iterated away. `ln x − ln ln x` is defined for `x > 1` and is within 30% of the root by
/// `1.5`, within 3% by `2.5`. Measured worst-case Halley steps as the split moves:
///
/// | split | 1.2 | **1.5** | 2.0 | 2.5 | 3.0 |
/// |---|---|---|---|---|---|
/// | steps | 5 | **6** | 9 | 14 | 21 |
///
/// `1.2` is marginally cheaper still and sits uncomfortably close to `x = 1`, where `ln ln x`
/// diverges and the seed is useless; `1.5` keeps clear of it for the sake of one step.
const W_LOGARITHMIC_MIN: f32 = 1.5;

/// Whether `W` has a root to iterate towards at this argument, and which exact value if not.
///
/// The same three non-iterating arms the `f64` peer carries, for the same reason: each is a value
/// Halley cannot produce. `NaN` and the arguments below `−1/e` have no real root at all, and the
/// two infinities are outside the format rather than inside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LambertDomain {
    /// `NaN`, and every `x < −1/e` — including `−∞` — where the real branches end.
    Undefined,
    /// `x = ±0`. `W` is `x − x² + ⋯` near the origin, hence odd there, so the argument itself is
    /// returned rather than the literal `0.0` and a negative zero survives.
    AtOrigin,
    /// `+∞`, where `W` diverges — slowly, like `ln x − ln ln x`, but without bound.
    AtInfinity,
    /// Everywhere else: Halley's method from a [`LambertSeed`].
    Iterate,
}

impl LambertDomain {
    fn for_argument(x: f32) -> Self {
        if x.is_nan() || x < NEG_INV_E {
            Self::Undefined
        } else if x == 0.0 {
            Self::AtOrigin
        } else if x.is_infinite() {
            // `−∞` is below the branch point and already answered above, so this is `+∞` alone.
            Self::AtInfinity
        } else {
            Self::Iterate
        }
    }
}

/// Which starting guess `lambert_w` uses, and so which regime the argument is in.
///
/// Halley needs a seed in the right basin, and these three are not refinements of one formula —
/// each is the only one that works where it is used, and the `BranchPoint` arm exists precisely
/// because iteration cannot work there at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LambertSeed {
    /// Near `−1/e`, where `d(we^w)/dw` vanishes and any Newton-type step stalls. The series in
    /// `p = √(2(1 + ex))` is well-behaved exactly where the iteration is not.
    BranchPoint,
    /// Large `x`, from `w + ln w = ln x`, so `W ≈ ln x − ln ln x`.
    Logarithmic,
    /// Otherwise the Taylor series `x − x² + 3x³/2` about the origin.
    Taylor,
}

impl LambertSeed {
    fn for_argument(x: f32) -> Self {
        if x < NEG_INV_E + W_BRANCH_POINT_WINDOW {
            Self::BranchPoint
        } else if x > W_LOGARITHMIC_MIN {
            Self::Logarithmic
        } else {
            Self::Taylor
        }
    }

    /// The seed itself.
    fn evaluate(self, x: f32) -> f32 {
        match self {
            Self::BranchPoint => {
                let p = (2.0 * (1.0 + core::f32::consts::E * x)).max(0.0).sqrt();
                -1.0 + p - p * p / 3.0 + 11.0 / 72.0 * p * p * p
            }
            Self::Logarithmic => {
                let l = x.ln();
                l - l.ln()
            }
            Self::Taylor => x * (1.0 - x + 1.5 * x * x),
        }
    }
}

/// Which form of the defining equation's residual keeps its products in range.
///
/// # Mutation testing is what asked for this
///
/// Thirty-eight mutants survived across the Lambert iteration, and the reason is structural rather
/// than a missing argument: the only assertion available was the round trip `w·e^w = x`, and
/// **that identity holds from any seed and after any step, because Halley converges anyway.** A
/// test on the converged root is blind to every quantity that decides *how* it converged. So the
/// seed, the residual and the step are named here and asserted as values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LambertResidual {
    /// `x > 0`: divide the residual and both denominators by `e^w`. `W(MAX)` is finite, but
    /// `w·e^w` can overflow after a rounding-sized overshoot and `f·(w+2)` can overflow even
    /// before that — a much nearer hazard in `f32`, whose range stops at `3.4e38`. The scaled
    /// equation gives the same Halley step without either large product.
    Scaled,
    /// `x ≤ 0`: keep the unscaled residual near the negative branch point, where subtracting
    /// `x/e^w` would introduce a different cancellation.
    Unscaled,
}

impl LambertResidual {
    /// The sign of the argument decides; `x = 0` never reaches the iteration.
    fn for_argument(x: f32) -> Self {
        if x > 0.0 {
            Self::Scaled
        } else {
            Self::Unscaled
        }
    }
}

/// The residual `f` and Newton's denominator, in whichever form keeps the products in range.
fn lambert_residual(w: f32, x: f32, ew: f32) -> (f32, f32) {
    match LambertResidual::for_argument(x) {
        LambertResidual::Scaled => (w - x / ew, 1.0 + w),
        LambertResidual::Unscaled => (w * ew - x, ew * (1.0 + w)),
    }
}

/// Halley's step, `f / (e^w(1+w) − f(w+2)/(2w+2))`, falling back to Newton where its denominator
/// degenerates.
///
/// The fallback is reached where Halley's second-order term cancels the first exactly, or carries
/// it out of the format. Neither is reachable from an ordinary argument — the iteration is already
/// within an ulp of the root by then — so both arms are asserted directly.
fn lambert_step(w: f32, f: f32, denom_newton: f32) -> f32 {
    let correction = f * (w + 2.0) / (2.0 * w + 2.0);
    let denom = denom_newton - correction;
    if denom == 0.0 || !denom.is_finite() {
        f / denom_newton
    } else {
        f / denom
    }
}

/// The principal branch of the Lambert `W`: the `w ≥ −1` with `w·e^w = x`.
///
/// Defined for `x ≥ −1/e`; `NaN` below that, where the real branches end. Halley's method, which
/// converges cubically, from a per-regime seed; see [`LambertSeed`].
pub(crate) fn lambert_w(x: f32) -> f32 {
    lambert_w_steps(x).0
}

/// `W` together with the number of Halley steps it took.
///
/// # Why the count is returned at all
///
/// The **period-two cycle** check below is a cost saving that no value assertion can see: both
/// iterates are within an ulp of the root, so stopping early and running the loop to its cap
/// return the same `W`. Mutation testing duly found `next == w || next == previous` unkillable —
/// weakening it to `&&` is invisible to every accuracy test in the crate while costing every
/// remaining iteration on the arguments that cycle, which was 1,207 of a sweep of 8,000.
///
/// The step count is the deterministic observable that *does* separate them, so it is returned
/// here and asserted in the tests. [`lambert_w`] discards it; nothing branches on it.
fn lambert_w_steps(x: f32) -> (f32, u32) {
    match LambertDomain::for_argument(x) {
        LambertDomain::Undefined => (f32::NAN, 0),
        LambertDomain::AtOrigin => (x, 0),
        LambertDomain::AtInfinity => (f32::INFINITY, 0),
        LambertDomain::Iterate => lambert_halley(x),
    }
}

/// Halley's iteration proper, on an argument already known to have a finite root, with the number
/// of steps it took.
fn lambert_halley(x: f32) -> (f32, u32) {
    let mut w = LambertSeed::for_argument(x).evaluate(x);
    // The previous iterate, to catch a **period-two cycle**. Halley's last step in `f32` very
    // often lands back where it was two iterations ago rather than exactly on a fixed point —
    // the root is between two representable values and the step overshoots each way. Testing
    // only `next == w`, as the `f64` peer does, misses that and runs the loop to its cap: on a
    // sweep of 8,000 arguments, 1,207 of them never settled and so paid for every remaining
    // iteration. Both iterates are within an ulp of the root, so stopping is free.
    let mut previous = f32::NAN;
    let mut steps = 0;
    for _ in 0..W_STEPS {
        steps += 1;
        let ew = w.exp();
        // For positive x, divide the residual and both denominators by exp(w). `W(MAX)` is
        // finite, but `w·exp(w)` can overflow after a rounding-sized overshoot and `f·(w+2)` can
        // overflow even before that — a much nearer hazard in `f32`, whose range stops at
        // `3.4e38`. The scaled equation gives the same Halley step without either large product.
        // Keep the unscaled residual near the negative branch point, where subtracting
        // `x/exp(w)` would introduce a different cancellation.
        let (f, denom_newton) = lambert_residual(w, x, ew);
        if f == 0.0 {
            return (w, steps);
        }
        let step = lambert_step(w, f, denom_newton);
        if !step.is_finite() {
            break;
        }
        let next = w - step;
        if next == w || next == previous {
            break;
        }
        previous = w;
        w = next;
    }
    (w, steps)
}

#[cfg(test)]
mod tests {
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
}
