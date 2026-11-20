//! The principal branch of the Lambert `W`, and the three seeds that reach it.
//!
//! A child module so the parent keeps both its sub-topics — `W` and the Riemann zeta — inside the
//! crate's 400-line budget while carrying tests for each; the `f32` peer at
//! [`crate::kernels::f32::transcendental`] splits the same way and for the same reason. The split
//! is also where the decomposition lives: [`LambertDomain`], [`LambertSeed`], [`LambertResidual`]
//! and [`lambert_step`] are named so the routing and the arithmetic can be asserted as values,
//! which the converged root cannot do — Halley reaches the same root from any of them.

use crate::constants::f64::NEG_INV_E;

/// Iteration cap for the `W` refinement. Halley's method triples the correct digits each
/// step, so convergence is reached in a handful from any of the starting guesses below.
///
/// Generous rather than measured, and the generosity is load-bearing: the Taylor seed is carried
/// to `x = 3`, where it returns `34.5` against a true `W(3) = 1.05`, and about twenty steps go on
/// iterating that away. The `f32` peer rederived the split at `1.5` and the cap at `12` for
/// exactly this reason; here the extra steps cost accuracy nothing and the number stays.
const W_STEPS: u32 = 60;

/// Width of the branch-point window, in `x` above `−1/e`.
///
/// The seed inside it is a series in `p = √(2(1 + ex))`, and `1 + ex` is a difference of two
/// `O(1)` quantities that cancels as `x → −1/e`, so the window cannot be widened indefinitely
/// without the radicand losing all its significance. The `f32` peer needs `1e-2`, `2^29` further
/// out, because the same absolute cancellation error is that much larger relative to its format.
const W_BRANCH_POINT_WINDOW: f64 = 1e-3;

/// Above this argument the logarithmic seed replaces the Taylor one.
///
/// `3.0` is well past where `x − x² + 3x³/2` is a useful approximation — see [`W_STEPS`], which
/// is sized to absorb the difference. `ln x − ln ln x` is defined for `x > 1` and is the closer
/// seed from about `x ≈ 1`; the `f32` peer measured the crossover and moved its split to `1.5`.
const W_LOGARITHMIC_MIN: f64 = 3.0;

/// Whether `W` has a root to iterate towards at this argument, and which exact value if not.
///
/// The three non-iterating arms are not special cases of the iteration — each is a value Halley
/// cannot produce. `NaN` and the arguments below `−1/e` have no real root at all, and the two
/// infinities are outside the format rather than inside it.
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
    fn for_argument(x: f64) -> Self {
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

/// Which starting guess [`lambert_w`] uses, and so which regime the argument is in.
///
/// Halley needs a seed in the right basin, and these three are not refinements of one formula —
/// each is the only one that works where it is used, and the `BranchPoint` arm exists precisely
/// because iteration cannot work there at all.
///
/// # Why the seed's accuracy is the wrong thing to assert
///
/// Sixteen mutants lived in these three lines and survived the whole suite, and the reason is not
/// that the seed is good. It is that the seed does not have to be: the Taylor branch returns
/// `34.5` at `x = 3`, where `W(3)` is `1.05`, and Halley recovers from that without trouble
/// because it is allowed [sixty steps](W_STEPS). So a test that demanded the seed be *close*
/// would either fail against the real kernel or be too loose to notice a mutation.
///
/// Nor is its cost, as the code stands. Counting steps until `w` lands within four ulps of the
/// converged answer, this seed needs **22 at `x = 3`** — the top of the Taylor regime — while four
/// mutated variants need eleven, twenty-two, twenty-two and twenty-two over the same range. The
/// real seed is not better than the mutants by the measure that would have to catch them.
///
/// What *is* assertable is the decision and the arithmetic, separately: which arm an argument
/// selects, and what each arm's series evaluates to. Both are pinned in this module's tests, and
/// neither needs the root to move.
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
    fn for_argument(x: f64) -> Self {
        if x < NEG_INV_E + W_BRANCH_POINT_WINDOW {
            Self::BranchPoint
        } else if x > W_LOGARITHMIC_MIN {
            Self::Logarithmic
        } else {
            Self::Taylor
        }
    }

    /// The seed itself.
    fn evaluate(self, x: f64) -> f64 {
        match self {
            Self::BranchPoint => {
                let p = (2.0 * (1.0 + core::f64::consts::E * x)).max(0.0).sqrt();
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
/// The two are the same equation divided through by `e^w`, so they vanish at the same root; what
/// differs is which intermediate leaves the format first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LambertResidual {
    /// `x > 0`: divide the residual and both denominators by `e^w`. `W(MAX)` is finite, but
    /// `w·e^w` can overflow after a rounding-sized overshoot and `f·(w+2)` can overflow even
    /// before that. The scaled equation gives the same Halley step without either large product.
    Scaled,
    /// `x ≤ 0`: keep the unscaled residual near the negative branch point, where subtracting
    /// `x/e^w` would introduce a different cancellation.
    Unscaled,
}

impl LambertResidual {
    /// The sign of the argument decides; `x = 0` never reaches the iteration, so the boundary
    /// belongs to whichever arm — it is [`LambertDomain::AtOrigin`] that answers there.
    fn for_argument(x: f64) -> Self {
        if x > 0.0 {
            Self::Scaled
        } else {
            Self::Unscaled
        }
    }
}

/// The residual `f` and Newton's denominator, in whichever form keeps the products in range.
fn lambert_residual(w: f64, x: f64, ew: f64) -> (f64, f64) {
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
fn lambert_step(w: f64, f: f64, denom_newton: f64) -> f64 {
    let correction = f * (w + 2.0) / (2.0 * w + 2.0);
    let denom = denom_newton - correction;
    if denom == 0.0 || !denom.is_finite() {
        f / denom_newton
    } else {
        f / denom
    }
}

/// The principal branch of the Lambert `W` function: the `w ≥ −1` with `w e^w = x`.
///
/// Defined for `x ≥ −1/e`; `NaN` below that, where the real branches end.
///
/// # Method
///
/// Halley's method on `f(w) = w e^w − x`, which converges cubically, from a starting guess chosen
/// per regime; see [`LambertSeed`] for the three and [`LambertDomain`] for the arguments that
/// never reach the iteration at all.
pub(crate) fn lambert_w(x: f64) -> f64 {
    match LambertDomain::for_argument(x) {
        LambertDomain::Undefined => f64::NAN,
        LambertDomain::AtOrigin => x,
        LambertDomain::AtInfinity => f64::INFINITY,
        LambertDomain::Iterate => lambert_halley(x),
    }
}

/// Halley's iteration proper, on an argument already known to have a finite root.
fn lambert_halley(x: f64) -> f64 {
    let mut w = LambertSeed::for_argument(x).evaluate(x);
    for _ in 0..W_STEPS {
        let ew = w.exp();
        let (f, denom_newton) = lambert_residual(w, x, ew);
        if f == 0.0 {
            return w;
        }
        let step = lambert_step(w, f, denom_newton);
        if !step.is_finite() {
            break;
        }
        let next = w - step;
        if next == w {
            break;
        }
        w = next;
    }
    w
}

#[cfg(test)]
mod tests {
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
}
