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
/// **A backstop rather than a schedule, which it did not use to be.** The measured worst case is
/// **23 steps**, at `x = 2.96` — the top of the Taylor band, where the seed returns `34.5`
/// against a true `W(3) = 1.05` and the iteration has to work that off. Sixty leaves 2.6× over
/// that, which is headroom rather than waste because nothing reaches it any more.
///
/// It is left at sixty deliberately. The `f32` peer argues for a tight cap — "a backstop seven
/// times larger than any argument needs is one that hides a non-converging case instead of
/// reporting it" — and that argument is now served by
/// [`no_argument_runs_the_halley_loop_to_its_cap`](tests), which fails if any argument reaches
/// the cap at all. With the cap *tested* rather than merely sized, lowering it would only convert
/// a loop that runs long into a root that is silently truncated, and the measured worst case is
/// over one band of a domain that spans the whole format.
///
/// Before the period-two cycle check in [`lambert_w_steps`], 1,064 arguments out of a
/// 27,218-point sweep ran all sixty.
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
    lambert_w_steps(x).0
}

/// `W` together with the number of Halley steps it took.
///
/// # Why the count is returned at all
///
/// The **period-two cycle** check below is a cost saving that no value assertion can see, so the
/// step count is the only deterministic observable that separates having it from not. The `f32`
/// peer returns its count for the same reason and records the same finding; this kernel went
/// without the check for longer, and the measurement is worth keeping:
///
/// Over a sweep of 27,218 arguments spanning the whole branch — dense at the branch point, every
/// decade to the top of the format, and across the seed split — testing only `next == w`
/// **ran the loop to its cap of 60 on 1,064 of them**. Halley's last step in `f64` often lands
/// back where it was two iterations ago rather than exactly on a fixed point: the root lies
/// between two representable values and the step overshoots each way forever. Adding
/// `next == previous` takes the worst case to **23 steps** and the total across that sweep from
/// **101,829 to 47,751 — 53% of the iterations gone**.
///
/// Those are counts, so they reproduce on any machine. What they cost is close to nothing:
/// 26,259 of the 27,218 roots are returned bit-identically, 350 move, and of those the residual
/// `|ln w + w − ln x|` is *better* in 83 cases and worse in 267 — worst case `3.3e-16` against
/// `1.1e-16`, which is a step from half an ulp to one and a half. The accuracy envelope over the
/// whole sweep does not move at all: `1.137e-13` either way.
///
/// [`lambert_w`] discards the count; nothing in the kernel branches on it.
fn lambert_w_steps(x: f64) -> (f64, u32) {
    match LambertDomain::for_argument(x) {
        LambertDomain::Undefined => (f64::NAN, 0),
        LambertDomain::AtOrigin => (x, 0),
        LambertDomain::AtInfinity => (f64::INFINITY, 0),
        LambertDomain::Iterate => lambert_halley(x),
    }
}

/// Halley's iteration proper, on an argument already known to have a finite root, with the number
/// of steps it took.
fn lambert_halley(x: f64) -> (f64, u32) {
    let mut w = LambertSeed::for_argument(x).evaluate(x);
    // The previous iterate, to catch a period-two cycle; see [`lambert_w_steps`] for what that
    // costs and what it saves. `NaN` so the first comparison cannot fire by accident.
    let mut previous = f64::NAN;
    let mut steps = 0;
    for _ in 0..W_STEPS {
        steps += 1;
        let ew = w.exp();
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
mod tests;
