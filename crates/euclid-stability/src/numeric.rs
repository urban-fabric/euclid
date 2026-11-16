//! Hunting numerical instability, rather than asserting its absence.
//!
//! [`tolerance`](crate::tolerance) answers "are these two numbers close". This module answers
//! the harder questions: *where* does a kernel go wrong, and is the error it has the error it is
//! entitled to. Three instruments, each aimed at a failure that a pointwise accuracy test
//! structurally cannot see.
//!
//! # Seams
//!
//! Almost every special-function kernel switches algorithm at thresholds — a series near zero,
//! a continued fraction in the middle, an asymptotic form far out. If the two branches disagree
//! at the crossover, the function has a **discontinuity** at a point no test happened to sample,
//! and every value on one side is systematically off.
//!
//! [`seam_ratio`] measures it with no reference value at all: for a mathematically continuous
//! `f`, moving the input by one representable step must move the output by about
//! `|f'(x)| · ulp(x)`. A step a million times larger than that is a seam, and the only way to
//! produce one is a bug. This finds branch-boundary errors *without knowing where the branches
//! are*.
//!
//! # Wiggle
//!
//! A kernel can be accurate at every point and still not **monotone**, if its error oscillates
//! by more than the spacing between consecutive outputs. For a CDF, a quantile, or anything
//! feeding a root-find or a binary search, that is a real defect: the search can fail to
//! converge even though every individual value is within tolerance. Pointwise accuracy tests
//! cannot see it, because the property is about the *relationship* between neighbours.
//!
//! [`scan_monotone`] walks consecutive representable values and reports the first inversion.
//!
//! # Conditioning-derived tolerances
//!
//! The usual way to pick a tolerance is to loosen it until the test passes, which measures
//! nothing. The defensible way is to derive it: the relative condition number of `f` at `x` is
//! `κ(x) = |x·f'(x) / f(x)|`, and a **forward-stable** implementation has relative error around
//! `κ(x)·ε`. So the bound comes from the problem rather than from the implementation, and the
//! property becomes "this kernel is forward stable" instead of "this kernel is within a number
//! I chose".
//!
//! [`condition_at`] estimates `κ`; [`forward_error_bound`] turns it into a tolerance. Where the
//! observed error exceeds it, the *implementation* is unstable; where `κ` itself is huge, the
//! **problem** is ill-conditioned and no implementation can do better — which is a distinction
//! worth being able to make, and one a hand-picked tolerance erases.

/// `f64`'s unit round-off: the largest relative error a single correctly-rounded operation can
/// introduce. `EPSILON` is the gap at 1, and round-to-nearest halves it.
pub const UNIT_ROUNDOFF: f64 = f64::EPSILON / 2.0;

/// How much larger `f`'s step between two adjacent inputs is than it ought to be.
///
/// Stepping the input by one ulp should move the output by about `|f'(x)| · ulp(x)` — or, if
/// that is below the output's own representational floor, by about `ulp(f(x))`. This returns the
/// ratio of the *observed* step to that expectation, so a smooth function scores around 1 and a
/// branch disagreement scores in the millions.
///
/// # Why not simply count ulps
///
/// Because that measure breaks at a zero of `f`, which the first version of this module did.
/// `ln(1.0)` is exactly `0` and `ln(next_up(1.0))` is `2.2e-16`; those are four quadrillion ulps
/// apart, since ulp spacing collapses towards zero while the derivative does not. A raw ulp
/// count therefore reports a seam at every root — the detector fired on `ln` and on `sqrt`
/// before it fired on anything real.
///
/// Dividing by the expected step fixes it, because at a root the expected step collapses by
/// exactly the same factor the observed one does. The ratio stays near 1 and the detector stays
/// quiet where it should.
///
/// Returns `None` where the comparison is meaningless: non-finite values, or a derivative
/// estimate that cannot be formed.
///
/// # Three permanent survivors under `cargo mutants`, and why
///
/// This function keeps three equivalent mutants that cannot be excluded in
/// `.cargo/mutants.toml`, because `exclude_re` matches a mutation's *description* and not its
/// line — and each of these shares its description with a mutation elsewhere in the same
/// function that is real and pinned. Recorded here so the next person to read a report does not
/// re-derive it:
///
/// - **Both `||`s in the finiteness guard below** (`!here.is_finite() || !there.is_finite() ||
///   next == x`). Weakening either is unobservable, because the closing
///   `ratio.is_finite().then_some(ratio)` reaches the same verdict unaided: a non-finite `here`
///   or `there` makes `observed` infinite or NaN, so the ratio is too, so the result is `None`
///   either way. The guard is an early-out, not a correctness gate.
/// - **`x + window == x` mutated to `x - window == x`** in the window guard. The clause
///   immediately after it already *is* `x - window == x`, so the mutation duplicates a test that
///   is still performed.
///
/// Everything else in both guards is real and has a test. Note especially that the *other*
/// mutations of the window guard are **not** equivalent, which is why the exclusions cannot be
/// written by operator: at `x = -f64::INFINITY` three of them make this function report a finite
/// ratio of zero for an argument where nothing was measured. See
/// `seam_ratio_declines_an_infinite_argument`.
#[must_use]
pub fn seam_ratio(f: impl Fn(f64) -> f64, x: f64) -> Option<f64> {
    let next = next_up(x);
    let (here, there) = (f(x), f(next));
    if !here.is_finite() || !there.is_finite() || next == x {
        return None;
    }
    let observed = (there - here).abs();

    // The derivative is estimated over a window far wider than one ulp, so a seam inside the
    // window cannot inflate the expectation and hide itself.
    let window = f64::EPSILON.sqrt() * x.abs().max(f64::MIN_POSITIVE);
    // Decline in the subnormal range. Two things break there at once: the derivative window is
    // enormous relative to `x` (it is floored at `MIN_POSITIVE` to avoid underflowing to
    // nothing), and a one-ulp step is a *percent-scale* relative change rather than a
    // sixteen-digit one, because subnormals have so few significant bits left. So consecutive
    // outputs legitimately differ hugely and the whole premise of the measure fails. `cbrt` was
    // reported as seamed at `1e-321` for exactly this reason — the detector's fault, not
    // `cbrt`'s.
    let subnormal = |v: f64| v != 0.0 && v.abs() < f64::MIN_POSITIVE;
    if subnormal(x) || subnormal(next) || x + window == x || x - window == x {
        return None;
    }
    let (up, down) = (f(x + window), f(x - window));
    if !up.is_finite() || !down.is_finite() {
        return None;
    }
    let derivative = ((up - down) / (2.0 * window)).abs();

    let input_step = (next - x).abs();
    // Two floors, and both are needed. The mathematical change is `|f'| * ulp(x)`; the
    // representational floor is `ulp(f(x))`, below which a correctly-rounded kernel may still
    // move by a bit or two for no reason at all.
    let expected = (derivative * input_step)
        .max(ulp_of(here))
        .max(f64::MIN_POSITIVE);

    let ratio = observed / expected;
    ratio.is_finite().then_some(ratio)
}

/// The gap between `x` and the next representable value away from zero.
fn ulp_of(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::INFINITY;
    }
    (next_up(x.abs()) - x.abs()).abs()
}

/// The next representable `f64` above `x`.
#[must_use]
pub fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    let bits = x.to_bits();
    f64::from_bits(if x > 0.0 { bits + 1 } else { bits - 1 })
}

/// A seam found by [`scan_for_seam`], [`scan_window`] or [`hunt_seams`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Seam {
    /// The input just below the jump.
    pub at: f64,
    /// The input just above it.
    pub next: f64,
    /// `f` at those two inputs.
    pub values: (f64, f64),
    /// How many times larger the step was than expected. See [`seam_ratio`].
    pub ratio: f64,
}

/// Walk `steps` consecutive representable values up from `start` and return the first place `f`
/// steps more than `tolerated` times further than it should.
///
/// Scanning *adjacent* values is what makes this work: there is no true value to compare
/// against, only the requirement that a continuous function cannot move far when its input
/// barely moves. That turns "is this kernel right" — which needs an oracle — into "is this
/// kernel continuous", which needs nothing.
///
/// `tolerated` is a multiple of the expected step, so it is scale-free. A smooth kernel scores
/// about 1 and rarely exceeds 10; something in the hundreds is worth looking at, and 1e6 is a
/// branch disagreement.
#[must_use]
pub fn scan_for_seam(
    f: impl Fn(f64) -> f64,
    start: f64,
    steps: usize,
    tolerated: f64,
) -> Option<Seam> {
    let mut x = start;
    for _ in 0..steps {
        let next = next_up(x);
        if next == x {
            break;
        }
        if let Some(ratio) = seam_ratio(&f, x)
            && ratio > tolerated
        {
            return Some(Seam {
                at: x,
                next,
                values: (f(x), f(next)),
                ratio,
            });
        }
        x = next;
    }
    None
}

/// Values a kernel is likely to switch algorithm at.
///
/// This list is the strategy, and it is worth being explicit about why it beats sweeping.
/// Thresholds are chosen by implementers, and implementers choose *round* numbers: a power of
/// two because it is exact, a small integer because a series converges by there, a half because
/// it halves an interval. A seam at `1.5` or `8.0` is overwhelmingly more likely than one at
/// `3.7214`, so the search should spend its budget there.
pub const LIKELY_THRESHOLDS: &[f64] = &[
    0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 16.0, 20.0, 25.0,
    32.0, 50.0, 64.0, 100.0, 128.0, 256.0, 512.0, 1024.0,
];

/// Scan `steps` consecutive representable values either side of `centre`.
///
/// The counterpart to [`scan_for_seam`], which only walks upward. A threshold sits *between* two
/// branches, so the scan has to start below it.
#[must_use]
pub fn scan_window(
    f: impl Fn(f64) -> f64,
    centre: f64,
    steps: usize,
    tolerated: f64,
) -> Option<Seam> {
    let mut start = centre;
    for _ in 0..steps {
        start = next_down(start);
    }
    scan_for_seam(f, start, steps * 2, tolerated)
}

/// The next representable `f64` below `x`.
#[must_use]
pub fn next_down(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = x.to_bits();
    f64::from_bits(if x > 0.0 { bits - 1 } else { bits + 1 })
}

/// Scan a window around each candidate and return every seam found, worst first.
///
/// # Why not sweep the range instead
///
/// Because a sparse sweep cannot work, and finding that out is worth recording. The obvious
/// design samples the range geometrically and checks each sample against its neighbour — but
/// each check still spans only one ulp, so it detects a seam only if a sample happens to land
/// within one ulp of it. With a few thousand samples across twelve decades that probability is
/// indistinguishable from zero, and a planted seam at `1e6` went undetected.
///
/// Widening the comparison to span whole sampling intervals does not rescue it either. Over an
/// interval the function legitimately changes by `|f'| · Δx`, which for any reasonable sampling
/// density dwarfs a small discontinuity — a `1e-7` jump is invisible against a `0.0069` genuine
/// change. A coarse scan can only find a seam comparable in size to the function's own
/// variation, which is not the interesting case.
///
/// So the search is directed rather than uniform: scan *densely* in the places a threshold is
/// likely to be. [`LIKELY_THRESHOLDS`] is the default list, and passing the kernel's documented
/// crossovers does better still.
#[must_use]
pub fn hunt_seams(
    f: impl Fn(f64) -> f64,
    candidates: &[f64],
    steps: usize,
    tolerated: f64,
) -> Vec<Seam> {
    let mut found: Vec<Seam> = candidates
        .iter()
        .filter_map(|&centre| scan_window(&f, centre, steps, tolerated))
        .collect();
    found.sort_by(|a, b| b.ratio.total_cmp(&a.ratio));
    found
}

/// An order inversion found by [`scan_monotone`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Inversion {
    /// The input whose output was too large.
    pub at: f64,
    /// The next input up, whose output was smaller.
    pub next: f64,
    /// The two outputs, in input order.
    pub values: (f64, f64),
}

/// Walk `steps` consecutive representable values up from `start` and return the first place a
/// non-decreasing `f` decreases.
///
/// Equality is allowed — a saturating or plateauing function is still monotone. Only a strict
/// decrease is an inversion.
#[must_use]
pub fn scan_monotone(f: impl Fn(f64) -> f64, start: f64, steps: usize) -> Option<Inversion> {
    let mut x = start;
    let mut here = f(x);
    for _ in 0..steps {
        let next = next_up(x);
        if next == x {
            break;
        }
        let there = f(next);
        if here.is_finite() && there.is_finite() && there < here {
            return Some(Inversion {
                at: x,
                next,
                values: (here, there),
            });
        }
        x = next;
        here = there;
    }
    None
}

/// Estimate the relative condition number `κ(x) = |x·f'(x) / f(x)|` by central difference.
///
/// How much `f` amplifies a relative perturbation of its input. `κ ≈ 1` is benign; `κ = 1e8`
/// means eight digits are lost to the *problem* regardless of how the kernel is written, which
/// is the distinction this function exists to make.
///
/// The step is `∜ε·|x|`, the standard choice for a central difference: large enough that the
/// subtraction `f(x+h) - f(x-h)` does not vanish into round-off, small enough that the `O(h²)`
/// truncation stays below it. The estimate is a few digits, not exact, so use it to decide
/// whether an error is *explained* rather than to assert a precise bound.
///
/// Returns `None` where it is undefined — at `x = 0`, where a relative condition number has no
/// meaning, or where `f` is non-finite nearby.
#[must_use]
pub fn condition_at(f: impl Fn(f64) -> f64, x: f64) -> Option<f64> {
    if x == 0.0 || !x.is_finite() {
        return None;
    }
    let value = f(x);
    if value == 0.0 || !value.is_finite() {
        return None;
    }

    let step = f64::EPSILON.powf(0.25) * x.abs();
    let (up, down) = (f(x + step), f(x - step));
    if !up.is_finite() || !down.is_finite() {
        return None;
    }

    let derivative = (up - down) / (2.0 * step);
    let condition = (x * derivative / value).abs();
    condition.is_finite().then_some(condition)
}

/// The relative error a **forward-stable** implementation of `f` is entitled to at `x`.
///
/// `safety · κ(x) · ε`, floored at `safety · ε` so a perfectly conditioned point still gets the
/// round-off it cannot avoid. `safety` absorbs the few operations any real kernel performs
/// beyond the one the condition number accounts for; 8 to 32 is reasonable, and a kernel needing
/// 1e6 is not forward stable.
///
/// Using this instead of a literal turns the assertion from "within a number I picked" into
/// "forward stable", which is a claim about the implementation that survives someone rewriting
/// it. It also fails *informatively*: an error above this bound at low `κ` is an unstable
/// implementation, while a large `κ` says the problem itself is the difficulty.
#[must_use]
pub fn forward_error_bound(f: impl Fn(f64) -> f64, x: f64, safety: f64) -> Option<f64> {
    let condition = condition_at(f, x)?;
    Some((safety * condition.max(1.0) * UNIT_ROUNDOFF).max(safety * UNIT_ROUNDOFF))
}

/// The relative error a forward-stable **two-stage** computation `outer(inner(x))` is entitled
/// to.
///
/// # Why a composition needs its own bound
///
/// This exists because of a result the suite produced and the obvious approach gets wrong.
/// Applying [`forward_error_bound`] to a composition measures the conditioning of the composite
/// *map*, and for `erf_inv(erf(x))` that is 1 — it is the identity. Yet the computed round trip
/// has a relative error around eleven times round-off, and exceeded the naive bound by a factor
/// of five.
///
/// The composite being well conditioned does not make the *computation* stable, because the
/// error is introduced in the **intermediate**. `erf` saturates, so at `x = 1.76` its derivative
/// is `0.05`: a relative error of `ε` in `erf(x)` becomes an absolute error that `erf_inv`
/// amplifies by `1/0.05 ≈ 20`. The two stages' condition numbers multiply to 1 precisely because
/// they are inverses, and that cancellation hides the amplification rather than describing it.
///
/// The standard bound keeps both terms: `(κ_outer·κ_inner + κ_outer)·u`. The first is the input
/// error carried through both stages; the second is the *inner stage's own rounding*, amplified
/// by the outer stage. For the round trip above that is `1 + 11 ≈ 12`, which is what is
/// observed — so the computation is forward stable after all, and only the bound was wrong.
///
/// The practical reading: never compose through a saturating intermediate and expect the
/// composite's conditioning to describe the result.
#[must_use]
pub fn two_stage_error_bound(
    inner: impl Fn(f64) -> f64,
    outer: impl Fn(f64) -> f64,
    x: f64,
    safety: f64,
) -> Option<f64> {
    let inner_condition = condition_at(&inner, x)?;
    let intermediate = inner(x);
    let outer_condition = condition_at(&outer, intermediate)?;

    let amplification = outer_condition * inner_condition + outer_condition;
    Some(safety * amplification.max(1.0) * UNIT_ROUNDOFF)
}

#[cfg(test)]
mod tests;
