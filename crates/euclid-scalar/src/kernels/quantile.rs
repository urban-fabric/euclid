//! Quantiles of the gamma and beta distributions: the inverses of the regularized incomplete
//! gamma and beta functions.
//!
//! Interior quantiles use bracketed Newton steps on the forward CDF. Accuracy depends on the
//! CDF and the spacing of representable inputs. At sufficiently large gamma shapes, an analytic
//! tail bound identifies the single representable value to which all interior quantiles round.
//!
//! The structure is the same in both cases:
//!
//! 1. A cheap starting guess, good to a few digits.
//! 2. Expand outward from it to a bracket with the root strictly inside.
//! 3. Newton steps, falling back to bisection whenever a step would leave the bracket.

use super::gamma::ln_gamma;
use super::normal::ndtri;
use crate::math::density::{beta_pdf, gamma_pdf, log_beta_normaliser};
use crate::math::roots::{bracket_is_narrow, bracketed_newton_step, wilson_hilferty_guess};

/// Iteration cap for the root-find.
///
/// Sized to cross `f64`'s **exponent** range, not its mantissa. The ~60 steps that exhaust a
/// mantissa are enough only once the bracket is a single octave wide; starting from `[0, 1]`,
/// as the beta quantile does, bisection has to walk *down* one octave per step to reach a
/// small root, and there are 1075 octaves between 1 and the smallest subnormal.
///
/// At 200 this silently truncated. `beta_reg_inv(p, 1e-3, b)` stopped after 200 halvings at
/// `2^-200 ≈ 3.1e-61` and returned that for *every* `p` — the true median for `a = 1e-3` is
/// `0.5^1000 ≈ 9.3e-302`, perfectly representable — so the quantile was a constant function of
/// its own argument over the whole small-shape regime.
///
/// Raising it costs nothing in the ordinary case: [`solve_increasing`] exits as soon as the
/// bracket stops shrinking relative to its own magnitude, which Newton reaches in a handful of
/// steps.
const MAX_STEPS: u32 = 1_200;

/// Cap on the geometric bracket expansion, in octaves.
///
/// Separate from [`MAX_STEPS`] because it measures something different — how far the seed may
/// be from the root — and because getting it wrong is silent. At 200 the gamma quantile could
/// expand its bracket only `2^±200` from the Wilson–Hilferty seed, so for `a = 1e-3` it
/// returned `6.2e-64` for every `p` from `1e-12` to `0.5`, the root at `9.3e-302` lying far
/// below where the search gave up. 1100 octaves spans `f64` from `MIN_POSITIVE` to `MAX`.
const BRACKET_OCTAVES: u32 = 1_100;

/// Coefficient on the shape in [`residual_tolerance`]; see there for the measurement.
const RESIDUAL_SHAPE_COEFF: f64 = 2.0;

/// Upper bound on [`residual_tolerance`]. See there for why an uncapped shape term is a bug.
const RESIDUAL_TOLERANCE_CAP: f64 = 9.313_225_746_154_785e-10;

/// How close the residual must get before the root is as good as it is going to be.
///
/// # Why this scales with the shape
///
/// Asking for a residual below an ulp of `p` is the right target at a small shape and
/// unreachable at a large one. Inverting a CDF at shape `a` returns a root of size `O(a)`, so
/// the answer is only resolved to `ulp(a)`, and the probability it maps to therefore has a floor
/// of roughly `ε·a` relative. Measured backward error at the flat one-ulp tolerance:
///
/// ```text
/// shape      0.5       2       1e2       1e3       1e4
/// floor/eps    2       2       254      2023     65000
/// ```
///
/// — about `2` to `6` times `ε·a`. Below that floor the loop is subdividing a bracket whose
/// endpoints all map to the same `p`, which costs bisections and buys nothing.
///
/// # Choosing the coefficient
///
/// Swept against `gamma_lr_inv` over 299 probabilities per shape, steps and backward error both:
///
/// ```text
/// coeff   a=0.5    a=2     a=1e2    a=1e3    a=1e4     worst backward-error change
///   0     2197    2267     5282     5752     3537      (baseline, flat one ulp)
///   1    -6.7%  -11.6%   -21.1%   -24.7%   -29.1%      a=2 only, 4.5e-16 -> 6.4e-16
///   2   -10.0%  -13.3%   -28.8%   -39.9%   -50.6%      a=2 only, 4.5e-16 -> 1.1e-15
///   3   -11.6%  -13.6%   -31.8%   -43.1%   -69.2%      a=1e3 degrades too, 4.5e-13 -> 6.6e-13
/// ```
///
/// `2` is the knee. It is the largest coefficient at which no shape except `a = 2` moves at all
/// — `a = 1e3` is still bit-identical and `a = 1e2` and `a = 1e4` come out slightly *better* —
/// and the `a = 2` cost is `2` ulp of `p` becoming `5`, against the `≈384` ulp this inverse is
/// allowed by `the_gamma_quantile_inverts_its_own_cdf`. At `3` the large-shape gain keeps
/// growing but `a = 1e3` starts to pay, which is the signal to stop.
///
/// `1` is the choice that costs nothing measurable at all, if that is ever wanted back.
///
/// # The shape term must be capped, and leaving it uncapped was a real bug
///
/// `ε·(1 + 2a)` is a *relative* tolerance on a probability, so it is only meaningful while it
/// stays far below `1`. The fit above was measured over `a ≤ 1e4`, where `ε·2a ≤ 4.4e-12`, and
/// extrapolating it is nonsense: at `a = 2^120` it evaluates to `5.8e17`, which accepts **any**
/// residual whatsoever. The solver then returned its first midpoint, and
/// `gamma_quantile_checks_the_format_spacing_before_snapping_to_the_shape` caught it — that test
/// asserts the `1e-4` quantile is still distinguishable from the mean at `a = 2^110`, and it
/// no longer was.
///
/// [`RESIDUAL_TOLERANCE_CAP`] bounds it at `2^-30`, which is above the whole measured range and
/// far below the point where the tolerance stops discriminating. Shapes past that get no benefit
/// from this exit, which is correct: nothing out there was measured, and the genuinely
/// unresolvable regime is already handled by [`gamma_quantiles_round_to_shape`].
///
/// Note the failure only appeared under `EUCLID_STABILITY_SOAK=100`. At the default effort the
/// suite passed.
fn residual_tolerance(shape: f64) -> f64 {
    let scaled = f64::EPSILON * (1.0 + RESIDUAL_SHAPE_COEFF * shape.max(0.0));
    if scaled.is_finite() {
        scaled.min(RESIDUAL_TOLERANCE_CAP)
    } else {
        RESIDUAL_TOLERANCE_CAP
    }
}

/// Return whichever of a converged bracket's endpoints or midpoint comes closest to `p`.
///
/// The loop below converges to a bracket one or two ulps wide, and the midpoint of that
/// bracket is not necessarily the best of the three candidates. Usually it makes no
/// difference; where the CDF is steep it makes a large one. For `I_x(0.1, 0.01)` the whole
/// climb from `0.34` to `1` happens inside the last ulp below 1, so adjacent representable `x`
/// differ by `2.7e-4` of probability — and picking the wrong one of three throws that away.
///
/// The endpoint residuals are passed in rather than re-derived. An endpoint only becomes one by
/// being an earlier `x` whose `cdf(x)` the loop had just evaluated, so recomputing them here
/// spent two of this routine's three CDF evaluations on numbers the caller already held — and a
/// CDF evaluation is the expensive thing in the whole solve, a series or continued fraction that
/// needs `O(√a)` terms. `cdf` is a pure function of `x`, so a remembered residual is the same
/// `f64` the call would return and the choice between the three candidates is unchanged to the
/// bit.
///
/// `None` means the loop exited before that side ever moved, which only the first iteration can
/// do; then the evaluation still happens.
fn best_of(
    cdf: &impl Fn(f64) -> f64,
    p: f64,
    lo: f64,
    hi: f64,
    residual_lo: Option<f64>,
    residual_hi: Option<f64>,
) -> f64 {
    let mut best = lo.midpoint(hi);
    let mut err = (cdf(best) - p).abs();
    for (candidate, known) in [(lo, residual_lo), (hi, residual_hi)] {
        let other = known.unwrap_or_else(|| cdf(candidate) - p).abs();
        if other < err {
            err = other;
            best = candidate;
        }
    }
    best
}

/// Solve an increasing CDF within a bracket. The stopping rule scales with the bracket so small
/// roots receive the same relative resolution as large roots.
fn solve_increasing(
    cdf: impl Fn(f64) -> f64,
    pdf: impl Fn(f64) -> f64,
    p: f64,
    mut lo: f64,
    mut hi: f64,
    residual_tolerance: f64,
) -> f64 {
    let mut x = lo.midpoint(hi);
    // The residual at each endpoint, kept as the endpoint is set so [`best_of`] does not have to
    // re-evaluate the CDF to recover it.
    let (mut residual_lo, mut residual_hi) = (None, None);
    for _ in 0..MAX_STEPS {
        let f = cdf(x) - p;
        if f == 0.0 {
            return x;
        }
        // Stop once the residual is within one ulp of `p`. Any `x` whose CDF lands this close is
        // as good an answer as any other, because the remaining disagreement is below what `p`
        // itself can represent — and the loop's other exit, a bracket narrowed to an ulp, can
        // take dozens of bisections to reach in a flat region of the CDF.
        //
        // Measured on `gamma_lr_inv` over 599 probabilities per shape, against the identical
        // bracket and CDF: **9.9% fewer steps at `a = 0.5` and 16.9% at `a = 2`**, with the
        // backward error `|P(a, x) − p| / p` **bit-identical** at every shape tested, and no
        // change at all for large shapes where the residual floor sits above an ulp anyway. The
        // returned root can move by up to 9 ulp, which is the width of the set of `x` sharing
        // that residual rather than an error.
        //
        // One ulp specifically: at `4·ε` the step count falls a further 12% and the backward
        // error gets 1.8x worse, so this is the last tolerance that costs nothing.
        if f.abs() <= residual_tolerance * p.abs() {
            return x;
        }
        if f < 0.0 {
            lo = x;
            residual_lo = Some(f);
        } else {
            hi = x;
            residual_hi = Some(f);
        }
        // Stop once the bracket cannot be subdivided any further.
        if bracket_is_narrow(lo, hi, f64::EPSILON) {
            return best_of(&cdf, p, lo, hi, residual_lo, residual_hi);
        }
        x = bracketed_newton_step(x, f, pdf(x), lo, hi);
    }
    best_of(&cdf, p, lo, hi, residual_lo, residual_hi)
}

/// Wilson–Hilferty starting guess: the cube-root transform of a gamma is nearly normal, so a
/// normal quantile maps across. Used only to seed the bracket search, so its accuracy affects
/// speed rather than the answer.
fn gamma_seed(p: f64, a: f64) -> f64 {
    let guess = wilson_hilferty_guess(ndtri(p), a);
    if guess.is_finite() && guess > 0.0 {
        return guess;
    }
    // For a small shape the transform falls outside the support — at `a = 1e-3` the cube root
    // is about `−110` — so the fallback decides where the bracket search starts.
    //
    // It used to start at the mean, `a.max(1e-10)`, which is the wrong *scale* rather than
    // merely a loose guess. A small-shape gamma puts its mass at `x ≪ a`: the median at
    // `a = 1e-3` is `0.5^1000 ≈ 9.3e-302`, about 299 orders below the mean, so
    // [`expand_gamma_bracket`] had to halve its way down the whole exponent range to reach it.
    // Measured at `a = 1e-3`, that cost **755 CDF evaluations per quantile** against the two or
    // three the seed is supposed to need.
    //
    // Integrating the density near zero gives `P(a, x) ≈ x^a / Γ(1+a)` for `x ≪ 1`, so the root
    // is near `(p·Γ(1+a))^{1/a}` — which is the seed the `f32` peer in
    // [`super::f32::quantile`] has always used. Formed in log space because the value spans the
    // whole exponent range and `powf` would overflow the intermediate.
    let log_seed = (p.ln() + ln_gamma(a + 1.0)) / a;
    // `exp` saturates to zero below the subnormal floor, and a zero seed cannot be halved
    // towards anything — so the floor is the smallest value the expansion can still move.
    let seed = log_seed.exp();
    if seed.is_finite() && seed > 0.0 {
        seed
    } else {
        f64::MIN_POSITIVE
    }
}

/// Whether every interior gamma quantile rounds to its shape at this spacing.
///
/// When half the spacing is at least 64 standard deviations, the Chernoff
/// bounds for Gamma(a, 1) put both outer tails below exp(-1024), smaller than
/// the least positive f64 probability. The upper bound follows from
/// `u - ln(1+u) >= u²/(2(1+u))` at `u = 64/sqrt(a)`, where `u <= 1`.
pub(crate) fn gamma_quantiles_round_to_shape(shape: f64, spacing_below: f64) -> bool {
    spacing_below >= 128.0 * shape.sqrt()
}

/// Which tail of the gamma the quantile solves against.
///
/// Not a convenience: the two arms solve *different equations* for the same root, and the second
/// exists because `P` cannot represent its own argument near one. Both the CDF and the target
/// shift with the choice, which is why the pair travels as one named thing rather than as a
/// `bool` consulted twice — the two consultations have to agree, and nothing but their shared
/// origin makes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GammaTail {
    /// `p ≤ ½`: solve `P(a, x) = p` directly.
    Lower,
    /// `p > ½`: solve `−Q(a, x) = p − 1` instead. `P` has a **rounding plateau** near one — it is
    /// within an ulp of `1` across a wide band of `x`, so the residual there carries no
    /// information about which `x` is the root, while `Q` resolves the same band to full relative
    /// precision. Negated so the function stays increasing, which the solver requires.
    Upper,
}

impl GammaTail {
    fn for_probability(p: f64) -> Self {
        if p > 0.5 { Self::Upper } else { Self::Lower }
    }

    /// The increasing function whose root is the quantile.
    fn cdf(self, a: f64, x: f64) -> f64 {
        match self {
            Self::Lower => super::gamma::gamma_lr(a, x),
            Self::Upper => -super::gamma::gamma_ur(a, x),
        }
    }

    /// The target `p`, moved into the same frame as [`Self::cdf`].
    fn target(self, p: f64) -> f64 {
        match self {
            Self::Lower => p,
            Self::Upper => p - 1.0,
        }
    }
}

/// Which route produces the gamma quantile at this `(p, a)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GammaQuantileRoute {
    /// A shape that is not a positive finite number, or a `p` outside `[0, 1]` — `NaN` included,
    /// which the range check rejects by failing rather than by comparing.
    Undefined,
    /// `p = 0`: the infimum of the support, attained exactly.
    AtZero,
    /// `p = 1`: the supremum, which the gamma does not attain.
    AtInfinity,
    /// Every interior quantile at this shape rounds to the shape itself, so there is nothing to
    /// solve; see [`gamma_quantiles_round_to_shape`].
    RoundsToShape,
    /// An interior probability at a resolvable shape: bracket and solve.
    Solve,
}

impl GammaQuantileRoute {
    fn for_arguments(p: f64, a: f64) -> Self {
        if !a.is_finite() || a <= 0.0 || !(0.0..=1.0).contains(&p) {
            Self::Undefined
        } else if p == 0.0 {
            Self::AtZero
        } else if p == 1.0 {
            Self::AtInfinity
        } else if gamma_quantiles_round_to_shape(a, a - a.next_down()) {
            Self::RoundsToShape
        } else {
            Self::Solve
        }
    }
}

/// The gamma quantile: the `x` with `P(a, x) = p`.
pub(crate) fn gamma_lr_inv(p: f64, a: f64) -> f64 {
    match GammaQuantileRoute::for_arguments(p, a) {
        GammaQuantileRoute::Undefined => f64::NAN,
        GammaQuantileRoute::AtZero => 0.0,
        GammaQuantileRoute::AtInfinity => f64::INFINITY,
        GammaQuantileRoute::RoundsToShape => a,
        GammaQuantileRoute::Solve => gamma_quantile_solve(p, a),
    }
}

/// Bracket and solve for an interior gamma quantile at a resolvable shape.
fn gamma_quantile_solve(p: f64, a: f64) -> f64 {
    let tail = GammaTail::for_probability(p);
    let cdf = |x: f64| tail.cdf(a, x);

    // Bracket by expanding geometrically from the seed in whichever direction is needed. This
    // does not assume the seed is on either particular side of the root, which is why it
    // survives shapes where Wilson-Hilferty is poor.
    let seed = gamma_seed(p, a);
    let p = tail.target(p);
    let Some((lo, hi)) =
        deviation_bracket(&cdf, p, seed, a).or_else(|| expand_gamma_bracket(&cdf, p, seed))
    else {
        return f64::INFINITY;
    };
    // Evaluated here rather than inside the density, and after the early returns above so the
    // cases that never solve do not pay for it. See [`gamma_pdf`].
    let log_normaliser = ln_gamma(a);
    solve_increasing(
        cdf,
        |x| gamma_pdf(x, a, log_normaliser),
        p,
        lo,
        hi,
        residual_tolerance(a),
    )
}

/// Half-width of the [`deviation_bracket`], in standard deviations of the distribution.
///
/// Wide enough that the seed's own error cannot put the root outside it, and the margin is
/// enormous at the shapes where the bracket is used: Wilson–Hilferty's relative error falls like
/// `1/a`, so at `a = 1e6` the seed sits about `0.001` standard deviations from the root against
/// the eight allowed here. It does not need to be tight — being inside an *octave* is the whole
/// point — and a loose value costs at most one extra halving while a tight one risks a miss.
const DEVIATION_HALF_WIDTH: f64 = 8.0;

/// A bracket a few standard deviations around the seed, where the seed is good enough to use one.
///
/// [`expand_gamma_bracket`] moves by factors of two, so the narrowest bracket it can return is a
/// full octave. At large shape that is vastly wider than the distribution it is bracketing:
/// `Gamma(a, 1)` has standard deviation `√a`, so an octave around the seed at `a = 1e6` spans
/// about 500 standard deviations — and `P` has underflowed to *exactly zero* across nearly all
/// of it. The solver then takes a step with a zero residual and a zero density, which
/// [`bracketed_newton_step`] can only answer by halving, and it halves for the nine or so steps
/// it takes to get from octave scale down to `√a`. Traced at `a = 1e6, p = 0.3`, the first seven
/// steps all reported `f = −0.3` exactly — the CDF returning zero — before the residual moved at
/// all. Each was a full CDF evaluation, which at that shape is a series of `O(√a)` terms.
///
/// Two evaluations replace them. The bracket is **verified before it is used**, so a seed that
/// is poor — a small shape, an extreme `p`, anywhere Wilson–Hilferty degrades — simply fails the
/// check and falls back to the geometric expansion, which assumes nothing about the seed.
///
/// Returns `None` both when the bracket would not be narrower than an octave anyway (so there is
/// nothing to win and the two evaluations would be wasted) and when it fails to straddle `p`.
fn deviation_bracket(cdf: &impl Fn(f64) -> f64, p: f64, seed: f64, a: f64) -> Option<(f64, f64)> {
    let half_width = DEVIATION_HALF_WIDTH * a.sqrt();
    // Only worth two probes when the result beats what expansion gives for free. A quarter of
    // the seed keeps this strictly inside the octave `[seed/2, seed]` or `[seed, 2·seed]`.
    if !(half_width.is_finite() && half_width < 0.25 * seed) {
        return None;
    }
    let lo = (seed - half_width).max(f64::MIN_POSITIVE);
    let hi = seed + half_width;
    (cdf(lo) <= p && cdf(hi) >= p).then_some((lo, hi))
}

/// Enclose an increasing CDF target by doubling or halving the seed.
/// An exact seed remains the upper endpoint.
/// `None` means the upper endpoint left the finite range.
fn expand_gamma_bracket(cdf: &impl Fn(f64) -> f64, p: f64, seed: f64) -> Option<(f64, f64)> {
    let (mut lo, mut hi) = (seed, seed);
    if cdf(seed) < p {
        for _ in 0..BRACKET_OCTAVES {
            lo = hi;
            hi *= 2.0;
            if !hi.is_finite() {
                return None;
            }
            if cdf(hi) >= p {
                break;
            }
        }
    } else {
        for _ in 0..BRACKET_OCTAVES {
            hi = lo;
            lo *= 0.5;
            if lo == 0.0 || cdf(lo) <= p {
                break;
            }
        }
    }

    Some((lo, hi))
}

/// The beta quantile: the `x` in `[0, 1]` with `I_x(a, b) = p`.
///
/// The support is bounded, so the bracket is `[0, 1]` with no search required — this case is
/// unconditionally safe for any `a`, `b`, including the U-shaped densities (`a, b < 1`) whose
/// mass piles into both corners.
pub(crate) fn beta_reg_inv(p: f64, a: f64, b: f64) -> f64 {
    match BetaQuantileRoute::for_arguments(p, a, b) {
        BetaQuantileRoute::Undefined => f64::NAN,
        BetaQuantileRoute::AtZero => 0.0,
        BetaQuantileRoute::AtOne => 1.0,
        BetaQuantileRoute::Solve => beta_quantile_solve(p, a, b),
    }
}

/// Which route produces the beta quantile at this `(p, a, b)`.
///
/// One arm fewer than [`GammaQuantileRoute`], and the missing one is the point: the beta's
/// support is bounded, so both endpoints are attained and there is no unresolvable-shape case to
/// detect — the bracket `[0, 1]` is correct for every `a` and `b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BetaQuantileRoute {
    /// Either shape not a positive finite number, or a `p` outside `[0, 1]` — `NaN` included,
    /// which the range check rejects by failing rather than by comparing.
    Undefined,
    /// `p = 0`: the left endpoint of the support, attained exactly.
    AtZero,
    /// `p = 1`: the right endpoint, which unlike the gamma's the beta does attain.
    AtOne,
    /// An interior probability: solve on `[0, 1]`.
    Solve,
}

impl BetaQuantileRoute {
    fn for_arguments(p: f64, a: f64, b: f64) -> Self {
        if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 || !(0.0..=1.0).contains(&p) {
            Self::Undefined
        } else if p == 0.0 {
            Self::AtZero
        } else if p == 1.0 {
            Self::AtOne
        } else {
            Self::Solve
        }
    }
}

/// Solve for an interior beta quantile. The bracket is the whole support, known up front.
fn beta_quantile_solve(p: f64, a: f64, b: f64) -> f64 {
    let cdf = |x: f64| super::beta::beta_reg(a, b, x);
    let log_beta = log_beta_normaliser(a, b);
    solve_increasing(
        cdf,
        |x| beta_pdf(x, a, b, log_beta),
        p,
        0.0,
        1.0,
        residual_tolerance(a.min(b)),
    )
}

#[cfg(test)]
mod tests;
