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
/// unresolvable regime is already handled by [`gamma::gamma_quantiles_round_to_shape`].
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

mod beta;
mod gamma;

// Re-exported so the family keeps one path per function: every caller reached
// `kernels::quantile::gamma_lr_inv` before the split and still does.
pub(crate) use beta::beta_reg_inv;
pub(crate) use gamma::gamma_lr_inv;

#[cfg(test)]
mod tests;
