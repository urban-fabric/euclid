//! Single-precision peers of [`super::super::quantile`], with a shared `f32`
//! Newton/bisection loop and format-specific stopping rule.

use super::gamma::ln_gamma;
use super::normal::ndtri;
use crate::math::density::{beta_pdf, gamma_pdf, log_beta_normaliser};
use crate::math::roots::{bracket_is_narrow, bracketed_newton_step, wilson_hilferty_guess};

/// Iteration cap. Bisection alone halves the bracket each step, so this many steps is far more
/// than enough to reach the representable limit from any starting bracket.
const MAX_STEPS: u32 = 200;

/// Wilson–Hilferty starting guess: the cube-root transform of a gamma is nearly normal, so a
/// normal quantile maps across. Good to a few digits over most of the parameter range.
fn gamma_start(p: f32, a: f32) -> f32 {
    let guess = wilson_hilferty_guess(ndtri(p), a);
    if guess.is_finite() && guess > 0.0 {
        guess
    } else {
        // The transform degrades for very small shape; fall back to the small-x form
        // P(a,x) ~ x^a / Gamma(a+1).
        //
        // The floor is `f32::MIN_POSITIVE`, not the `1e-300` this line used to carry. That
        // literal came over from an `f64` context and **underflows to zero in `f32`**, so
        // `.max(1e-300)` was `.max(0.0)` — a no-op for a value already non-negative, and a floor
        // that floored nothing. Harmless in practice, because the caller applies
        // `.max(f32::MIN_POSITIVE)` to the result anyway, but a constant that silently means
        // something else in the narrower type is exactly the hazard this tree keeps producing.
        (p * (ln_gamma(a + 1.0)).exp())
            .powf(1.0 / a)
            .max(f32::MIN_POSITIVE)
    }
}

/// Whether every interior quantile of `Gamma(a, 1)` rounds to the shape at `f32`'s spacing.
///
/// Native, where this used to widen both arguments and call the `f64` predicate. The inequality
/// is pure arithmetic — there was never anything for the wider format to add — but the *spacing*
/// it tests is `f32`'s, so widening it was doubly beside the point: the quantity being compared
/// only exists in this format.
///
/// When half the spacing below `a` is at least 64 standard deviations, the Chernoff bounds for
/// `Gamma(a, 1)` put both outer tails below `exp(−1024)`, under the least positive `f32`
/// probability, so no interior `p` can resolve away from the mean. `√a` is the standard
/// deviation; the factor of 128 is 64 doubled to compare against the full spacing rather than
/// its half.
fn quantiles_round_to_shape(a: f32) -> bool {
    (a - a.next_down()) >= 128.0 * a.sqrt()
}

/// The gamma quantile: the `x` with `P(a, x) = p`.
///
/// # Native, and what that costs at a small shape
///
/// This used to widen to `f64` for `a ≤ 0.01`, with the note that "rounding the CDF before
/// inversion amplifies its error at small shapes". That note was exactly right, and the
/// amplification is worth stating as a number because it is a property of the *problem* rather
/// than of any implementation.
///
/// Near zero `P(a, x) ≈ x^a / Γ(1+a)`, so `∂P/∂x = aP/x` and therefore
///
/// ```text
/// δx/x = (δP/P) / a
/// ```
///
/// The inversion's condition number is `1/a`. At `a = 0.01` that is a hundredfold, so a CDF good
/// to one `f32` ulp yields a root good to about a hundred — and no choice of solver, bracket or
/// stopping rule changes it, because the forward function is genuinely that flat: `P` moves
/// through its whole range only as `x` crosses `e^{1/a}`. Solving for `ln x` instead does not
/// help either; `∂P/∂(ln x) = aP` carries the same factor.
///
/// Measured against the `f64` kernel rounded to `f32`, worst over `p ∈ (0, 1)`:
///
/// | `a` | `1e-3` | `0.01` | `0.05` | `0.5` | `1` | `10` | `100` |
/// |---|---|---|---|---|---|---|---|
/// | ulp | `111` | `115` | `59` | `9` | `8` | `12` | `50` |
///
/// which tracks `1/a` where that is large, and the CDF's own widening error at the other end.
/// So the `f64` route was not better by virtue of being better code — it had 29 more bits to
/// spend on a hundredfold amplification, and `f32` does not. A consumer needing a small-shape
/// gamma quantile to `f32`'s full resolution needs `f64`; this function cannot supply it, and
/// says so rather than quietly borrowing the wider format to look as though it can.
pub(crate) fn gamma_lr_inv(p: f32, a: f32) -> f32 {
    match GammaQuantileRoute::for_arguments(p, a) {
        GammaQuantileRoute::Undefined => f32::NAN,
        GammaQuantileRoute::AtZero => 0.0,
        GammaQuantileRoute::AtInfinity => f32::INFINITY,
        GammaQuantileRoute::RoundsToShape => a,
        GammaQuantileRoute::Solve => gamma_quantile_solve(p, a),
    }
}

/// Which tail of the gamma the quantile solves against.
///
/// The same two arms the `f64` peer in [`crate::kernels::quantile`] carries, and for the same
/// reason: they solve *different equations* for the same root, and both the CDF and the target
/// shift with the choice. The pair travels as one named thing because the two uses have to agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GammaTail {
    /// `p ≤ ½`: solve `P(a, x) = p` directly.
    Lower,
    /// `p > ½`: solve `−Q(a, x) = p − 1` instead. `P` has a rounding plateau near one — wider
    /// here than in `f64`, with 24 bits of mantissa rather than 53 — where the residual carries
    /// no information about which `x` is the root. Negated so the function stays increasing.
    Upper,
}

impl GammaTail {
    fn for_probability(p: f32) -> Self {
        if p > 0.5 { Self::Upper } else { Self::Lower }
    }

    /// The increasing function whose root is the quantile.
    fn cdf(self, a: f32, x: f32) -> f32 {
        match self {
            Self::Lower => super::gamma::gamma_lr(a, x),
            Self::Upper => -super::gamma::gamma_ur(a, x),
        }
    }

    /// The target `p`, moved into the same frame as [`Self::cdf`].
    fn target(self, p: f32) -> f32 {
        match self {
            Self::Lower => p,
            Self::Upper => p - 1.0,
        }
    }
}

/// Which route produces the gamma quantile at this `(p, a)`.
///
/// The same five arms as the `f64` peer; [`quantiles_round_to_shape`] differs in taking the shape
/// alone, because this format's spacing is derived rather than passed in.
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
    /// solve; see [`quantiles_round_to_shape`].
    RoundsToShape,
    /// An interior probability at a resolvable shape: bracket and solve.
    Solve,
}

impl GammaQuantileRoute {
    fn for_arguments(p: f32, a: f32) -> Self {
        if !a.is_finite() || a <= 0.0 || !(0.0..=1.0).contains(&p) {
            Self::Undefined
        } else if p == 0.0 {
            Self::AtZero
        } else if p == 1.0 {
            Self::AtInfinity
        } else if quantiles_round_to_shape(a) {
            Self::RoundsToShape
        } else {
            Self::Solve
        }
    }
}

/// Bracket and solve for an interior gamma quantile at a resolvable shape.
fn gamma_quantile_solve(p: f32, a: f32) -> f32 {
    let tail = GammaTail::for_probability(p);
    let cdf = |x: f32| tail.cdf(a, x);

    // Bracket: walk out from the guess until the CDF straddles p.
    let seed = gamma_start(p, a);
    let p = tail.target(p);
    let mut lo = 0.0f32;
    let mut hi = seed.max(f32::MIN_POSITIVE);
    let mut steps = 0;
    while cdf(hi) < p && steps < MAX_STEPS {
        lo = hi;
        hi *= 2.0;
        steps += 1;
    }
    if hi.is_infinite() {
        return f32::INFINITY;
    }
    // Every nonzero lower endpoint was accepted by `cdf(hi) < p` above.
    let log_normaliser = ln_gamma(a);
    solve_increasing(
        cdf,
        |x| gamma_pdf(x, a, log_normaliser),
        p,
        lo,
        hi,
        seed.clamp(lo, hi),
    )
}

/// Safeguarded Newton steps share the bracket-relative stopping rule for both distributions.
fn solve_increasing(
    cdf: impl Fn(f32) -> f32,
    pdf: impl Fn(f32) -> f32,
    p: f32,
    mut lo: f32,
    mut hi: f32,
    mut x: f32,
) -> f32 {
    for _ in 0..MAX_STEPS {
        let f = cdf(x) - p;
        if f == 0.0 {
            return x;
        }
        // Keep the bracket tight around the root.
        if f < 0.0 {
            lo = x;
        } else {
            hi = x;
        }
        let next = bracketed_newton_step(x, f, pdf(x), lo, hi);
        // The relative check includes adjacent endpoints, where bisection rounds to an edge.
        if bracket_is_narrow(lo, hi, 4.0 * f32::EPSILON) {
            return next;
        }
        x = next;
    }
    x
}

/// The beta quantile: the `x` in `[0, 1]` with `I_x(a, b) = p`.
pub(crate) fn beta_reg_inv(p: f32, a: f32, b: f32) -> f32 {
    match BetaQuantileRoute::for_arguments(p, a, b) {
        BetaQuantileRoute::Undefined => f32::NAN,
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
    fn for_arguments(p: f32, a: f32, b: f32) -> Self {
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
fn beta_quantile_solve(p: f32, a: f32, b: f32) -> f32 {
    // Note the internal argument order: (a, b, x), not (x, a, b).
    let cdf = |x: f32| super::beta::beta_reg(a, b, x);

    // The support is bounded, so the bracket is known up front — no expansion needed. That
    // makes this case unconditionally safe: bisection alone would converge.
    let log_beta = log_beta_normaliser(a, b);
    solve_increasing(
        cdf,
        |x| beta_pdf(x, a, b, log_beta),
        p,
        0.0,
        1.0,
        a / (a + b),
    )
}

#[cfg(test)]
mod tests;
