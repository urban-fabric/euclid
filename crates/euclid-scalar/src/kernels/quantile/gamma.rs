//! The gamma quantile: the `x` with `P(a, x) = p`.
//!
//! A child module so the parent stays inside the crate's 400-line budget. The two quantiles
//! share the solver in [`super`] and nothing else: the gamma's support is unbounded, so it needs
//! a seed, a bracket search and a shape at which every interior quantile stops being
//! resolvable — none of which the beta has.

use super::*;

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
            Self::Lower => crate::kernels::gamma::gamma_lr(a, x),
            Self::Upper => -crate::kernels::gamma::gamma_ur(a, x),
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

#[cfg(test)]
mod tests;
