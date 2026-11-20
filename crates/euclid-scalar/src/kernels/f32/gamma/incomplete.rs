//! The regularized incomplete gamma pair `P(a, x)` and `Q(a, x)`, at single precision.
//!
//! A child module so the parent stays inside the crate's 400-line budget; the `f64` peer at
//! [`crate::kernels::gamma::incomplete`] splits the same way. Its own topic rather than a
//! continuation of `Γ`: the pair are probabilities in *two* arguments, and the route turns on the
//! relationship between `a` and `x` rather than on the magnitude of either. Every term count and
//! crossover here is rederived at this format rather than inherited.

use super::*;

/// Iteration cap for the incomplete-gamma series and continued fraction.
///
/// Both need roughly `O(sqrt(a))` terms near `x ~ a`, so a small cap silently truncates at
/// large shape: at 200 the series stopped short from about `a = 1e3`, leaving `P(1e5, median)`
/// at `0.237` instead of `0.5` and putting a visible discontinuity at the `x = a + 1` branch
/// switch. All the series terms are positive, so this is pure truncation with no cancellation —
/// raising the cap fixes it outright, and costs nothing at small shape because both loops
/// break as soon as the relative term falls below `EPS`.
const MAX_ITER: usize = 100_000;
// Loosened from the `f64` `1e-15`: a relative tolerance below `f32::EPSILON` (~1.2e-7) can
// never be met, so the series/continued fractions would spin to `MAX_ITER`. `1e-6` sits
// just above the noise floor — full `f32` accuracy, prompt convergence.
const EPS: f32 = 1e-6;

/// At or above this shape the incomplete gamma switches to an Edgeworth-corrected normal
/// approximation; the single-precision peer of
/// [`crate::kernels::gamma::GAMMA_EDGEWORTH_MIN`], which carries the full explanation.
///
/// Four orders of magnitude lower than the `f64` threshold, because single precision loses the
/// series that much sooner. Measured against the `f64` reference, worst case over
/// `z = (x−a)/√a ∈ [−1, 1]`:
///
/// ```text
///          series    Edgeworth              series    Edgeworth
/// a = 1e2  2.0e-6    4.3e-4       a = 1e4   1.5e-3    4.2e-6
/// a = 1e3  1.0e-4    4.1e-5       a = 1e5   3.5e-2    5.0e-7
///                                 a = 1e6   3.7e-1    2.8e-9
/// ```
///
/// so `1e3` is where the approximation overtakes the series, and from there the series only
/// gets worse — reaching `NaN` by `a = 1e8`.
pub(crate) const GAMMA_EDGEWORTH_MIN: f32 = 1e3;

/// Shapes for which cancellation warrants a dedicated small-shape route.
/// Below this shape the complement needs a series that retains terms of order `a`.
///
/// Set to `0.01`, which is where the `f64` deferral this replaced used to begin — so the native
/// small-shape branch covers exactly the region that used to be widened, and the regular series
/// above it is untouched. The `f64` peer's own boundary is `1e-3`; the difference is not an
/// inheritance error but the same boundary measured against a coarser format, where the regular
/// series loses its grip on an `O(a)` term sooner.
pub(crate) const SMALL_SHAPE: f32 = 0.01;

/// `ln Γ(1+a) / a`.
///
/// No separate Taylor series, unlike the `f64` peer: [`LN_GAMMA1P_FACTORED`] already expresses
/// `lnΓ(1+a)` as `a(a−1)·g(a)`, so dividing by `a` is just `(a−1)·g(a)` — the `a` cancels
/// *symbolically* rather than arithmetically, which is better than a series would be. It removes
/// the division entirely, so the quotient cannot round, and it stays exact as `a → 0` where
/// dividing a subnormal by a subnormal would not.
fn ln_gamma1p_over_a(a: f32) -> f32 {
    let mut g = 0.0;
    for &coefficient in LN_GAMMA1P_FACTORED.iter().rev() {
        g = g * a + coefficient;
    }
    (a - 1.0) * g
}

/// Terms of the small-shape series. The `f64` peer takes 24.
///
/// The series is `Σ (−x)^k / (k!(a+k))` and runs over `x < 1 + a`, so with `a` at most `0.01`
/// the ratio between terms is below `1/k`: the `k`-th term is bounded by `1/(k!·k)`. That passes
/// under half an `f32` ulp of the leading term at `k = 9` (`1/(9!·9) = 3.1e-7`) and under a
/// quarter of one by `k = 10`, so twelve is already well past the point where adding terms
/// changes nothing — and twenty-four would just be the `f64` count copied across.
const SMALL_SHAPE_TERMS: u32 = 12;

/// DLMF 8.7.1 with the constant term separated, for `x < 1+a`.
fn small_shape_log_lower(a: f32, x: f32) -> f32 {
    let mut power = -x;
    let mut sum = power / (a + 1.0);
    for k in 2..=SMALL_SHAPE_TERMS {
        let k = k as f32;
        power *= -x / k;
        sum += power / (a + k);
    }
    let v = a * sum;
    // ln1p(v)/v rounds near one at this cutoff. Avoid dividing subnormal values.
    let log_ratio = if v.abs() <= f32::EPSILON {
        1.0
    } else {
        v.ln_1p() / v
    };
    a * (x.ln() - ln_gamma1p_over_a(a) + sum * log_ratio)
}

/// Which series or approximation evaluates the regularized incomplete gamma.
///
/// Four regimes on three thresholds, and naming them is what makes the pair below readable:
/// `gamma_lr` and `gamma_ur` share every branch, differing only in which of the two tails each
/// forms directly and which it takes as a complement. Written as nested `if`s — which is how
/// this file had it — that structure is invisible, and the small-shape case in particular read
/// as an afterthought when it is the one regime where neither ordinary route works.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IncompleteGammaMethod {
    /// Not a positive finite shape with a non-negative argument.
    Invalid,
    /// `x` is `0` or `+∞`, where both tails are exactly `0` or `1`.
    Limit,
    /// `a ≤ `[`SMALL_SHAPE`] and `x < a + 1`: the DLMF 8.7.1 series, which keeps the `O(a)`
    /// terms the ordinary routes drop. Part of the branch that used to widen to `f64`.
    SmallShapeSeries,
    /// `a ≤ `[`SMALL_SHAPE`] and `x ≥ a + 1`: the upper continued fraction with the tiny shape
    /// applied **after** it, so the probability takes a single subnormal rounding.
    ///
    /// Separate from [`Self::SmallShapeSeries`] because the series above is only valid for
    /// `x < 1 + a` — it is a Taylor expansion in `x`. Collapsing the two into one small-shape
    /// arm is a mistake that reads as harmless and is not: it put the series to work at `x = 4`,
    /// where `P + Q` came back as `1.0000009537`. Caught by
    /// `the_native_f32_small_shape_gamma_tails_are_complementary`.
    SmallShapeFraction,
    /// `a ≥ `[`GAMMA_EDGEWORTH_MIN`]: the Edgeworth-corrected normal, where both series fail.
    Edgeworth,
    /// `x < a + 1`: the lower-tail series converges fastest here.
    LowerSeries,
    /// `x ≥ a + 1`: the upper-tail continued fraction does.
    UpperFraction,
}

impl IncompleteGammaMethod {
    fn for_arguments(a: f32, x: f32) -> Self {
        if !valid_incomplete_gamma_inputs(a, x) {
            Self::Invalid
        } else if x == 0.0 || x.is_infinite() {
            Self::Limit
        } else if a <= SMALL_SHAPE {
            // The small shape still has to pick a tail: the series is an expansion in `x`.
            if use_lower_gamma_series(a, x) {
                Self::SmallShapeSeries
            } else {
                Self::SmallShapeFraction
            }
        } else if a >= GAMMA_EDGEWORTH_MIN {
            Self::Edgeworth
        } else if use_lower_gamma_series(a, x) {
            Self::LowerSeries
        } else {
            Self::UpperFraction
        }
    }
}

/// Regularized **lower** incomplete gamma `P(a, x) = γ(a, x) / Γ(a)`, `a > 0`, `x ≥ 0`.
pub(crate) fn gamma_lr(a: f32, x: f32) -> f32 {
    match IncompleteGammaMethod::for_arguments(a, x) {
        IncompleteGammaMethod::Invalid => f32::NAN,
        // `P(a, 0) = 0` and `P(a, ∞) = 1`. Both are needed explicitly: without them the series
        // below return `NaN` at an infinite argument, which propagates into `erf` and the whole
        // normal family built on it.
        IncompleteGammaMethod::Limit => f32::from(u8::from(x.is_infinite())),
        IncompleteGammaMethod::SmallShapeSeries => small_shape_log_lower(a, x).exp(),
        // Past the series' radius the lower tail is the complement of a small upper tail, which
        // is the orientation that keeps its digits.
        IncompleteGammaMethod::SmallShapeFraction => 1.0 - gamma_ur(a, x),
        IncompleteGammaMethod::Edgeworth => gamma_edgeworth(a, x).0,
        IncompleteGammaMethod::LowerSeries => {
            let series = gamma_lower_series(a, x, EPS, MAX_ITER);
            log_scaled_gamma(a, x, series.ln()).exp()
        }
        IncompleteGammaMethod::UpperFraction => 1.0 - gamma_ur(a, x),
    }
}

/// Regularized **upper** incomplete gamma `Q(a, x) = Γ(a, x) / Γ(a) = 1 - P(a, x)`.
pub(crate) fn gamma_ur(a: f32, x: f32) -> f32 {
    match IncompleteGammaMethod::for_arguments(a, x) {
        IncompleteGammaMethod::Invalid => f32::NAN,
        IncompleteGammaMethod::Limit => f32::from(u8::from(!x.is_infinite())),
        // `exp_m1` rather than `1 - exp`: the small-shape lower tail is close to one here, so
        // the complement would cancel away exactly the digits the branch exists to keep.
        IncompleteGammaMethod::SmallShapeSeries => -small_shape_log_lower(a, x).exp_m1(),
        IncompleteGammaMethod::SmallShapeFraction => {
            let h = gamma_upper_fraction(a, x, LENTZ_TINY_F32, EPS, MAX_ITER);
            // Γ(1+a) = a Γ(a). Multiply the tiny shape only after evaluating the fraction so the
            // probability undergoes a single subnormal rounding.
            a * (-x + a * (x.ln() - ln_gamma1p_over_a(a)) + h.ln()).exp()
        }
        IncompleteGammaMethod::Edgeworth => gamma_edgeworth(a, x).1,
        IncompleteGammaMethod::LowerSeries => 1.0 - gamma_lr(a, x),
        IncompleteGammaMethod::UpperFraction => {
            log_scaled_gamma(a, x, 0.0).exp()
                * gamma_upper_fraction(a, x, LENTZ_TINY_F32, EPS, MAX_ITER)
        }
    }
}

#[cfg(test)]
mod tests;
