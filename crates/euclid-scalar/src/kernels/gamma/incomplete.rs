//! The regularized incomplete gamma pair `P(a, x)` and `Q(a, x)`.
//!
//! A child module so the parent stays inside the crate's 400-line budget. This is its own topic
//! rather than a continuation of `Γ`: the pair are *probabilities* in two arguments, and which
//! route evaluates them turns on the relationship between `a` and `x` rather than on the
//! magnitude of either — see [`IncompleteGammaMethod`]. The small-shape and Edgeworth regimes
//! exist only here.

use super::*;

fn small_shape_log_lower(a: f64, x: f64) -> f64 {
    let mut power = -x;
    let mut sum = power / (a + 1.0);
    for k in 2..=24 {
        let k = f64::from(k);
        power *= -x / k;
        sum += power / (a + k);
    }
    let v = a * sum;
    // ln1p(v)/v rounds near one at this cutoff. Avoid dividing subnormal values.
    let log_ratio = if v.abs() <= f64::EPSILON {
        1.0
    } else {
        v.ln_1p() / v
    };
    a * (x.ln() - ln_gamma1p_over_a(a) + sum * log_ratio)
}

/// At or above this shape the incomplete gamma switches to an Edgeworth-corrected normal
/// approximation.
///
/// Both of the series routes fail above it, and they fail *quietly*. Measured `P(a, a)`
/// against its known asymptote `½ + 1/(3√(2πa))`:
///
/// ```text
/// a = 1e7   2.9e-8        a = 1e9.5  7.5e-2
/// a = 1e8   3.4e-7        a = 1e10   3.2e-1
/// a = 1e9   1.6e-3        a = 1e11   7.5e-1
/// ```
///
/// and further out it stops being a probability at all: `P(1e15, a + √a)` came back as
/// `−9.06`, and from `a = 1e16` as `NaN`. Two mechanisms compound — the series needs `O(√a)`
/// terms and runs into [`MAX_ITER`], and the front factor `exp(a ln x − x − lnΓ(a))` subtracts
/// three `O(a ln a)` quantities, which past `a ≈ 1e14` leaves nothing at all.
///
/// A `Gamma(a, 1)` standardised by its own mean and variance has skewness `2/√a`, so the
/// Edgeworth expansion
///
/// ```text
/// P(a, x) ≈ Φ(z) − φ(z)·(z² − 1)/(3√a),      z = (x − a)/√a
/// ```
///
/// has absolute error `O(1/a)` — *improving* exactly where the series collapses, and exact in
/// the limit. Verified against the series over `a ∈ [1e6, 1e8]` and `z ∈ [−3, 6]`, where the
/// measured discrepancy is `1.8e-5`, `1.8e-6`, `3.4e-7` — one order per decade of `a`, as
/// `O(1/a)` predicts. At the threshold both routes carry about `3e-8`, so the join costs
/// nothing; above it the approximation is the more accurate of the two.
///
/// No tail accuracy is given up. `Q` decays like `e^{−a·D(x/a)}`, which underflows for any
/// `a` past a few thousand once `x` is away from `a` — `Q(1e6, 2e6)` is already exactly zero.
pub(crate) const GAMMA_EDGEWORTH_MIN: f64 = 3e7;

/// Regularized **lower** incomplete gamma `P(a, x) = γ(a, x) / Γ(a)`, `a > 0`, `x ≥ 0`.
/// Which series or approximation evaluates the regularized incomplete gamma.
///
/// Both tails route identically and differ only in what each arm returns, so the decision is
/// named once here and [`gamma_lr`] and [`gamma_ur`] are each nothing but a `match` over it. The
/// single-precision peer carries the same enum with the same arms; the ordering below is this
/// kernel's own, and the two small-shape bounds are mutually exclusive with the Edgeworth one, so
/// placing Edgeworth first changes nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IncompleteGammaMethod {
    /// Not a positive finite shape with a non-negative argument.
    Invalid,
    /// `x = 0`, where the lower tail is `0` and the upper is `1`.
    AtZero,
    /// `x = +∞`, the complementary limit.
    ///
    /// Tested for rather than left to the series: without it the routes below produce `NaN` at
    /// an infinite argument, which then propagates into everything built on `gamma_lr` — `erf`,
    /// and through it the whole normal family.
    AtInfinity,
    /// `a ≥ `[`GAMMA_EDGEWORTH_MIN`]: the Edgeworth-corrected normal, where both series fail.
    Edgeworth,
    /// `a ≥ 20` and `|x − a| ≤ 0.4a`: Temme's uniform asymptotic expansion.
    ///
    /// Placed after [`Self::Edgeworth`] so that arm keeps the region it was measured on, and
    /// before the series arms because it supersedes them wherever it applies. It is the only
    /// route here whose cost does not grow with the shape — see [`super::temme`] for why that
    /// is the same fact as its accuracy.
    Temme,
    /// `a ≤ `[`SMALL_SHAPE`] and `x < a + 1`: the DLMF 8.7.1 series, which keeps the `O(a)` terms
    /// the ordinary routes drop.
    SmallShapeSeries,
    /// `a ≤ SMALL_SHAPE` and `x ≥ a + 1`: the upper fraction with the tiny shape applied
    /// **after** it, so the probability takes a single subnormal rounding.
    ///
    /// Separate from [`Self::SmallShapeSeries`] because that series is an expansion in `x` and is
    /// only valid for `x < 1 + a`. Collapsing the two into one small-shape arm reads as harmless
    /// and is not: in the `f32` peer it put the series to work at `x = 4`, where `P + Q` came
    /// back as `1.0000009537`.
    SmallShapeFraction,
    /// `x < a + 1`: the lower-tail series converges fastest here.
    LowerSeries,
    /// `x ≥ a + 1`: the upper-tail continued fraction does.
    UpperFraction,
}

impl IncompleteGammaMethod {
    /// The invalid and limiting arguments first, then the shape, then which tail converges.
    fn for_arguments(a: f64, x: f64) -> Self {
        if !valid_incomplete_gamma_inputs(a, x) {
            Self::Invalid
        } else if x == 0.0 {
            Self::AtZero
        } else if x.is_infinite() {
            Self::AtInfinity
        } else if a >= GAMMA_EDGEWORTH_MIN {
            Self::Edgeworth
        } else if temme::use_temme(a, x) {
            Self::Temme
        } else if use_lower_gamma_series(a, x) {
            // The small shape still has to pick a tail: the series is an expansion in `x`.
            if a <= SMALL_SHAPE {
                Self::SmallShapeSeries
            } else {
                Self::LowerSeries
            }
        } else if a <= SMALL_SHAPE {
            Self::SmallShapeFraction
        } else {
            Self::UpperFraction
        }
    }
}

pub(crate) fn gamma_lr(a: f64, x: f64) -> f64 {
    match IncompleteGammaMethod::for_arguments(a, x) {
        IncompleteGammaMethod::Invalid => f64::NAN,
        IncompleteGammaMethod::AtZero => 0.0,
        IncompleteGammaMethod::AtInfinity => 1.0,
        IncompleteGammaMethod::Edgeworth => {
            let (lower, _) = gamma_edgeworth(a, x);
            lower
        }
        IncompleteGammaMethod::Temme => temme::temme_lower(a, x),
        IncompleteGammaMethod::SmallShapeSeries => small_shape_log_lower(a, x).exp(),
        IncompleteGammaMethod::LowerSeries => {
            let series = gamma_lower_series(a, x, EPS, MAX_ITER);
            log_scaled_gamma(a, x, series.ln()).exp()
        }
        // Either upper arm is reached through the complement, which is what keeps the lower tail
        // from being formed by a cancelling subtraction where it is the small one.
        IncompleteGammaMethod::SmallShapeFraction | IncompleteGammaMethod::UpperFraction => {
            1.0 - gamma_ur(a, x)
        }
    }
}

pub(crate) fn gamma_ur(a: f64, x: f64) -> f64 {
    match IncompleteGammaMethod::for_arguments(a, x) {
        IncompleteGammaMethod::Invalid => f64::NAN,
        IncompleteGammaMethod::AtZero => 1.0,
        IncompleteGammaMethod::AtInfinity => 0.0,
        IncompleteGammaMethod::Edgeworth => {
            let (_, upper) = gamma_edgeworth(a, x);
            upper
        }
        IncompleteGammaMethod::Temme => temme::temme_upper(a, x),
        IncompleteGammaMethod::SmallShapeSeries => -small_shape_log_lower(a, x).exp_m1(),
        IncompleteGammaMethod::LowerSeries => 1.0 - gamma_lr(a, x),
        IncompleteGammaMethod::SmallShapeFraction => {
            let h = gamma_upper_fraction(a, x, LENTZ_TINY_F64, EPS, MAX_ITER);
            // Γ(1+a) = a Γ(a). Multiply the tiny shape only after evaluating the fraction so the
            // probability undergoes a single subnormal rounding.
            a * (-x + a * (x.ln() - ln_gamma1p_over_a(a)) + h.ln()).exp()
        }
        IncompleteGammaMethod::UpperFraction => {
            let h = gamma_upper_fraction(a, x, LENTZ_TINY_F64, EPS, MAX_ITER);
            log_scaled_gamma(a, x, 0.0).exp() * h
        }
    }
}

#[cfg(test)]
mod tests;
