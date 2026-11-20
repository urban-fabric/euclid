//! The exponential integrals `E₁` and `Ei`.
//!
//! A child module so the parent stays inside the crate's 400-line budget. Both are the same
//! function on different rays — `Ei(−x) = −E₁(x)` — so they share the continued fraction in
//! [`super`] and the series routes here. The trigonometric pair sit in
//! [`super::trigonometric`], being the *imaginary* ray of the same fraction.

use super::*;

/// Which route evaluates `E₁`.
///
/// Shaped like its sibling [`EiRoute`], and routed through `partial_cmp` for the same reason:
/// ordering against zero names the `NaN`, the negative half-line and the two positive regimes in
/// one exhaustive match, where the equivalent `x.is_nan() || x < 0.0` guard has an operator a
/// mutant can weaken. That particular mutant is *equivalent* rather than live — a negative `x`
/// reaches `ln x` in the series and comes back `NaN` anyway — but an exhaustive match has nothing
/// to weaken in the first place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum E1Route {
    /// A value the function takes exactly: the pole at the origin, the limit at `+∞`, and the
    /// `NaN` that states the domain.
    Value(f64),
    /// The power series below [`E1_SERIES_MAX`], which cancels `ln x` against a growing sum.
    Series,
    /// The continued fraction above it: the stable route in the tail, where `E₁` decays like
    /// `e^{−x}/x` and the series' cancellation would dominate.
    ContinuedFraction,
}

pub(super) fn e1_route(x: f64) -> E1Route {
    use core::cmp::Ordering;

    match x.partial_cmp(&0.0) {
        // The real `E₁` is defined for a positive argument.
        None | Some(Ordering::Less) => E1Route::Value(f64::NAN),
        Some(Ordering::Equal) => E1Route::Value(f64::INFINITY),
        Some(Ordering::Greater) if x.is_infinite() => E1Route::Value(0.0),
        Some(Ordering::Greater) if x <= E1_SERIES_MAX => E1Route::Series,
        Some(Ordering::Greater) => E1Route::ContinuedFraction,
    }
}

/// `E₁(x) = −γ − ln x + Σ_{k≥1} (−1)^{k+1} x^k/(k·k!)`, for small positive `x`.
pub(super) fn e1_series(x: f64) -> f64 {
    let mut power_over_factorial = 1.0f64; // x^k/k!, carried across terms
    let series = compensated_series(|k| {
        let k1 = f64::from(k) + 1.0;
        power_over_factorial *= x / k1;
        let term = power_over_factorial / k1; // x^k/(k k!)
        Some(if k.is_multiple_of(2) { term } else { -term })
    });
    -EULER_GAMMA - x.ln() + series
}

/// `E₁(x) = ∫ₓ^∞ e^{−t}/t dt`, for `x > 0`.
///
/// Series below `x = 1`, continued fraction above it; see [`E1Route`].
pub(crate) fn exp_int_e1(x: f64) -> f64 {
    match e1_route(x) {
        E1Route::Value(value) => value,
        E1Route::Series => e1_series(x),
        // On the real axis the imaginary part is exactly zero throughout, so nothing is lost by
        // sharing the complex routine.
        E1Route::ContinuedFraction => e1_continued_fraction((x, 0.0), ((-x).exp(), 0.0)).0,
    }
}

/// `Ei(x) = −PV ∫_{−x}^∞ e^{−t}/t dt`, the exponential integral.
///
/// For positive argument every term of the series is positive, so there is no cancellation and
/// the series is preferred as far as it converges cheaply; the asymptotic expansion takes over
/// where it is comfortably accurate.
pub(crate) fn exp_int_ei(x: f64) -> f64 {
    match ei_route(x) {
        EiRoute::Value(value) => value,
        EiRoute::Reflection => -exp_int_e1(-x),
        EiRoute::Series => ei_series(x),
        EiRoute::Asymptotic => ei_asymptotic(x),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum EiRoute {
    Value(f64),
    Reflection,
    Series,
    Asymptotic,
}

pub(super) fn ei_route(x: f64) -> EiRoute {
    use core::cmp::Ordering;

    match x.partial_cmp(&0.0) {
        None => EiRoute::Value(f64::NAN),
        Some(Ordering::Equal) => EiRoute::Value(f64::NEG_INFINITY),
        Some(Ordering::Less) => EiRoute::Reflection,
        Some(Ordering::Greater) if x.is_infinite() => EiRoute::Value(f64::INFINITY),
        Some(Ordering::Greater) if x < EI_ASYMPTOTIC_MIN => EiRoute::Series,
        Some(Ordering::Greater) => EiRoute::Asymptotic,
    }
}

/// `Ei(x) = gamma + ln x + sum_{k≥1} x^k / (k k!)` for small positive x.
pub(super) fn ei_series(x: f64) -> f64 {
    let mut power_over_factorial = 1.0f64;
    let series = compensated_series(|k| {
        let k1 = f64::from(k) + 1.0;
        power_over_factorial *= x / k1;
        Some(power_over_factorial / k1)
    });
    EULER_GAMMA + x.ln() + series
}

pub(super) fn ei_asymptotic(x: f64) -> f64 {
    // Split e^x into two factors so an intermediate exponential cannot overflow
    // while the final value is still representable.
    let exp_half = (0.5 * x).exp();
    exp_half * ((exp_half / x) * ei_asymptotic_sum(x))
}

/// Sum `k!/x^k` through the least term for the asymptotic expansion of `Ei(x)`.
pub(super) fn ei_asymptotic_sum(x: f64) -> f64 {
    let mut term = 1.0f64;
    let mut total = 1.0f64;
    for k in 1..MAX_TERMS {
        let prev = term;
        term *= f64::from(k) / x;
        if term > prev {
            break; // past the minimum term; the asymptotic series diverges beyond it
        }
        total += term;
        if term_is_negligible(term, total, 1e-18) {
            break;
        }
    }
    total
}

#[cfg(test)]
mod tests;
