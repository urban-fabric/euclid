//! The standard normal distribution: CDF, log-CDF, and quantile.
//!
//! These are the functions every probabilistic consumer reaches for, and the ones most easily
//! got wrong. All three are built on the error function rather than on a separate
//! approximation, so they inherit its accuracy and cannot drift away from it.
//!
//! Far in the lower tail, the probability can underflow while its logarithm remains
//! representable. The scaled complementary error function keeps the log-CDF finite
//! across that part of the range.

use crate::math::normal::{self, refine_lower, tail_start_from_log};

const LN_2: f64 = core::f64::consts::LN_2;
const FRAC_1_SQRT_2: f64 = core::f64::consts::FRAC_1_SQRT_2;

/// The standard normal CDF `Phi(x) = P(Z <= x)`.
///
/// `Phi(x) = erfc(-x/sqrt 2) / 2`. Using `erfc` rather than `1 - erf` keeps the left tail
/// accurate: it is the tail where the answer is small, so the complement form would cancel.
pub(crate) fn ndtr(x: f64) -> f64 {
    normal::ndtr(x)
}

/// `ln Phi(x)`, finite as far into the left tail as the exponent range allows.
///
/// Two regimes. For `x` above about `-1`, `Phi(x)` is comfortably representable and the direct
/// `ln(erfc(z)/2)` is accurate. Below that, `erfc(z)` underflows long before `ln Phi` stops
/// being representable, so the identity `erfc(z) = erfcx(z) e^{-z^2}` is used to move the
/// vanishing factor into the log:
///
/// ```text
/// ln Phi(x) = ln erfcx(z) - z^2 - ln 2,   z = -x / sqrt 2
/// ```
///
/// At `x = -100` this returns about `-5005.5`; the naive form returns `-inf` from `x = -38`.
///
/// # The upper tail needs its own branch too
///
/// The mirror problem sits on the right, and is easy to miss because nothing overflows. As
/// `x → +∞`, `Phi(x) → 1` and `ln Phi(x) → 0` from below, so the answer is a *small negative
/// number* — but `Phi(x)` itself rounds to exactly `1.0` once `x` passes about `8.3`, and then
/// `ln(1.0)` is `0`. The value is still accurate in absolute terms (it is wrong by `8e-17`) and
/// useless in relative terms, which is the sense a log-likelihood cares about.
///
/// `ln Phi(x) = ln(1 - Phi(-x))` fixes it: the complement is the *small* tail, so `ln1p` of it
/// keeps full relative accuracy. `log_ndtr(8.25)` is `-7.98e-17` rather than `0`.
pub(crate) fn log_ndtr(x: f64) -> f64 {
    normal::log_ndtr(x)
}

/// The standard normal quantile (probit) `Phi^{-1}(p)`.
///
/// Solved rather than approximated: an initial guess is refined by Newton's method on
/// `ln Phi(x) = ln p`, whose derivative `phi(x)/Phi(x)` is available in closed form. Working in
/// the log domain is what makes the tail work — in probability space both the residual and the
/// derivative underflow, while their logs stay finite.
///
/// The guess comes from `erf_inv` in the central region and from [`tail_start_from_log`] below about
/// `p = 1e-15`, where `2p - 1` has rounded to exactly `-1` and `erf_inv` can only return
/// `-inf`.
///
/// # Accuracy
/// Full double precision for `p` in roughly `[1e-300, 0.5]`, and the same by symmetry above
/// `0.5` *as far as `1 - p` is representable*. For an extreme upper quantile, ask for
/// `-ndtri(q)` with the small `q` directly rather than `ndtri(1 - q)`; the subtraction is what
/// loses the digits, not the solver.
pub(crate) fn ndtri(p: f64) -> f64 {
    normal::ndtri(p, 1e-15)
}

/// The inverse complementary error function: the `x` with `erfc(x) = y`.
///
/// Derived from the quantile rather than from `erf_inv(1 - y)`, which cancels for small `y`.
/// Since `erfc(x) = 2 Phi(-x sqrt 2)`, the inverse is `-ndtri(y/2) / sqrt 2`, which stays
/// accurate as far into the tail as [`ndtri`] does.
/// Which route inverts `erfc` at this probability.
///
/// `erfc` maps the whole line onto `(0, 2)`, so the two ends of that interval are limits rather
/// than values and everything outside it is off the domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ErfcInvRoute {
    /// Outside `[0, 2]`. `NaN` included without a separate test: `contains` is already false for
    /// it, so an `is_nan` arm here would be dead.
    Undefined,
    /// `y = 0`, the infimum of `erfc`'s range, approached as `x → +∞`.
    AtZero,
    /// `y = 2`, the supremum, approached as `x → −∞`.
    AtTwo,
    /// An interior probability: invert through the normal quantile, in whichever coordinate the
    /// halving survives. See [`HalvingCoordinate`].
    Interior,
}

impl ErfcInvRoute {
    fn for_probability(y: f64) -> Self {
        if !(0.0..=2.0).contains(&y) {
            Self::Undefined
        } else if y == 0.0 {
            Self::AtZero
        } else if y == 2.0 {
            Self::AtTwo
        } else {
            Self::Interior
        }
    }
}

/// Which coordinate the half-probability `y/2` is formed in.
///
/// `erfc(x) = 2Φ(−x√2)`, so the inverse wants `Φ⁻¹(y/2)`. Halving is exact for every normal `y`
/// — the exponent simply drops by one — but **not** once `y` is subnormal, where the mantissa has
/// no bit left to shift into and `y/2` rounds, possibly to zero. That is a threshold on the
/// *representation* rather than on the magnitude of the answer, so it is tested for directly as
/// `half + half == y` rather than compared against a constant.
///
/// The same enum the `f32` peer in [`super::f32::normal`] carries, where the band matters more:
/// subnormals start at `1.18e-38` there against `2.2e-308` here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HalvingCoordinate {
    /// `y/2` is exact. Invert it directly through the quantile.
    Exact,
    /// `y/2` would round. Form `ln(y/2) = ln y − ln 2` instead and refine in the log domain,
    /// where the probability stays representable far past where it underflows.
    Logarithmic,
}

impl HalvingCoordinate {
    fn for_probability(y: f64, half: f64) -> Self {
        if half + half == y {
            Self::Exact
        } else {
            Self::Logarithmic
        }
    }
}

pub(crate) fn erfc_inv(y: f64) -> f64 {
    match ErfcInvRoute::for_probability(y) {
        ErfcInvRoute::Undefined => f64::NAN,
        ErfcInvRoute::AtZero => f64::INFINITY,
        ErfcInvRoute::AtTwo => f64::NEG_INFINITY,
        ErfcInvRoute::Interior => {
            let half = 0.5 * y;
            match HalvingCoordinate::for_probability(y, half) {
                HalvingCoordinate::Exact => -ndtri(half) * FRAC_1_SQRT_2,
                HalvingCoordinate::Logarithmic => {
                    let ln_half = y.ln() - LN_2;
                    -refine_lower(ln_half, tail_start_from_log(ln_half)) * FRAC_1_SQRT_2
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
