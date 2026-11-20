//! The regime tables, and the four-at-once evaluation they route to.
//!
//! A child module so the parent stays inside the crate's 400-line budget; the `f64` peer at
//! [`crate::kernels::airy::regimes`] splits the same way. What lives here is the *routing*
//! rather than any one solution: [`AiryRegime`] divides the line and `quad` is the shared
//! evaluation both public pairs are carved out of.

use super::*;

/// `(Ai, Ai′, Bi, Bi′)` on the negative axis beyond [`MARCH_MAX`], from the Hankel expansion.
pub(super) fn negative_asymptotic(ax: f32) -> (f32, f32, f32, f32) {
    let z = zeta(ax);
    if !z.is_finite() {
        // Past `|x| ≈ 1.4e25` the phase is not merely inaccurate but unrepresentable: `ζ` itself
        // has overflowed, so there is no argument to hand to `sin`. Zero is finite, inside the
        // envelope, and the mean of the oscillation — the honest answer once the phase is gone.
        // Accuracy departs far earlier, around `|x| ≈ 1e5`, where `ulp(ζ)` passes `2π`.
        return (0.0, 0.0, 0.0, 0.0);
    }
    let j_third = hankel(1.0 / 3.0, z).first_kind;
    let j_neg_third = hankel(-1.0 / 3.0, z).first_kind;
    let j_two_thirds = hankel(2.0 / 3.0, z).first_kind;
    let j_neg_two_thirds = hankel(-2.0 / 3.0, z).first_kind;
    (
        (ax.sqrt() / 3.0) * (j_third + j_neg_third),
        (ax / 3.0) * (j_two_thirds - j_neg_two_thirds),
        (ax / 3.0).sqrt() * (j_neg_third - j_third),
        (ax / SQRT_3) * (j_neg_two_thirds + j_two_thirds),
    )
}

/// Which pair of routes the **positive** axis uses.
///
/// The two functions cross over at different arguments — `Ai` at [`AI_SERIES_MAX`] because it
/// cancels, `Bi` at the much later [`BI_SERIES_MAX`] because it does not — so this names the
/// combination rather than one method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PositiveRegime {
    /// Both from the series.
    BothSeries,
    /// `Ai` from the `K` quadrature, `Bi` still from the series.
    AiQuadrature,
    /// Both from their Bessel asymptotics.
    BothBessel,
}

impl PositiveRegime {
    pub(super) fn for_argument(x: f32) -> Self {
        if x > BI_SERIES_MAX {
            Self::BothBessel
        } else if x > AI_SERIES_MAX {
            Self::AiQuadrature
        } else {
            Self::BothSeries
        }
    }
}

/// `(Aiᵉ, Aiᵉ′, Biᵉ, Biᵉ′)` on the positive axis below [`LEADING_MIN`].
pub(super) fn positive_scaled(x: f32, z: f32) -> (f32, f32, f32, f32) {
    match PositiveRegime::for_argument(x) {
        // Both halves from their Bessel asymptotics, which are natively scaled.
        PositiveRegime::BothBessel => {
            let (ai, ai_prime) = ai_pair_scaled(x, z).into_parts();
            let (bi, bi_prime) = bi_pair_scaled(x, z).into_parts();
            (ai, ai_prime, bi, bi_prime)
        }
        // Both from the series, which is unscaled, so each pair takes its own exponential.
        PositiveRegime::BothSeries => {
            let (ai, ai_prime, bi, bi_prime) = series_airy(x);
            let (growth, decay) = (z.exp(), (-z).exp());
            (ai * growth, ai_prime * growth, bi * decay, bi_prime * decay)
        }
        // `Ai` has left the series but `Bi` has not: the quadrature for one, the series for the
        // other, and only the latter needs the exponential.
        PositiveRegime::AiQuadrature => {
            let (_, _, series_bi, series_bi_prime) = series_airy(x);
            let decay = (-z).exp();
            let (ai, ai_prime) = ai_pair_scaled(x, z).into_parts();
            (ai, ai_prime, series_bi * decay, series_bi_prime * decay)
        }
    }
}

/// All four unscaled values at once.
/// Which regime of the whole axis an Airy argument falls in.
///
/// # Why the whole axis, and not one chain per entry point
///
/// [`quad`] and [`quad_scaled`] each used to open with their own chain of comparisons against
/// `NaN`, the two infinities, zero, and [`LEADING_MIN`] — in *different orders*, which is the kind
/// of thing that is correct by luck rather than by construction. Naming the regimes puts the axis
/// in one place: the two entry points now agree about which regime an argument is in even where
/// they disagree about what to return for it, and [`PositiveRegime`] subdivides the one regime
/// that needs it.
///
/// # Mutation testing is what asked for this
///
/// Five mutants lived in those chains. Every one of them moved an argument between two routes
/// that agree about its value — the handovers are *sited* where they agree, which is what makes
/// them good handovers — so no value assertion could separate them. A named regime is a table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AiryRegime {
    /// Not a number.
    Undefined,
    /// `x = +∞`: `Ai` has long since underflowed and `Bi` diverged.
    AtPositiveInfinity,
    /// `x = −∞`: the oscillation continues, but its envelope has decayed to zero.
    AtNegativeInfinity,
    /// `|x| ≤ `[`SERIES_MAX`], negative: the Maclaurin series, whose terms alternate on this side
    /// rather than conspiring.
    NegativeSeries,
    /// `SERIES_MAX < |x| ≤ `[`MARCH_MAX`]: the Taylor march, for the band where series and
    /// asymptotic both fall short.
    NegativeMarch,
    /// `|x| > `[`MARCH_MAX`]: the `J` Hankel asymptotic of the oscillatory tail.
    NegativeAsymptotic,
    /// `x ≥ `[`LEADING_MIN`]: every asymptotic correction is below rounding. Unscaled, both
    /// functions have left the format and the limits are produced directly rather than by
    /// multiplying a finite scaled value by an infinite exponential.
    Leading,
    /// `0 ≤ x < LEADING_MIN`, including `−0.0`: subdivided by [`PositiveRegime`].
    Positive,
}

impl AiryRegime {
    /// The non-finite cases first, then the sign, then the two magnitude ladders.
    pub(super) fn for_argument(x: f32) -> Self {
        if x.is_nan() {
            return Self::Undefined;
        }
        if x == f32::INFINITY {
            return Self::AtPositiveInfinity;
        }
        if x == f32::NEG_INFINITY {
            return Self::AtNegativeInfinity;
        }
        if x < 0.0 {
            let ax = -x;
            return if ax <= SERIES_MAX {
                Self::NegativeSeries
            } else if ax <= MARCH_MAX {
                Self::NegativeMarch
            } else {
                Self::NegativeAsymptotic
            };
        }
        if x >= LEADING_MIN {
            return Self::Leading;
        }
        Self::Positive
    }
}

pub(super) fn quad(x: f32) -> (f32, f32, f32, f32) {
    match AiryRegime::for_argument(x) {
        AiryRegime::Undefined => (f32::NAN, f32::NAN, f32::NAN, f32::NAN),
        AiryRegime::AtPositiveInfinity => (0.0, 0.0, f32::INFINITY, f32::INFINITY),
        AiryRegime::AtNegativeInfinity => (0.0, 0.0, 0.0, 0.0),
        AiryRegime::NegativeSeries => series_airy(x),
        AiryRegime::NegativeMarch => march_airy(x),
        AiryRegime::NegativeAsymptotic => negative_asymptotic(-x),
        AiryRegime::Leading => (0.0, -0.0, f32::INFINITY, f32::INFINITY),
        AiryRegime::Positive => positive_unscaled(x),
    }
}

/// The quadruple on `0 ≤ x < `[`LEADING_MIN`].
///
/// Each pair takes the route that is unscaled *natively*, so no exponential is formed below the
/// relevant threshold. That matters: recovering an unscaled value from a scaled one costs a
/// relative `ζ·ε`, because `ζ`'s own rounding is amplified by `e^{ζ}` — and `ζ` reaches 18 by
/// `x = 9`, so in this format that is already `2e-6`.
pub(super) fn positive_unscaled(x: f32) -> (f32, f32, f32, f32) {
    let regime = PositiveRegime::for_argument(x);
    match regime {
        PositiveRegime::BothSeries => return series_airy(x),
        PositiveRegime::AiQuadrature | PositiveRegime::BothBessel => {}
    }
    let z = zeta(x);
    let (scaled_ai, scaled_ai_prime) = ai_pair_scaled(x, z).into_parts();
    let decay = (-z).exp();
    let (ai, ai_prime) = (scaled_ai * decay, scaled_ai_prime * decay);
    let (bi, bi_prime) = match regime {
        PositiveRegime::AiQuadrature => {
            let (_, _, series_bi, series_bi_prime) = series_airy(x);
            (series_bi, series_bi_prime)
        }
        PositiveRegime::BothSeries | PositiveRegime::BothBessel => {
            crate::math::scaling::unscale_pair(bi_pair_scaled(x, z), z).into_parts()
        }
    };
    (ai, ai_prime, bi, bi_prime)
}

/// All four scaled values at once. The negative axis is unscaled by convention.
pub(super) fn quad_scaled(x: f32) -> (f32, f32, f32, f32) {
    match AiryRegime::for_argument(x) {
        AiryRegime::Undefined => (f32::NAN, f32::NAN, f32::NAN, f32::NAN),
        // The negative axis is unscaled by convention, so there the scaled quadruple *is* the
        // unscaled one — including at `−∞`, where the envelope has decayed to zero.
        AiryRegime::AtNegativeInfinity
        | AiryRegime::NegativeSeries
        | AiryRegime::NegativeMarch
        | AiryRegime::NegativeAsymptotic => quad(x),
        // `Aiᵉ ~ 1/(2√π x^{1/4}) → 0`, `Biᵉ` likewise; the derivatives grow without bound.
        AiryRegime::AtPositiveInfinity => (0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY),
        AiryRegime::Leading => crate::math::airy::leading_scaled(x),
        AiryRegime::Positive => positive_scaled(x, zeta(x)),
    }
}
