//! The regime tables, and the four-at-once evaluation they route to.
//!
//! A child module so the parent stays inside the crate's 400-line budget. What lives here is the
//! *routing* rather than any one solution: [`AiryRegime`] divides the line, [`AiryPairing`]
//! decides whether the two solutions can be asked for separately or have to be taken together,
//! and `quad` is the shared evaluation both public pairs are carved out of.

use super::*;

/// `(Aiᵉ, Aiᵉ′, Biᵉ, Biᵉ′)` on the positive axis below [`LEADING_MIN`].
///
/// The two pairs cross over at different arguments — `Ai` at [`AI_SERIES_MAX`] because it
/// cancels, `Bi` at the much later [`BI_SERIES_MAX`] because it does not — so each takes its
/// own route and the series, when either needs it, is evaluated once.
pub(super) fn positive_scaled(x: f64, z: f64) -> (f64, f64, f64, f64) {
    match positive_regime(x) {
        // Both halves from their Bessel asymptotics, which are natively scaled.
        PositiveRegime::BothBessel => {
            let (ai, ai_prime) = ai_pair_scaled(x, z).into_parts();
            let (bi, bi_prime) = bi_pair_scaled(x, z).into_parts();
            (ai, ai_prime, bi, bi_prime)
        }
        // Both from the series, which is unscaled, so each pair takes its own exponential.
        PositiveRegime::BothSeries => {
            let (ai, ai_prime, bi, bi_prime) = series::airy(x);
            let (growth, decay) = (z.exp(), (-z).exp());
            (ai * growth, ai_prime * growth, bi * decay, bi_prime * decay)
        }
        // `Ai` has left the series but `Bi` has not: the quadrature for one, the series for the
        // other, and only the latter needs the exponential.
        PositiveRegime::AiQuadrature => {
            let (_, _, series_bi, series_bi_prime) = series::airy(x);
            let decay = (-z).exp();
            let (ai, ai_prime) = ai_pair_scaled(x, z).into_parts();
            (ai, ai_prime, series_bi * decay, series_bi_prime * decay)
        }
    }
}

/// `(Ai, Ai′, Bi, Bi′)` on the negative axis beyond [`MARCH_MAX`], from the Hankel expansion.
pub(super) fn negative_asymptotic(ax: f64) -> (f64, f64, f64, f64) {
    let z = zeta(ax);
    if !z.is_finite() {
        // Past `|x| ≈ 3e205` the phase is not merely inaccurate but unrepresentable: `ζ`
        // itself has overflowed, so there is no argument to hand to `sin`. Zero is finite,
        // inside the envelope, and the mean of the oscillation — the honest answer when the
        // phase is gone. Accuracy in fact departs far earlier, around `|x| ≈ 1e11`, where
        // `ulp(ζ)` passes `2π`; see the precision note on the trait.
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

/// The scaled quadruple `(Aiᵉ, Aiᵉ′, Biᵉ, Biᵉ′)`, finite at every finite argument.
pub(super) fn quad_scaled(x: f64) -> (f64, f64, f64, f64) {
    match AiryRegime::for_argument(x) {
        AiryRegime::Undefined => (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        // No scaling on this side, by the convention in the module header — including at `−∞`,
        // where the envelope has decayed to zero.
        AiryRegime::AtNegativeInfinity
        | AiryRegime::NegativeSeries
        | AiryRegime::NegativeMarch
        | AiryRegime::NegativeAsymptotic => quad(x),
        // `Aiᵉ ~ 1/(2√π x^{1/4}) → 0`, `Biᵉ` likewise; the derivatives grow without bound.
        AiryRegime::AtPositiveInfinity => (0.0, f64::NEG_INFINITY, 0.0, f64::INFINITY),
        AiryRegime::Leading => crate::math::airy::leading_scaled(x),
        AiryRegime::Positive => positive_scaled(x, zeta(x)),
    }
}

/// The unscaled quadruple `(Ai, Ai′, Bi, Bi′)`.
///
/// Overflows and underflows above `x ≈ 104` exactly as the true values do; reach for the scaled
/// forms there.
/// Whether `Ai` and `Bi` can be computed apart at this argument.
///
/// # Mutation testing is what asked for this
///
/// Narrowing this comparison survived in both [`pair`] and [`pair_scaled`], and the reason is
/// stated in [`pair`]'s own doc: the separable route is arithmetically identical to the
/// quadruple's, so routing the whole band back through `quad` changes **cost and nothing else**.
/// No value assertion can see it. The decision is what carries the saving, so the decision is
/// what gets pinned — and since performance is a goal here, that is not a formality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AiryPairing {
    /// `(AI_SERIES_MAX, LEADING_MIN)`: the two pairs share no intermediates, so asking for one
    /// half can skip the other.
    Separable,
    /// Everywhere else — the negative axis, the origin's series, the leading-order tail, the
    /// non-finite cases — the quadruple shares its work and is the cheaper route.
    Shared,
}

impl AiryPairing {
    /// The band is open at both ends, and every non-finite argument falls outside it.
    pub(super) fn for_argument(x: f64) -> Self {
        if x > AI_SERIES_MAX && x < LEADING_MIN {
            Self::Separable
        } else {
            Self::Shared
        }
    }
}

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
/// Matched rather than compared, so adding a regime is a compile error at every site that has to
/// decide what to do with it — which a chain of `if`s would have accepted in silence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AiryRegime {
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
    /// The non-finite cases first, then the sign, then the magnitude ladder.
    pub(super) fn for_argument(x: f64) -> Self {
        if x.is_nan() {
            return Self::Undefined;
        }
        if x == f64::INFINITY {
            return Self::AtPositiveInfinity;
        }
        if x == f64::NEG_INFINITY {
            return Self::AtNegativeInfinity;
        }
        if negative_axis(x) {
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

pub(super) fn quad(x: f64) -> (f64, f64, f64, f64) {
    match AiryRegime::for_argument(x) {
        AiryRegime::Undefined => (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        AiryRegime::AtPositiveInfinity => (0.0, 0.0, f64::INFINITY, f64::INFINITY),
        AiryRegime::AtNegativeInfinity => (0.0, 0.0, 0.0, 0.0),
        AiryRegime::NegativeSeries => series::airy(x),
        AiryRegime::NegativeMarch => march::airy(x),
        AiryRegime::NegativeAsymptotic => negative_asymptotic(-x),
        AiryRegime::Leading => (0.0, -0.0, f64::INFINITY, f64::INFINITY),
        AiryRegime::Positive => positive_unscaled(x),
    }
}
