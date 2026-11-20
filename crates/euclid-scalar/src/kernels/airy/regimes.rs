//! The regime tables, and the four-at-once evaluation they route to.
//!
//! A child module so the parent stays inside the crate's 400-line budget. What lives here is the
//! *routing* rather than any one solution: [`AiryRegime`] divides the line, [`PositiveRegime`]
//! subdivides the one part of it that needs subdividing, [`AiryPairing`] decides whether the two
//! solutions can be asked for separately or have to be taken together, and `quad` is the shared
//! evaluation both public pairs are carved out of.
//!
//! Every route here returns a [`Solutions<Slope<f64>>`](Solutions) — the Airy quadruple, named:
//! two solutions of `y″ = xy`, each carrying its own derivative. It used to be a bare
//! `(f64, f64, f64, f64)` threaded through eight functions and reached with `.0` to `.3`, which
//! is four same-typed numbers in a convention no compiler was checking.

use super::*;
use crate::math::airy::OscillatoryPhase;

/// Which pair of routes the **positive** axis uses.
///
/// The two functions cross over at different arguments — `Ai` at [`AI_SERIES_MAX`] because it
/// cancels, `Bi` at the much later [`BI_SERIES_MAX`] because it does not — so this names the
/// combination rather than one method. That asymmetry is the single most consequential thing in
/// the family, and it is the reason there are three arms here instead of two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PositiveRegime {
    /// `x ≤ `[`AI_SERIES_MAX`]: both from the series.
    BothSeries,
    /// `AI_SERIES_MAX < x ≤ `[`BI_SERIES_MAX`]: `Ai` from the `K` quadrature, `Bi` still from
    /// the series.
    AiQuadrature,
    /// `x > `[`BI_SERIES_MAX`]: both from their Bessel asymptotics.
    BothBessel,
}

impl PositiveRegime {
    /// The later handover first, so each arm is reached by one comparison.
    pub(super) fn for_argument(x: f64) -> Self {
        if x > BI_SERIES_MAX {
            Self::BothBessel
        } else if x > AI_SERIES_MAX {
            Self::AiQuadrature
        } else {
            Self::BothSeries
        }
    }
}

/// The scaled quadruple on the positive axis below [`LEADING_MIN`].
///
/// Each pair takes its own route, and the series — when either needs it — is evaluated once.
pub(super) fn positive_scaled(x: f64, z: f64) -> Solutions<Slope<f64>> {
    match PositiveRegime::for_argument(x) {
        // Both halves from their Bessel asymptotics, which are natively scaled.
        PositiveRegime::BothBessel => Solutions::new(ai_pair_scaled(x, z), bi_pair_scaled(x, z)),
        // Both from the series, which is unscaled, so each pair takes its own exponential — and
        // they take *different* ones: `Aiᵉ = e^{ζ}Ai` grows where `Biᵉ = e^{−ζ}Bi` decays.
        PositiveRegime::BothSeries => {
            let series = series::airy(x);
            Solutions::new(
                series.first_kind.scaled_by(z.exp()),
                series.second_kind.scaled_by((-z).exp()),
            )
        }
        // `Ai` has left the series but `Bi` has not: the quadrature for one, the series for the
        // other, and only the latter needs the exponential.
        PositiveRegime::AiQuadrature => Solutions::new(
            ai_pair_scaled(x, z),
            series::airy(x).second_kind.scaled_by((-z).exp()),
        ),
    }
}

/// The unscaled quadruple on `0 ≤ x < `[`LEADING_MIN`].
///
/// Each pair takes the route that is unscaled *natively*, so no exponential is formed below the
/// relevant threshold. That matters: recovering an unscaled value from a scaled one costs a
/// relative `ζ·ε`, because `ζ`'s own rounding is amplified by `e^{ζ}`. Measured, that is
/// `1.7e-15` at `x = 20` and `1.5e-13` at `x = 100` — unavoidable once the exponential is
/// needed, and worth avoiding entirely while it is not.
///
/// Named rather than inlined in [`quad`] so the decay multiply is reachable: the unscaled
/// surface's own sweeps carry a budget wide enough to absorb a change to it, and `Ai′` is small
/// across this band. The single-precision peer carries the same split.
pub(super) fn positive_unscaled(x: f64) -> Solutions<Slope<f64>> {
    let regime = PositiveRegime::for_argument(x);
    match regime {
        PositiveRegime::BothSeries => return series::airy(x),
        PositiveRegime::AiQuadrature | PositiveRegime::BothBessel => {}
    }
    let z = zeta(x);
    let ai = ai_pair_scaled(x, z).scaled_by((-z).exp());
    let bi = match regime {
        PositiveRegime::AiQuadrature => series::airy(x).second_kind,
        PositiveRegime::BothSeries | PositiveRegime::BothBessel => {
            crate::math::scaling::unscale_pair(bi_pair_scaled(x, z), z)
        }
    };
    Solutions::new(ai, bi)
}

/// The quadruple on the negative axis beyond [`MARCH_MAX`], from the Hankel expansion.
///
/// Four separate `J` evaluations rather than one shared sweep: the four orders `±⅓`, `±⅔` are
/// genuinely different expansions, so unlike the positive axis there is nothing to share.
pub(super) fn negative_asymptotic(ax: f64) -> Solutions<Slope<f64>> {
    let z = match OscillatoryPhase::for_zeta(zeta(ax)) {
        OscillatoryPhase::Known(z) => z,
        // The phase has overflowed; see `OscillatoryPhase` for why zero is the honest answer.
        OscillatoryPhase::Lost => return Solutions::uniform(Slope::uniform(0.0)),
    };
    let j_third = hankel(1.0 / 3.0, z).first_kind;
    let j_neg_third = hankel(-1.0 / 3.0, z).first_kind;
    let j_two_thirds = hankel(2.0 / 3.0, z).first_kind;
    let j_neg_two_thirds = hankel(-2.0 / 3.0, z).first_kind;
    Solutions::new(
        Slope::new(
            (ax.sqrt() / 3.0) * (j_third + j_neg_third),
            (ax / 3.0) * (j_two_thirds - j_neg_two_thirds),
        ),
        Slope::new(
            (ax / 3.0).sqrt() * (j_neg_third - j_third),
            (ax / SQRT_3) * (j_neg_two_thirds + j_two_thirds),
        ),
    )
}

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

/// The unscaled quadruple `(Ai, Ai′, Bi, Bi′)`.
///
/// Overflows and underflows above `x ≈ 104` exactly as the true values do; reach for the scaled
/// forms there, which is what the saturating `Leading` arm below is saying.
pub(super) fn quad(x: f64) -> Solutions<Slope<f64>> {
    match AiryRegime::for_argument(x) {
        AiryRegime::Undefined => Solutions::uniform(Slope::uniform(f64::NAN)),
        AiryRegime::AtPositiveInfinity => {
            Solutions::new(Slope::uniform(0.0), Slope::uniform(f64::INFINITY))
        }
        AiryRegime::AtNegativeInfinity => Solutions::uniform(Slope::uniform(0.0)),
        AiryRegime::NegativeSeries => series::airy(x),
        AiryRegime::NegativeMarch => march::airy(x),
        AiryRegime::NegativeAsymptotic => negative_asymptotic(-x),
        AiryRegime::Leading => Solutions::new(Slope::new(0.0, -0.0), Slope::uniform(f64::INFINITY)),
        AiryRegime::Positive => positive_unscaled(x),
    }
}

/// The scaled quadruple `(Aiᵉ, Aiᵉ′, Biᵉ, Biᵉ′)`, finite at every finite argument.
pub(super) fn quad_scaled(x: f64) -> Solutions<Slope<f64>> {
    match AiryRegime::for_argument(x) {
        AiryRegime::Undefined => Solutions::uniform(Slope::uniform(f64::NAN)),
        // No scaling on this side, by the convention in the module header — including at `−∞`,
        // where the envelope has decayed to zero.
        AiryRegime::AtNegativeInfinity
        | AiryRegime::NegativeSeries
        | AiryRegime::NegativeMarch
        | AiryRegime::NegativeAsymptotic => quad(x),
        // `Aiᵉ ~ 1/(2√π x^{1/4}) → 0`, `Biᵉ` likewise; the derivatives grow without bound.
        AiryRegime::AtPositiveInfinity => Solutions::new(
            Slope::new(0.0, f64::NEG_INFINITY),
            Slope::new(0.0, f64::INFINITY),
        ),
        AiryRegime::Leading => crate::math::airy::leading_scaled(x),
        AiryRegime::Positive => positive_scaled(x, zeta(x)),
    }
}
