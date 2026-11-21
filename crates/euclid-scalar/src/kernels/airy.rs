//! The Airy functions `Ai`, `Bi` and their derivatives — the two solutions of `y″ = xy`.
//!
//! This is the equation of a linear turning point, where a solution changes character from
//! oscillatory to exponential at a single value of `x`. That one property dictates the whole
//! implementation: there is no method that is good on both sides, so the axis is split at the
//! turning point and each side gets the method its own behavior calls for.
//!
//! # The four routes
//!
//! | range | method | why |
//! |-------|--------|-----|
//! | `0 ≤ x ≤ 1` (`Ai`, `Ai′`) | Maclaurin series, [`series`] | exact near the origin; the only route that survives `x → 0` |
//! | `x > 1` (`Ai`, `Ai′`) | `K_{1/3}`, `K_{2/3}` quadrature | no cancellation, and uniform in the argument |
//! | `0 ≤ x ≤ 9` (`Bi`, `Bi′`) | Maclaurin series | a *sum* of positive terms on this side, so it stays exact |
//! | `x > 9` (`Bi`, `Bi′`) | `I_{±1/3}`, `I_{±2/3}` asymptotic | overflow-free in the scaled form |
//! | `−2 ≤ x < 0` | Maclaurin series | the terms alternate here rather than conspiring |
//! | `−9 ≤ x < −2` | Taylor march, [`march`] | the band where series and asymptotic both fall short |
//! | `x < −9` | `J_{±1/3}`, `J_{±2/3}` Hankel asymptotic | the oscillatory tail |
//! | `x ≥ 1e10` | leading order | every correction term is below rounding |
//!
//! `Ai` and `Bi` leave the series at *different* arguments — `1` against `9` — because only
//! `Ai` is a difference of two growing quantities on the positive axis. That asymmetry is the
//! single most consequential thing in this module; see [`AI_SERIES_MAX`] for the measurements
//! that site it.
//!
//! Three of those reach into [`crate::kernels::bessel`] rather than growing their own series,
//! through the connection formulas below. That is why its `K` quadrature takes `ν` as a
//! parameter instead of fixing it at `0` and `1`, and why its asymptotic expansions are written
//! for general order: with `ζ = ⅔|x|^{3/2}`,
//!
//! ```text
//! x > 0:   Ai(x)  = π^{−1}√(x/3)·K_{1/3}(ζ)        Ai′(x)  = −π^{−1}(x/√3)·K_{2/3}(ζ)
//!          Bi(x)  = √(x/3)[I_{−1/3}(ζ) + I_{1/3}(ζ)]
//!          Bi′(x) = (x/√3)[I_{−2/3}(ζ) + I_{2/3}(ζ)]
//!
//! x < 0:   Ai(−z) = (√z/3)[J_{1/3}(ζ) + J_{−1/3}(ζ)]
//!          Bi(−z) = √(z/3)[J_{−1/3}(ζ) − J_{1/3}(ζ)]
//!          Ai′(−z) = (z/3)[J_{2/3}(ζ) − J_{−2/3}(ζ)]
//!          Bi′(−z) = (z/√3)[J_{−2/3}(ζ) + J_{2/3}(ζ)]
//! ```
//!
//! # The scaled forms are the primitive ones
//!
//! On the positive axis the pair leaves `f64`'s range long before the arguments stop being
//! useful: `Ai(104)` underflows and `Bi(104)` overflows, while `Aiᵉ(104) = 0.024` and
//! `Biᵉ(104) = 0.049` are ordinary numbers. So, as with `I` and `K`, the scaled forms
//!
//! ```text
//! Aiᵉ(x) = e^{ζ}Ai(x)      Biᵉ(x) = e^{−ζ}Bi(x)
//! ```
//!
//! are computed directly and the unscaled values recovered by one multiplication that is
//! *allowed* to overflow or underflow, as the true values do.
//!
//! **On the negative axis the scaled forms are defined to equal the unscaled ones.** There the
//! functions oscillate inside a decaying envelope and need no scaling; `ζ` is the phase rather
//! than a magnitude, so `e^{ζ}` would be meaningless. This is a convention the mathematics does
//! not settle, and it matches what the scaled forms are *for*: keeping a value in range.
//!
//! # All four values are produced together, where that is free
//!
//! On the negative axis and at the origin every route genuinely produces the quadruple: `series`
//! and `march` transport `f` and `g`, and all four values come out of one [`crate::math::airy::combine`]. So
//! the Wronskian `Ai·Bi′ − Ai′·Bi = 1/π` holds there *by construction* rather than by
//! coincidence, because the four do share one set of intermediates.
//!
//! **Above [`AI_SERIES_MAX`] on the positive axis they share nothing**, and the eight entry
//! points used to pay for that anyway. `Ai` comes from a `K_{1/3}`, `K_{2/3}` quadrature and `Bi`
//! from either the series or an `I` asymptotic; measured, the quadruple cost `496 ns` while the
//! `Ai` half alone was `292 ns` and the `Bi` half `52 ns`. [`pair`] now computes only the half
//! asked for, bit-identically, and falls back to the quadruple wherever the sharing is real.
//!
//! Note that [`crate::kernels::bessel`] does **not** make the same trade, though this paragraph
//! used to say it did: `j0` takes a single value from `hankel`, whose `P` and `Q` sums are
//! interleaved in one loop and so are shared for free rather than by choice.

use crate::kernels::bessel::asymptotic::{hankel, i_scaled};
use crate::kernels::bessel::modified_second::k_nu_pair_scaled;
use crate::math::pair::{Kind, Slope, Solutions};
use core::f64::consts::PI;

mod march;
mod series;

#[cfg(test)]
mod tests;

use series::SQRT_3;

/// At or below this `|x|` the **negative** axis uses the Maclaurin series, and it is where
/// [`march`] starts.
///
/// The negative axis tolerates far more series than the positive one: the terms alternate
/// rather than conspiring, so the measured absolute error at `x = −2` is `6e-17`. Starting the
/// march here rather than closer to the origin also keeps its step count down.
const SERIES_MAX: f64 = 2.0;

/// At or below this `x` the **positive** axis takes `Ai` and `Ai′` from the series; above it,
/// from the `K` quadrature.
///
/// This is the tightest threshold in the module, and it is set by the cancellation described in
/// [`series`]: `Ai = c₁f − c₂g` discards about `2ζ/ln 10` digits on this side. Measured
/// relative error in `Ai′`, series against quadrature:
///
/// | `x` | 1.0 | 1.5 | 1.8 | 1.95 | 2.25 | 2.5 |
/// |-----|-----|-----|-----|------|------|-----|
/// | series | `5e-16` | `4e-16` | `1.2e-15` | `1.5e-14` | `1.2e-14` | `2.6e-14` |
/// | `K` quadrature | `1.7e-16` | `4.3e-16` | `2.0e-16` | `0` | `1.8e-16` | `2.6e-16` |
///
/// The quadrature is better than the series from about `x = 1` on, and never worse anywhere
/// above it, so the crossover goes here rather than at [`SERIES_MAX`]. The series is still
/// needed *below* this point, and is the only thing that works there: `ζ = ⅔x^{3/2}` underflows
/// to zero around `x = 1e-205`, which would leave the quadrature integrating nothing at all.
const AI_SERIES_MAX: f64 = 1.0;

/// Up to this `x` the positive axis keeps taking `Bi` and `Bi′` from the series.
///
/// `Bi` is a sum of positive terms on this side, so none of `Ai`'s cancellation applies and the
/// series stays exact much further out — measured at or below `9e-16` all the way to `x = 40`.
/// The switch happens here anyway because the `I` asymptotic has reached machine precision by
/// `x = 9` (`1.7e-16`, against `5.8e-15` at `x = 8`), is cheaper, and gives the scaled form
/// without ever forming `e^{ζ}`.
const BI_SERIES_MAX: f64 = 9.0;

/// Up to this `|x|` the negative axis marches the equation outward; beyond it, Hankel.
///
/// The march holds machine precision indefinitely but costs a step per half-unit, so the
/// crossover is set by where the asymptotic becomes trustworthy: `1e-14` at `|x| = 8` against
/// `2e-10` at `|x| = 6`. See [`march`] for the measured table on both sides.
const MARCH_MAX: f64 = 9.0;

/// At or above this `x` every asymptotic correction term is below rounding, so the leading term
/// alone is used.
///
/// The first correction is `u₁/ζ = (5/72)/ζ`, which at `x = 1e10` is `7e-18`. Measured against
/// the Bessel routes the two agree to `1.2e-16` here and to `3e-15` an order of magnitude lower,
/// which is what sites the threshold. Taking this branch also keeps `ζ` out of the arithmetic
/// entirely — it would overflow above `x ≈ 4.17e205`, and the leading forms are exactly the ones
/// that stay representable: `Aiᵉ → 1/(2√π·x^{1/4})` and `Biᵉ → 1/(√π·x^{1/4})`.
const LEADING_MIN: f64 = 1e10;

/// `ζ = ⅔|x|^{3/2}`, as `⅔·|x|·√|x|` rather than through `powf`.
fn zeta(ax: f64) -> f64 {
    (2.0 / 3.0) * ax * ax.sqrt()
}

/// `(Aiᵉ, Aiᵉ′)` from the `K_{1/3}`, `K_{2/3}` quadrature, for `x > `[`AI_SERIES_MAX`].
///
/// Naturally scaled: [`k_nu_pair_scaled`] *is* `e^{ζ}K_ν(ζ)`, so no exponential is formed here at
/// all, and the result is finite however large `x` becomes.
fn ai_pair_scaled(x: f64, z: f64) -> Slope<f64> {
    // One sweep of quadrature nodes for both orders: the nodes depend only on `z`, so asking for
    // the pair together halves the `exp` and `sinh` work and returns the same bits. See
    // `k_nu_pair_scaled`.
    //
    // This was the largest single cost in the kernel. `Ai` and `Bi` were the slowest functions
    // in the crate by a factor of five — the positive axis goes through `K`, and it was paying
    // for two complete quadratures over an identical node list. Sharing the sweep cuts both by
    // **29%**, measured bit-identically across a 306,280-point sweep.
    let (k_third, k_two_thirds) = k_nu_pair_scaled(1.0 / 3.0, 2.0 / 3.0, z);
    Slope::new(
        (x / 3.0).sqrt() * k_third / PI,
        -(x / SQRT_3) * k_two_thirds / PI,
    )
}

/// `(Biᵉ, Biᵉ′)` from the scaled `I` asymptotic for `x > BI_SERIES_MAX`.
/// The asymptotic uses only the square of the order, so each `I₋ν + Iν`
/// contributes twice the value at positive order.
fn bi_pair_scaled(x: f64, z: f64) -> Slope<f64> {
    Slope::new(
        (x / 3.0).sqrt() * (2.0 * i_scaled(1.0 / 3.0, z)),
        (x / SQRT_3) * (2.0 * i_scaled(2.0 / 3.0, z)),
    )
}

/// Both signed zeros use the series at the origin.
fn negative_axis(x: f64) -> bool {
    x < 0.0
}

mod regimes;

// The parent carves the two public pairs out of `quad`, so it needs the regime tables and both
// four-at-once evaluations back.
use regimes::{AiryPairing, PositiveRegime, quad, quad_scaled};

/// `(Ai, Ai′)` or `(Bi, Bi′)` alone, skipping the other solution where it is separable.
///
/// # Why one pair and not the quadruple
///
/// The module header defends computing all four together on the grounds that the Wronskian then
/// holds by construction. That is true on the negative axis and at the origin, where `series` and
/// `march` produce `f` and `g` and all four values come out of one [`crate::math::airy::combine`] — there is
/// nothing to skip, and asking for a half would cost the same as asking for all of it.
///
/// Above [`AI_SERIES_MAX`] on the positive axis it is **not** true, because the two pairs share no
/// intermediates: `Ai` comes from a `K_{1/3}`, `K_{2/3}` quadrature and `Bi` from either the
/// series or an `I` asymptotic. Every entry point was paying for both. Measured over `x ∈ [1.5,
/// 30]`, `quad` costs `496 ns` while `ai_pair_scaled` alone is `292 ns` and `bi_pair_scaled`
/// alone is `52 ns` — so `bi` was paying nearly ten times what it needed and `ai` about one and
/// two thirds.
///
/// The arithmetic per half is character-for-character what [`quad`] does, in the same order, so
/// each value is bit-identical to what the quadruple produced. Splitting changes cost, not
/// results — including the Wronskian, which combines the two halves and gets the same two pairs
/// whether they were computed in one call or two.
fn pair(x: f64, kind: Kind) -> Slope<f64> {
    // Outside the separable band the quadruple shares its work and is the cheaper route; see
    // [`AiryPairing`] for the band and why the decision is named.
    match AiryPairing::for_argument(x) {
        AiryPairing::Shared => quad(x).of_kind(kind),
        AiryPairing::Separable => {
            let z = zeta(x);
            match kind {
                Kind::First => ai_pair_scaled(x, z).scaled_by((-z).exp()),
                Kind::Second => match PositiveRegime::for_argument(x) {
                    PositiveRegime::AiQuadrature => series::airy(x).second_kind,
                    PositiveRegime::BothSeries | PositiveRegime::BothBessel => {
                        crate::math::scaling::unscale_pair(bi_pair_scaled(x, z), z)
                    }
                },
            }
        }
    }
}

/// The scaled peer of [`pair`]; see there for why the split is only valid on part of the axis.
fn pair_scaled(x: f64, kind: Kind) -> Slope<f64> {
    match AiryPairing::for_argument(x) {
        AiryPairing::Shared => quad_scaled(x).of_kind(kind),
        AiryPairing::Separable => {
            let z = zeta(x);
            match kind {
                Kind::First => ai_pair_scaled(x, z),
                Kind::Second => match PositiveRegime::for_argument(x) {
                    PositiveRegime::AiQuadrature => {
                        series::airy(x).second_kind.scaled_by((-z).exp())
                    }
                    PositiveRegime::BothSeries | PositiveRegime::BothBessel => bi_pair_scaled(x, z),
                },
            }
        }
    }
}

/// The Airy function of the first kind `Ai(x)`.
pub(crate) fn ai(x: f64) -> f64 {
    pair(x, Kind::First).value
}

/// The derivative `Ai′(x)`.
pub(crate) fn ai_prime(x: f64) -> f64 {
    pair(x, Kind::First).derivative
}

/// The Airy function of the second kind `Bi(x)`.
pub(crate) fn bi(x: f64) -> f64 {
    pair(x, Kind::Second).value
}

/// The derivative `Bi′(x)`.
pub(crate) fn bi_prime(x: f64) -> f64 {
    pair(x, Kind::Second).derivative
}

/// The scaled `Aiᵉ(x) = e^{ζ}Ai(x)` for `x > 0`, and `Ai(x)` itself for `x ≤ 0`.
pub(crate) fn aie(x: f64) -> f64 {
    pair_scaled(x, Kind::First).value
}

/// The scaled `Aiᵉ′(x) = e^{ζ}Ai′(x)` for `x > 0`, and `Ai′(x)` itself for `x ≤ 0`.
pub(crate) fn aie_prime(x: f64) -> f64 {
    pair_scaled(x, Kind::First).derivative
}

/// The scaled `Biᵉ(x) = e^{−ζ}Bi(x)` for `x > 0`, and `Bi(x)` itself for `x ≤ 0`.
pub(crate) fn bie(x: f64) -> f64 {
    pair_scaled(x, Kind::Second).value
}

/// The scaled `Biᵉ′(x) = e^{−ζ}Bi′(x)` for `x > 0`, and `Bi′(x)` itself for `x ≤ 0`.
pub(crate) fn bie_prime(x: f64) -> f64 {
    pair_scaled(x, Kind::Second).derivative
}
