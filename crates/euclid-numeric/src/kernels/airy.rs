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
//! # All four values are produced together
//!
//! Every route returns the whole quadruple `(Ai, Ai′, Bi, Bi′)` and the eight entry points
//! select from it. That costs some arithmetic when a caller wants one value, and buys something
//! worth more: the Wronskian `Ai·Bi′ − Ai′·Bi = 1/π` holds *by construction* rather than by
//! coincidence, because the four share one set of intermediates. [`crate::kernels::bessel`]
//! makes the same trade for `J`/`Y`, for the same reason.

use crate::constants::f64::SQRT_PI;
use crate::kernels::bessel::asymptotic::{hankel, i_scaled};
use crate::kernels::bessel::modified_second::k_nu_scaled;
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
/// entirely — it would overflow above `x ≈ 3e205`, and the leading forms are exactly the ones
/// that stay representable: `Aiᵉ → 1/(2√π·x^{1/4})` and `Biᵉ → 1/(√π·x^{1/4})`.
const LEADING_MIN: f64 = 1e10;

/// `ζ = ⅔|x|^{3/2}`, as `⅔·|x|·√|x|` rather than through `powf`.
fn zeta(ax: f64) -> f64 {
    (2.0 / 3.0) * ax * ax.sqrt()
}

/// `(Aiᵉ, Aiᵉ′)` from the `K_{1/3}`, `K_{2/3}` quadrature, for `x > `[`AI_SERIES_MAX`].
///
/// Naturally scaled: [`k_nu_scaled`] *is* `e^{ζ}K_ν(ζ)`, so no exponential is formed here at
/// all, and the result is finite however large `x` becomes.
fn ai_pair_scaled(x: f64, z: f64) -> (f64, f64) {
    (
        (x / 3.0).sqrt() * k_nu_scaled(1.0 / 3.0, z) / PI,
        -(x / SQRT_3) * k_nu_scaled(2.0 / 3.0, z) / PI,
    )
}

/// `(Biᵉ, Biᵉ′)` from the `I_{±1/3}`, `I_{±2/3}` asymptotic, for `x > `[`BI_SERIES_MAX`].
///
/// Naturally scaled for the same reason: [`i_scaled`] is `e^{−ζ}I_ν(ζ)`.
fn bi_pair_scaled(x: f64, z: f64) -> (f64, f64) {
    (
        (x / 3.0).sqrt() * (i_scaled(-1.0 / 3.0, z) + i_scaled(1.0 / 3.0, z)),
        (x / SQRT_3) * (i_scaled(-2.0 / 3.0, z) + i_scaled(2.0 / 3.0, z)),
    )
}

/// `(Aiᵉ, Aiᵉ′, Biᵉ, Biᵉ′)` on the positive axis below [`LEADING_MIN`].
///
/// The two pairs cross over at different arguments — `Ai` at [`AI_SERIES_MAX`] because it
/// cancels, `Bi` at the much later [`BI_SERIES_MAX`] because it does not — so each takes its
/// own route and the series, when either needs it, is evaluated once.
fn positive_scaled(x: f64, z: f64) -> (f64, f64, f64, f64) {
    if x > BI_SERIES_MAX {
        let (ai, ai_prime) = ai_pair_scaled(x, z);
        let (bi, bi_prime) = bi_pair_scaled(x, z);
        return (ai, ai_prime, bi, bi_prime);
    }
    let (series_ai, series_ai_prime, series_bi, series_bi_prime) = series::airy(x);
    let decay = (-z).exp();
    let (bi, bi_prime) = (series_bi * decay, series_bi_prime * decay);
    if x <= AI_SERIES_MAX {
        let growth = z.exp();
        return (series_ai * growth, series_ai_prime * growth, bi, bi_prime);
    }
    let (ai, ai_prime) = ai_pair_scaled(x, z);
    (ai, ai_prime, bi, bi_prime)
}

/// `(Ai, Ai′, Bi, Bi′)` on the negative axis beyond [`MARCH_MAX`], from the Hankel expansion.
fn negative_asymptotic(ax: f64) -> (f64, f64, f64, f64) {
    let z = zeta(ax);
    if !z.is_finite() {
        // Past `|x| ≈ 3e205` the phase is not merely inaccurate but unrepresentable: `ζ`
        // itself has overflowed, so there is no argument to hand to `sin`. Zero is finite,
        // inside the envelope, and the mean of the oscillation — the honest answer when the
        // phase is gone. Accuracy in fact departs far earlier, around `|x| ≈ 1e11`, where
        // `ulp(ζ)` passes `2π`; see the precision note on the trait.
        return (0.0, 0.0, 0.0, 0.0);
    }
    let (j_third, _) = hankel(1.0 / 3.0, z);
    let (j_neg_third, _) = hankel(-1.0 / 3.0, z);
    let (j_two_thirds, _) = hankel(2.0 / 3.0, z);
    let (j_neg_two_thirds, _) = hankel(-2.0 / 3.0, z);
    (
        (ax.sqrt() / 3.0) * (j_third + j_neg_third),
        (ax / 3.0) * (j_two_thirds - j_neg_two_thirds),
        (ax / 3.0).sqrt() * (j_neg_third - j_third),
        (ax / SQRT_3) * (j_neg_two_thirds + j_two_thirds),
    )
}

/// `(Aiᵉ, Aiᵉ′, Biᵉ, Biᵉ′)` at or above [`LEADING_MIN`], where the corrections vanish.
fn leading_scaled(x: f64) -> (f64, f64, f64, f64) {
    let quarter = x.sqrt().sqrt();
    (
        1.0 / (2.0 * SQRT_PI * quarter),
        -quarter / (2.0 * SQRT_PI),
        1.0 / (SQRT_PI * quarter),
        quarter / SQRT_PI,
    )
}

/// The scaled quadruple `(Aiᵉ, Aiᵉ′, Biᵉ, Biᵉ′)`, finite at every finite argument.
fn quad_scaled(x: f64) -> (f64, f64, f64, f64) {
    if x.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    if x < 0.0 {
        // No scaling on this side, by the convention in the module header.
        return quad(x);
    }
    if x.is_infinite() {
        // `Aiᵉ ~ 1/(2√π x^{1/4}) → 0`, `Biᵉ` likewise; the derivatives grow without bound.
        return (0.0, f64::NEG_INFINITY, 0.0, f64::INFINITY);
    }
    if x >= LEADING_MIN {
        return leading_scaled(x);
    }
    positive_scaled(x, zeta(x))
}

/// The unscaled quadruple `(Ai, Ai′, Bi, Bi′)`.
///
/// Overflows and underflows above `x ≈ 104` exactly as the true values do; reach for the scaled
/// forms there.
fn quad(x: f64) -> (f64, f64, f64, f64) {
    if x.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    if x.is_infinite() {
        return if x > 0.0 {
            // `Ai → 0` from above, `Bi → ∞`.
            (0.0, 0.0, f64::INFINITY, f64::INFINITY)
        } else {
            // The oscillation never settles, so no limit exists; the envelope does go to zero.
            (0.0, 0.0, 0.0, 0.0)
        };
    }
    if x < 0.0 {
        let ax = -x;
        return if ax <= SERIES_MAX {
            series::airy(x)
        } else if ax <= MARCH_MAX {
            march::airy(x)
        } else {
            negative_asymptotic(ax)
        };
    }
    if x >= LEADING_MIN {
        // `Ai` has long since underflowed and `Bi` overflowed; produce those limits rather
        // than multiplying a finite scaled value by an infinite exponential.
        return (0.0, -0.0, f64::INFINITY, f64::INFINITY);
    }
    // Each pair takes the route that is unscaled *natively*, so no exponential is formed below
    // the relevant threshold. That matters: recovering an unscaled value from a scaled one
    // costs a relative `ζ·ε`, because `ζ`'s own rounding is amplified by `e^{ζ}`. Measured,
    // that is `1.7e-15` at `x = 20` and `1.5e-13` at `x = 100` — unavoidable once the
    // exponential is needed, and worth avoiding entirely while it is not.
    let z = zeta(x);
    let (ai, ai_prime) = if x <= AI_SERIES_MAX {
        let (series_ai, series_ai_prime, _, _) = series::airy(x);
        (series_ai, series_ai_prime)
    } else {
        let (scaled_ai, scaled_ai_prime) = ai_pair_scaled(x, z);
        let decay = (-z).exp();
        (scaled_ai * decay, scaled_ai_prime * decay)
    };
    let (bi, bi_prime) = if x <= BI_SERIES_MAX {
        let (_, _, series_bi, series_bi_prime) = series::airy(x);
        (series_bi, series_bi_prime)
    } else {
        let (scaled_bi, scaled_bi_prime) = bi_pair_scaled(x, z);
        let growth = z.exp();
        (scaled_bi * growth, scaled_bi_prime * growth)
    };
    (ai, ai_prime, bi, bi_prime)
}

/// The Airy function of the first kind `Ai(x)`.
pub(crate) fn ai(x: f64) -> f64 {
    quad(x).0
}

/// The derivative `Ai′(x)`.
pub(crate) fn ai_prime(x: f64) -> f64 {
    quad(x).1
}

/// The Airy function of the second kind `Bi(x)`.
pub(crate) fn bi(x: f64) -> f64 {
    quad(x).2
}

/// The derivative `Bi′(x)`.
pub(crate) fn bi_prime(x: f64) -> f64 {
    quad(x).3
}

/// The scaled `Aiᵉ(x) = e^{ζ}Ai(x)` for `x > 0`, and `Ai(x)` itself for `x ≤ 0`.
pub(crate) fn aie(x: f64) -> f64 {
    quad_scaled(x).0
}

/// The scaled `Aiᵉ′(x) = e^{ζ}Ai′(x)` for `x > 0`, and `Ai′(x)` itself for `x ≤ 0`.
pub(crate) fn aie_prime(x: f64) -> f64 {
    quad_scaled(x).1
}

/// The scaled `Biᵉ(x) = e^{−ζ}Bi(x)` for `x > 0`, and `Bi(x)` itself for `x ≤ 0`.
pub(crate) fn bie(x: f64) -> f64 {
    quad_scaled(x).2
}

/// The scaled `Biᵉ′(x) = e^{−ζ}Bi′(x)` for `x > 0`, and `Bi′(x)` itself for `x ≤ 0`.
pub(crate) fn bie_prime(x: f64) -> f64 {
    quad_scaled(x).3
}
