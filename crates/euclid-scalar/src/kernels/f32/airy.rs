//! The `f32` peer of the Airy family: `Ai`, `Bi`, their derivatives, and the scaled forms.
//!
//! Native single precision. This was the last function family in the tree to narrow the `f64`
//! reference, and the reason given was the strongest of the five such claims: Airy reaches its
//! values through *all three* of the mechanisms whose error floor is set by the working
//! precision — a cancelling Maclaurin series, an outward march of the equation, and a divergent
//! asymptotic expansion.
//!
//! The specific claim was that "in `f32` the series' `ε·e^{ζ}` cancellation alone would leave
//! `Ai(6)` with no correct digits". **That is numerically true.** Measured, the series in `f32`
//! carries a relative error of `2.35e1` in `Ai(6)` and `8.1e8` by `x = 9`, tracking the
//! `2ζ/ln 10` digit loss the series module predicts.
//!
//! It is also a statement about an argument the series is never evaluated at for `Ai`.
//! [`AI_SERIES_MAX`] is `1`, so `Ai(6)` comes from the `K` quadrature. Where the series *is*
//! used it is fine:
//!
//! | `x` | 0.5 | **1.0** | **2.0** | 3.0 | 6.0 | 9.0 |
//! |---|---|---|---|---|---|---|
//! | `Ai` series | `6.2e-9` | **`2.6e-7`** | **`1.2e-6`** | `4.5e-5` | `2.4e1` | `8.1e8` |
//! | `Bi` series | `3.6e-8` | `2.3e-8` | `6.9e-8` | `5.1e-8` | `2.8e-8` | `1.4e-7` |
//!
//! The bold columns are the two places it runs: `Ai` up to [`AI_SERIES_MAX`], and `|x| ≤ 2` as
//! the negative march's seed. `Bi`'s row never cancels at all — it is a sum of positive terms on
//! the positive axis — which is why its own handover sits as late as `9`.
//!
//! The other two mechanisms port because the pieces already exist natively: `Ai` on the positive
//! axis goes through `K_{1/3}`/`K_{2/3}`, whose quadrature retuned to this format uses about a
//! third of the `f64` node count, and `Bi` through the scaled `I` asymptotic. The march is the
//! same device `Y` uses and is sound here for the same reason — on the negative axis `Ai` and
//! `Bi` both oscillate with amplitude decaying like `|x|^{−1/4}`, so neither dominates and an
//! error introduced at one step is transported rather than amplified. On the *positive* axis the
//! pair is maximally unequal and marching `Ai` outward would be catastrophic, which is why the
//! march is reached only for `x < 0`.

#[cfg(test)]
mod tests;

use super::bessel::modified_first::i_scaled_asymptotic;
use super::bessel::modified_second::k_nu_pair_scaled;
use super::bessel::ordinary::hankel;
use crate::math::arithmetic::term_is_negligible;
use crate::math::pair::Slope;
use core::f32::consts::PI;

/// Iteration cap for the Maclaurin series. The `f64` peer allows 400.
///
/// The series runs over `|x| ≤ `[`BI_SERIES_MAX`], where the term ratio is `x³/(3k(3k−1))`, so it
/// converges superexponentially once `k` passes `x`. 64 is a backstop.
const MAX_TERMS: u32 = 64;

/// Relative size at which a series term stops contributing.
const TERM_EPS: f32 = 1e-8;

/// Up to this `|x|` the negative axis is seeded from the series; it is also the march's start.
const SERIES_MAX: f32 = 2.0;

/// At or below this `x` the **positive** axis takes `Ai` and `Ai′` from the series; above it,
/// from the `K` quadrature.
///
/// **Unchanged from the `f64` peer's `1.0`, and that is the measurement's answer.** The threshold
/// is set by the cancellation `Ai = c₁f − c₂g`, which discards about `2ζ/ln 10` *digits* — a
/// count of digits, not a count of ulps, so it consumes the same fraction of either format's
/// budget per unit of `ζ`. What changes is only how many digits there are to lose, and at
/// `x = 1` (`ζ = 0.67`) the loss is under one digit in both.
///
/// The series is still needed *below* this point and is the only thing that works there:
/// `ζ = ⅔x^{3/2}` underflows around `x = 1e-25` in this format, which would leave the quadrature
/// integrating nothing at all.
const AI_SERIES_MAX: f32 = 1.0;

/// Up to this `x` the positive axis keeps taking `Bi` and `Bi′` from the series.
///
/// **`5.0`, where the `f64` peer uses `9.0`**, and the reason is the *scaled* form rather than
/// the unscaled one.
///
/// `Bi` is a sum of positive terms on this side, so none of `Ai`'s cancellation applies and the
/// series itself stays at `1.4e-7` — the format's resolution — all the way out. The cost is not
/// the series but what has to be done to it: the scaled `Biᵉ` is the series value times `e^{−ζ}`,
/// and forming that exponential costs `|dζ| = ulp(ζ)/2` of *relative* error, because
/// `d(e^{−ζ})/e^{−ζ} = −dζ`. At `ζ = 16` — which is `x = 8.3` — that is `9.5e-7`, and nothing
/// about the series can recover it. The `I` asymptotic is *natively* scaled, so taking it sooner
/// avoids the exponential entirely.
///
/// Measured worst relative error of the four scaled forms over `x ∈ (0, 30]` as the handover
/// moves:
///
/// | handover | 2.0 | 3.0 | 4.0 | **5.0** | 6.0 | 6.5 | 9.0 |
/// |---|---|---|---|---|---|---|---|
/// | worst | `5.7e-3` | `1.9e-4` | `2.8e-6` | **`1.1e-6`** | `1.3e-6` | `2.0e-6` | `2.5e-6` |
///
/// Below `4` the asymptotic has not converged and the handover is the binding constraint; above
/// `5` the exponential is. `5.0` is the minimum, and it more than halves `f64`'s choice —
/// `2.5e-6` to `1.1e-6`.
const BI_SERIES_MAX: f32 = 5.0;

/// Up to this `|x|` the negative axis marches the equation outward; beyond it, Hankel.
///
/// **`6.0`, where the `f64` peer uses `9.0`.** The march holds the format's precision
/// indefinitely but costs a step per half-unit, so the crossover is set by where the asymptotic
/// becomes trustworthy — and a divergent expansion truncated at its least term becomes
/// trustworthy *sooner* in a coarser format, because the term it stops at only has to fall below
/// a larger `ε`. The same reasoning moved `J`'s Hankel handover from `18` to `7`.
const MARCH_MAX: f32 = 6.0;

/// At or above this `x` every asymptotic correction term is below rounding, so the leading term
/// alone is used.
///
/// **`1e4`, where the `f64` peer uses `1e10`.** The first correction is `u₁/ζ = (5/72)/ζ`, and
/// `ζ = ⅔x^{3/2}` is `6.7e5` at `x = 1e4`, putting that term at `1.0e-7` — just under this
/// format's half-ulp. Taking this branch also keeps `ζ` out of the arithmetic entirely: it
/// overflows above `x ≈ 1.4e25` here rather than `f64`'s `3e205`, and the leading forms are
/// exactly the ones that stay representable.
const LEADING_MIN: f32 = 1e4;

/// Largest march step. The `f64` peer uses the same.
const MAX_STEP: f32 = 0.5;

/// Taylor coefficients per march step. The `f64` peer uses 24.
///
/// The recurrence `cₙ₊₂ = (x₀cₙ + cₙ₋₁)/((n+2)(n+1))` divides by a factorial-like growth, so the
/// coefficients fall off faster than `h^n/n!` with `h ≤ 0.5`. **12** puts the first dropped term
/// near `1e-13`, far under this format's resolution; each step builds an array of this length, so
/// it is the march's whole cost.
const MARCH_ORDER: usize = 12;

/// `√3`, narrowed from the single definition in [`crate::math::airy::SQRT_3`] so this format
/// cannot drift from it.
const SQRT_3: f32 = crate::math::airy::SQRT_3 as f32;

/// `ζ = ⅔|x|^{3/2}`, as `⅔·|x|·√|x|` rather than through `powf`.
fn zeta(ax: f32) -> f32 {
    (2.0 / 3.0) * ax * ax.sqrt()
}

/// The two entire solutions of `y″ = xy` and their derivatives, from the Maclaurin series.
///
/// Neither Pochhammer symbol nor factorial is ever formed: dividing consecutive terms collapses
/// both into a two-factor denominator, leaving four plain recurrences in `x³`.
fn fg(x: f32) -> (f32, f32, f32, f32) {
    let x3 = x * x * x;
    let (mut a, mut b, mut p, mut q) = (1.0f32, x, x * x * 0.5, 1.0f32);
    let (mut f, mut g, mut f_prime, mut g_prime) = (a, b, p, q);
    for k in 1..MAX_TERMS {
        let k3 = 3.0 * k as f32;
        a *= x3 / (k3 * (k3 - 1.0));
        b *= x3 / (k3 * (k3 + 1.0));
        q *= x3 / (k3 * (k3 - 2.0));
        f += a;
        g += b;
        g_prime += q;
        // `p` starts at k = 1 with x²/2 rather than at k = 0, so its recurrence begins a step
        // later than the other three.
        if k >= 2 {
            p *= x3 / ((k3 - 1.0) * (k3 - 3.0));
            f_prime += p;
        }
        let largest = a.abs().max(b.abs()).max(p.abs()).max(q.abs());
        let scale = f.abs().max(g.abs()).max(f_prime.abs()).max(g_prime.abs());
        if term_is_negligible(largest, scale, TERM_EPS) {
            break;
        }
    }
    (f, g, f_prime, g_prime)
}

/// `(Ai, Ai′, Bi, Bi′)` at `x`, from the series.
fn series_airy(x: f32) -> (f32, f32, f32, f32) {
    let (f, g, f_prime, g_prime) = fg(x);
    crate::math::airy::combine(f, g, f_prime, g_prime)
}

/// One Taylor step of `y″ = xy`, carrying both the value and the derivative.
/// Two of this function's index arithmetic mutants are equivalent, and the reason is
/// [`MARCH_ORDER`]'s own measurement. Widening the coefficient buffer changes nothing — the loops
/// below are bounded by the order, not the array. Narrowing the Horner bound drops the top one or
/// two coefficients, which that constant records as "near `1e-13`, far under this format's
/// resolution": `c₁₂h¹²` at the largest step is below an ulp of the value it is added to.
fn march_step(x0: f32, y: f32, y_prime: f32, h: f32) -> (f32, f32) {
    let mut c = [0.0f32; MARCH_ORDER + 2];
    c[0] = y;
    c[1] = y_prime;
    for n in 0..MARCH_ORDER {
        let previous = if n >= 1 { c[n - 1] } else { 0.0 };
        c[n + 2] = (x0 * c[n] + previous) / (((n + 2) * (n + 1)) as f32);
    }
    // Horner, from the top down, for the value and for the derivative `Σ n·cₙh^{n−1}`.
    let mut value = 0.0f32;
    let mut derivative = 0.0f32;
    for n in (0..=MARCH_ORDER + 1).rev() {
        value = value * h + c[n];
        if n >= 1 {
            derivative = derivative * h + (n as f32) * c[n];
        }
    }
    (value, derivative)
}

/// Transport a solution and its derivative from `x_start` to `x_end` in equal steps.
fn transport(x_start: f32, x_end: f32, y: f32, y_prime: f32) -> (f32, f32) {
    let span = x_end - x_start;
    let count = (span.abs() / MAX_STEP).ceil().max(1.0);
    let h = span / count;
    let (mut value, mut derivative) = (y, y_prime);
    let mut x = x_start;
    let mut taken = 0.0f32;
    while taken < count {
        (value, derivative) = march_step(x, value, derivative, h);
        x += h;
        taken += 1.0;
    }
    (value, derivative)
}

/// `(Ai, Ai′, Bi, Bi′)` on the negative axis, by marching out from the series.
fn march_airy(x: f32) -> (f32, f32, f32, f32) {
    let start = -SERIES_MAX;
    let (f0, g0, f0_prime, g0_prime) = fg(start);
    let (f, f_prime) = transport(start, x, f0, f0_prime);
    let (g, g_prime) = transport(start, x, g0, g0_prime);
    crate::math::airy::combine(f, g, f_prime, g_prime)
}

/// `(Aiᵉ, Aiᵉ′)` from the `K_{1/3}`, `K_{2/3}` quadrature, for `x > `[`AI_SERIES_MAX`].
///
/// Naturally scaled: [`k_nu_pair_scaled`] *is* `e^{ζ}K_ν(ζ)`, so no exponential is formed here at
/// all and the result is finite however large `x` becomes.
///
/// One sweep of nodes for both orders, as the `f64` peer does: the nodes depend only on `z`.
fn ai_pair_scaled(x: f32, z: f32) -> Slope<f32> {
    let (k_third, k_two_thirds) = k_nu_pair_scaled(1.0 / 3.0, 2.0 / 3.0, z);
    Slope::new(
        (x / 3.0).sqrt() * k_third / PI,
        -(x / SQRT_3) * k_two_thirds / PI,
    )
}

/// `(Biᵉ, Biᵉ′)` from the scaled `I` asymptotic, for `x > `[`BI_SERIES_MAX`].
///
/// The asymptotic uses only the square of the order, so each `I₋ν + Iν` contributes twice the
/// value at positive order.
fn bi_pair_scaled(x: f32, z: f32) -> Slope<f32> {
    Slope::new(
        (x / 3.0).sqrt() * (2.0 * i_scaled_asymptotic(1.0 / 3.0, z)),
        (x / SQRT_3) * (2.0 * i_scaled_asymptotic(2.0 / 3.0, z)),
    )
}

/// `(Ai, Ai′, Bi, Bi′)` on the negative axis beyond [`MARCH_MAX`], from the Hankel expansion.
fn negative_asymptotic(ax: f32) -> (f32, f32, f32, f32) {
    let z = zeta(ax);
    if !z.is_finite() {
        // Past `|x| ≈ 1.4e25` the phase is not merely inaccurate but unrepresentable: `ζ` itself
        // has overflowed, so there is no argument to hand to `sin`. Zero is finite, inside the
        // envelope, and the mean of the oscillation — the honest answer once the phase is gone.
        // Accuracy departs far earlier, around `|x| ≈ 1e5`, where `ulp(ζ)` passes `2π`.
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

/// Which pair of routes the **positive** axis uses.
///
/// The two functions cross over at different arguments — `Ai` at [`AI_SERIES_MAX`] because it
/// cancels, `Bi` at the much later [`BI_SERIES_MAX`] because it does not — so this names the
/// combination rather than one method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PositiveRegime {
    /// Both from the series.
    BothSeries,
    /// `Ai` from the `K` quadrature, `Bi` still from the series.
    AiQuadrature,
    /// Both from their Bessel asymptotics.
    BothBessel,
}

impl PositiveRegime {
    fn for_argument(x: f32) -> Self {
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
fn positive_scaled(x: f32, z: f32) -> (f32, f32, f32, f32) {
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
    /// The non-finite cases first, then the sign, then the two magnitude ladders.
    fn for_argument(x: f32) -> Self {
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

fn quad(x: f32) -> (f32, f32, f32, f32) {
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
fn positive_unscaled(x: f32) -> (f32, f32, f32, f32) {
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
fn quad_scaled(x: f32) -> (f32, f32, f32, f32) {
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

/// The Airy function of the first kind `Ai(x)`.
pub(crate) fn ai(x: f32) -> f32 {
    quad(x).0
}

/// `Ai′(x)`.
pub(crate) fn ai_prime(x: f32) -> f32 {
    quad(x).1
}

/// The Airy function of the second kind `Bi(x)`.
pub(crate) fn bi(x: f32) -> f32 {
    quad(x).2
}

/// `Bi′(x)`.
pub(crate) fn bi_prime(x: f32) -> f32 {
    quad(x).3
}

/// The scaled `Aiᵉ(x) = e^{ζ}Ai(x)` for `x ≥ 0`, and `Ai(x)` below.
pub(crate) fn aie(x: f32) -> f32 {
    quad_scaled(x).0
}

/// The scaled `Aiᵉ′(x)`.
pub(crate) fn aie_prime(x: f32) -> f32 {
    quad_scaled(x).1
}

/// The scaled `Biᵉ(x) = e^{−ζ}Bi(x)` for `x ≥ 0`, and `Bi(x)` below.
pub(crate) fn bie(x: f32) -> f32 {
    quad_scaled(x).2
}

/// The scaled `Biᵉ′(x)`.
pub(crate) fn bie_prime(x: f32) -> f32 {
    quad_scaled(x).3
}
