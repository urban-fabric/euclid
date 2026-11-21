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
use crate::math::pair::{Slope, Solutions};
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
/// overflows above `x ≈ 6.39e25` here rather than `f64`'s `4.17e205`, and the leading forms are
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

/// The two entire solutions of `y″ = xy`, each with its derivative, from the Maclaurin series.
///
/// Neither Pochhammer symbol nor factorial is ever formed: dividing consecutive terms collapses
/// both into a two-factor denominator, leaving four plain recurrences in `x³`.
///
/// Grouped by *solution* rather than by derivative order, as the `f64` peer is: the march
/// transports `(f, f′)` and `(g, g′)` independently, so that pairing is the one the caller needs.
fn fg(x: f32) -> Solutions<Slope<f32>> {
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
    Solutions::new(Slope::new(f, f_prime), Slope::new(g, g_prime))
}

/// The Airy pair and its derivatives at `x`, from the series.
fn series_airy(x: f32) -> Solutions<Slope<f32>> {
    crate::math::airy::combine(fg(x))
}

/// One Taylor step of `y″ = xy`, carrying both the value and the derivative.
/// Two of this function's index arithmetic mutants are equivalent, and the reason is
/// [`MARCH_ORDER`]'s own measurement. Widening the coefficient buffer changes nothing — the loops
/// below are bounded by the order, not the array. Narrowing the Horner bound drops the top one or
/// two coefficients, which that constant records as "near `1e-13`, far under this format's
/// resolution": `c₁₂h¹²` at the largest step is below an ulp of the value it is added to.
fn march_step(x0: f32, at: Slope<f32>, h: f32) -> Slope<f32> {
    let mut c = [0.0f32; MARCH_ORDER + 2];
    c[0] = at.value;
    c[1] = at.derivative;
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
    Slope::new(value, derivative)
}

/// Transport a solution and its derivative from `x_start` to `x_end` in equal steps.
fn transport(x_start: f32, x_end: f32, from: Slope<f32>) -> Slope<f32> {
    let span = x_end - x_start;
    let count = (span.abs() / MAX_STEP).ceil().max(1.0);
    let h = span / count;
    let mut at = from;
    let mut x = x_start;
    let mut taken = 0.0f32;
    while taken < count {
        at = march_step(x, at, h);
        x += h;
        taken += 1.0;
    }
    at
}

/// The Airy pair and its derivatives on the negative axis, by marching out from the series.
///
/// Each half is a complete solution of the equation, so the two are transported independently
/// and combined only at the end.
fn march_airy(x: f32) -> Solutions<Slope<f32>> {
    let start = -SERIES_MAX;
    let seed = fg(start);
    crate::math::airy::combine(Solutions::new(
        transport(start, x, seed.first_kind),
        transport(start, x, seed.second_kind),
    ))
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

mod regimes;

// The parent carves the public pairs out of `quad`, so it needs both evaluations back.
use regimes::{quad, quad_scaled};

/// The Airy function of the first kind `Ai(x)`.
pub(crate) fn ai(x: f32) -> f32 {
    quad(x).first_kind.value
}

/// `Ai′(x)`.
pub(crate) fn ai_prime(x: f32) -> f32 {
    quad(x).first_kind.derivative
}

/// The Airy function of the second kind `Bi(x)`.
pub(crate) fn bi(x: f32) -> f32 {
    quad(x).second_kind.value
}

/// `Bi′(x)`.
pub(crate) fn bi_prime(x: f32) -> f32 {
    quad(x).second_kind.derivative
}

/// The scaled `Aiᵉ(x) = e^{ζ}Ai(x)` for `x ≥ 0`, and `Ai(x)` below.
pub(crate) fn aie(x: f32) -> f32 {
    quad_scaled(x).first_kind.value
}

/// The scaled `Aiᵉ′(x)`.
pub(crate) fn aie_prime(x: f32) -> f32 {
    quad_scaled(x).first_kind.derivative
}

/// The scaled `Biᵉ(x) = e^{−ζ}Bi(x)` for `x ≥ 0`, and `Bi(x)` below.
pub(crate) fn bie(x: f32) -> f32 {
    quad_scaled(x).second_kind.value
}

/// The scaled `Biᵉ′(x)`.
pub(crate) fn bie_prime(x: f32) -> f32 {
    quad_scaled(x).second_kind.derivative
}
