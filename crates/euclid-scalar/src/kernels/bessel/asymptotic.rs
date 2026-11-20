//! The large-argument asymptotic expansions, shared by every member of the family.
//!
//! All four kinds are the *same* series read through different factors. Writing
//!
//! ```text
//! a₀(ν) = 1,    a_k(ν) = a_{k−1}(ν)·(4ν² − (2k−1)²)/(8k)
//! ```
//!
//! the expansions are
//!
//! ```text
//! Jν ~ √(2/πx)[cos ω·P − sin ω·Q]      Iνᵉ ~ (2πx)^{−1/2} Σ (−1)^k a_k/x^k
//! Yν ~ √(2/πx)[sin ω·P + cos ω·Q]      Kνᵉ ~ √(π/2x)      Σ        a_k/x^k
//! ```
//!
//! with `ω = x − (ν/2 + ¼)π`, `P` the even-indexed terms of the alternating sum and `Q` the
//! odd-indexed ones. So one coefficient recurrence serves the whole family, and nothing is
//! transcribed anywhere.
//!
//! Every one of these series is **divergent**. Each is truncated at its minimum term, because
//! continuing past that point makes the result worse rather than better — which is what the
//! growth check in [`step_coefficient`]'s callers watches for.

use crate::constants::f64::SQRT_TAU;
use crate::math::arithmetic::term_is_negligible;
use crate::math::pair::{Converged, Solutions};
use core::f64::consts::PI;

/// Term cap for the expansions; the minimum-term check normally stops them far sooner.
pub(crate) const MAX_TERMS: u32 = 200;

/// Relative size at which a term stops contributing.
///
/// **`1e-19`, which is deliberately only a little below `f64`'s resolution.** `ε = 2.22e-16`, so
/// a term at `1e-19` of the sum moves the result by about a thousandth of an ulp, and every term
/// after it moves it by less. The value was `1e-22`, which asked the Hankel and scaled-`I` expansions, shared by `J`, `Y`, `I` and Airy to converge
/// three orders past the point where the format can represent the difference —
/// iterations bought and then discarded.
///
/// Raising it to `1e-19` is **free rather than a trade.** The outputs are bit-identical across a
/// 306,280-point sweep covering every kernel in this tree at both signs, while contributing to the gains recorded in the sibling modules, and about **2%** of Airy's.
///
/// Timings are the minimum of six interleaved A/B runs of two prebuilt binaries pinned to idle
/// cores, which cancels drift — this machine is shared, and a naive before/after on it reported
/// swings of 2x on *identical* code.
///
/// `1e-18` was measured too, and is another 1-3% faster, but it is no longer bit-identical: it
/// differs by up to 3 ulp on `Bi`. So `1e-19` is the last value that costs nothing at all, and
/// the margin over `ε` is what pays for a series whose tail is amplified by `largest/sum` where
/// it cancels.
const TERM_EPS: f64 = 1e-19;

/// Advance the coefficient: `a_k = a_{k−1}·(4ν² − (2k−1)²)/(8k)`.
fn step_coefficient(a: f64, four_nu_sq: f64, k: f64) -> f64 {
    let odd = 2.0 * k - 1.0;
    a * (four_nu_sq - odd * odd) / (8.0 * k)
}

/// `Iνᵉ(x) = e^{−x}Iν(x)` from its asymptotic expansion, for large `x`.
///
/// ```text
/// Iνᵉ(x) ~ (2πx)^{−1/2} Σ (−1)^k a_k(ν)/x^k,   a_k(ν) = a_{k−1}·(4ν² − (2k−1)²)/(8k)
/// ```
///
/// The same coefficients the `K` expansion uses, with the signs alternating — so nothing is
/// transcribed here either. Divergent, and truncated at its minimum term; continuing past it
/// makes the result worse, which the growth check watches for.
pub(crate) fn i_scaled(nu: f64, x: f64) -> f64 {
    i_scaled_terms(nu, x).value
}

/// `Iνᵉ`'s expansion, with the number of terms it took.
///
/// Instrumented because the two exits are otherwise indistinguishable. The expansion is
/// divergent, so `magnitude >= previous` always fires eventually and the second exit —
/// `magnitude < TERM_EPS`, for an argument large enough that the series reaches the format's
/// floor first — only ever saves terms. Mutation testing finds every spelling of it unkillable
/// for exactly that reason: weaken it and the growth check still stops the loop at the same
/// value, having summed more terms to get there. The count is what separates them.
pub(crate) fn i_scaled_terms(nu: f64, x: f64) -> Converged<f64> {
    debug_assert!(x > 0.0);
    let mut terms = 0;
    let four_nu_sq = 4.0 * nu * nu;
    let mut a = 1.0;
    let mut inv_power = 1.0;
    let mut total = 0.0;
    let mut previous = f64::INFINITY;
    for m in 0..MAX_TERMS {
        if m > 0 {
            a = step_coefficient(a, four_nu_sq, f64::from(m));
            inv_power /= x;
        }
        let magnitude = (a * inv_power).abs();
        if magnitude >= previous {
            break; // past the minimum term of a divergent expansion
        }
        previous = magnitude;
        total += if m % 2 == 1 {
            -a * inv_power
        } else {
            a * inv_power
        };
        terms += 1;
        if term_is_negligible(magnitude, 1.0, TERM_EPS) {
            break;
        }
    }
    // `√(2πx)` as `√(2π)·√x`, not `√(2πx)`: the product `2πx` overflows while the *result*
    // is still an entirely ordinary number. In `f32` that cost `I₀ᵉ(1e38)` its value
    // outright — `2π·1e38` is past `f32::MAX`, so the square root was infinite and the
    // quotient zero, where the true `4e-20` is perfectly representable. `f64` had the same
    // latent failure above `x = 2.9e307`. Splitting the root keeps every intermediate in
    // range and is exact to the same rounding.
    Converged::new(total / (SQRT_TAU * x.sqrt()), terms)
}

/// The Hankel expansion's two coefficient sums, with the number of terms they took.
///
/// ```text
/// P = Σ (−1)^k a_{2k}(ν)/x^{2k}            Q = Σ (−1)^k a_{2k+1}(ν)/x^{2k+1}
/// ```
///
/// with `a₀ = 1` and `a_k(ν) = a_{k−1}·(4ν² − (2k−1)²)/(8k)`, so the coefficients are generated
/// in the loop and nothing is transcribed. Both series are divergent and are truncated at their
/// minimum term; continuing past it makes them worse, which the growth check watches for.
///
/// Extracted from [`hankel`] because the loop's second exit was otherwise unobservable. `P` and
/// `Q` are summed together — the terms alternate between them by parity — and the loop has two
/// ways out: the growth check, which always fires because the expansion is divergent, and
/// `magnitude < TERM_EPS`, which fires first for an argument large enough that the terms reach
/// the format's floor before they turn. Weaken the second and the first still stops the loop at
/// the same `P` and `Q`, having summed more terms to get there, so `hankel`'s own output cannot
/// see the difference. The term count can, and that is the whole reason this is a function.
pub(crate) fn hankel_sums(nu: f64, x: f64) -> Converged<Solutions<f64>> {
    debug_assert!(x > 0.0);
    let four_nu_sq = 4.0 * nu * nu;
    let mut a = 1.0f64;
    let mut inv_power = 1.0f64; // x^{-m}
    let (mut p, mut q) = (0.0f64, 0.0f64);
    let mut previous = f64::INFINITY;
    let mut terms = 0;

    for m in 0..MAX_TERMS {
        if m > 0 {
            a = step_coefficient(a, four_nu_sq, f64::from(m));
            inv_power /= x;
        }
        let magnitude = (a * inv_power).abs();
        if magnitude >= previous {
            break; // past the minimum term of a divergent expansion
        }
        previous = magnitude;
        // Sign is (-1)^{m/2} for the even (P) terms and (-1)^{(m-1)/2} for the odd (Q) ones.
        let signed = if (m / 2) % 2 == 0 {
            a * inv_power
        } else {
            -a * inv_power
        };
        if m % 2 == 0 {
            p += signed;
        } else {
            q += signed;
        }
        terms += 1;
        if term_is_negligible(magnitude, 1.0, TERM_EPS) {
            break;
        }
    }
    Converged::new(Solutions::new(p, q), terms)
}

/// `(Jν(x), Yν(x))` from the Hankel asymptotic expansion, for large `x`.
///
/// ```text
/// Jν ~ √(2/πx)[cos ω · P − sin ω · Q]      Yν ~ √(2/πx)[sin ω · P + cos ω · Q]
/// ω = x − (ν/2 + ¼)π
/// ```
///
/// where `P` and `Q` come from [`hankel_sums`]. Returning the pair together costs nothing —
/// `P`, `Q` and `ω` are all shared between the two — and keeps `J` and `Y` consistent in the
/// tail, which is what the Wronskian test checks.
pub(crate) fn hankel(nu: f64, x: f64) -> Solutions<f64> {
    let sums = hankel_sums(nu, x).value;
    let (p, q) = (sums.first_kind, sums.second_kind);
    // The phase is `ω = x − (ν/2 + ¼)π`, and **it must not be formed as a single `f64`.**
    //
    // `x` is exact; `x − c` rounds to a multiple of `ulp(x)`, and that rounding *is* the angle's
    // absolute error. An oscillating function then returns it as that fraction of its envelope,
    // so the damage grows like `ulp(x)·√x`: half an ulp of phase against an amplitude of
    // `√(2/πx)` gives `ε√(2x/π)/2` of absolute error.
    //
    // This function formed `ω` directly until it was measured against MPFR at 256 bits over a
    // log-uniform sweep of the whole range. The surface promises "machine precision in absolute
    // terms at every argument (measured below `1e-15`)", and the direct form broke that from
    // `x = 100` upward:
    //
    // ```text
    // x        1e2      1e3      1e4      1e6      1e9      1e11     1e13
    // was    1.2e-15  4.5e-15  7.8e-15  1.6e-13  3.8e-12  4.7e-12  4.1e-10
    // ```
    //
    // The single-precision peer has always used the identities below, and its note explains why
    // in terms of `ulp(x)` at `x = 227` — adding that "the `f64` peer forms `ω` directly and can
    // afford to", since `ulp(227)` is `3e-14` in this format. That comparison is sound at that
    // argument and does not generalise: `ulp(x)` is linear in `x`, so by `x = 1e13` it is
    // `2e-3` radians here. The justification held exactly where it was taken.
    //
    // ```text
    // sin ω = sin x·cos c − cos x·sin c,    cos ω = cos x·cos c + sin x·sin c
    // ```
    //
    // `sin x` and `cos x` are evaluated at the *exact* argument, where the library's own
    // reduction keeps them to an ulp however large `x` is, and the shift is applied afterwards
    // through constants that are `O(1)`.
    let (sin_x, cos_x) = x.sin_cos();
    // `c = (ν/2 + ¼)π` is `O(1)` whatever `ν` is, so its sine and cosine are accurate to an ulp
    // computed directly — the precision problem is `x`, never the shift. Evaluated for general
    // `ν` rather than special-cased to `|cos c| = |sin c| = √2/2` at `ν ∈ {0, 1}`, so the Airy
    // family reaches it through `ν = ±⅓, ±⅔` as well.
    let (sin_c, cos_c) = ((nu * 0.5 + 0.25) * PI).sin_cos();
    let sin_omega = sin_x * cos_c - cos_x * sin_c;
    let cos_omega = cos_x * cos_c + sin_x * sin_c;
    let amplitude = (2.0 / (PI * x)).sqrt();
    Solutions::new(
        amplitude * (cos_omega * p - sin_omega * q),
        amplitude * (sin_omega * p + cos_omega * q),
    )
}
