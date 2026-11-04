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
use core::f64::consts::PI;

/// Term cap for the expansions; the minimum-term check normally stops them far sooner.
const MAX_TERMS: u32 = 200;

/// Relative size at which a term stops contributing.
const TERM_EPS: f64 = 1e-22;

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
    debug_assert!(x > 0.0);
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
        if magnitude < TERM_EPS {
            break;
        }
    }
    // `√(2πx)` as `√(2π)·√x`, not `√(2πx)`: the product `2πx` overflows while the *result*
    // is still an entirely ordinary number. In `f32` that cost `I₀ᵉ(1e38)` its value
    // outright — `2π·1e38` is past `f32::MAX`, so the square root was infinite and the
    // quotient zero, where the true `4e-20` is perfectly representable. `f64` had the same
    // latent failure above `x = 2.9e307`. Splitting the root keeps every intermediate in
    // range and is exact to the same rounding.
    total / (SQRT_TAU * x.sqrt())
}

/// `(Jν(x), Yν(x))` from the Hankel asymptotic expansion, for large `x`.
///
/// ```text
/// Jν ~ √(2/πx)[cos ω · P − sin ω · Q]      Yν ~ √(2/πx)[sin ω · P + cos ω · Q]
/// ω = x − (ν/2 + ¼)π
/// P = Σ (−1)^k a_{2k}(ν)/x^{2k}            Q = Σ (−1)^k a_{2k+1}(ν)/x^{2k+1}
/// ```
///
/// with `a₀ = 1` and `a_k(ν) = a_{k−1}·(4ν² − (2k−1)²)/(8k)`, so the coefficients are
/// generated in the loop and nothing is transcribed.
///
/// Both series are divergent and are truncated at their minimum term; continuing past it makes
/// them worse, which the growth check watches for. Returning the pair together costs nothing —
/// `P`, `Q` and `ω` are shared — and keeps `J` and `Y` consistent in the tail, which is what
/// the Wronskian test checks.
pub(crate) fn hankel(nu: f64, x: f64) -> (f64, f64) {
    debug_assert!(x > 0.0);
    let four_nu_sq = 4.0 * nu * nu;
    let mut a = 1.0f64;
    let mut inv_power = 1.0f64; // x^{-m}
    let (mut p, mut q) = (0.0f64, 0.0f64);
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
        if magnitude < TERM_EPS {
            break;
        }
    }

    let omega = x - (nu * 0.5 + 0.25) * PI;
    let (sin_omega, cos_omega) = omega.sin_cos();
    let amplitude = (2.0 / (PI * x)).sqrt();
    (
        amplitude * (cos_omega * p - sin_omega * q),
        amplitude * (sin_omega * p + cos_omega * q),
    )
}
