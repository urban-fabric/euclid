//! Single-precision peer of [`super::super::bessel`]. Same peak-factored power series; see
//! that module for why it replaced the Abramowitz & Stegun polynomials.
//!
//! The term cutoff is loosened to the `f32` noise floor, which shortens the outward walk.

use crate::constants::f32::SQRT_TAU;
use crate::kernels::f32::gamma::ln_gamma;
use core::f32::consts::TAU;

/// Cap on the outward walk in either direction from the peak. The significant terms span
/// `O(√x)` indices, so this covers arguments past `1e8` before it binds.
const MAX_TERMS: u32 = 100_000;

/// Relative size at which an outward term stops contributing.
const TERM_EPS: f32 = 1e-12;

/// The `I₀` series summed **relative to its peak term** at index `k_peak`, so the result is
/// `O(√x)` and cannot overflow. Shared by `ln_i0_sum` and the ratio.
fn i0_rel_sum(h: f32, k_peak: f32) -> f32 {
    let mut sum = 1.0f32;

    // Upward: t_{k+1}/t_k = h^2/(k+1)^2.
    let mut ratio = 1.0f32;
    let mut k = k_peak;
    for _ in 0..MAX_TERMS {
        let step = h / (k + 1.0);
        ratio *= step * step;
        sum += ratio;
        k += 1.0;
        if ratio < sum * TERM_EPS {
            break;
        }
    }

    // Downward: t_{k-1}/t_k = k^2/h^2.
    ratio = 1.0;
    k = k_peak;
    for _ in 0..MAX_TERMS {
        if k < 1.0 {
            break;
        }
        let step = k / h;
        ratio *= step * step;
        k -= 1.0;
        sum += ratio;
        if ratio < sum * TERM_EPS {
            break;
        }
    }

    sum
}

/// Above this peak index the Stirling remainder comes from its asymptotic series; below it,
/// from `ln_gamma` directly.
///
/// Lower than the `f64` threshold of 20, because single precision needs far less of the tail:
/// the first omitted term, `1/(1260k⁵)`, is already `2.4e-8` at `k = 8`.
const STIRLING_ASYMPTOTIC_MIN: f32 = 8.0;

/// The Stirling remainder `μ(k) = lnΓ(k+1) − (k ln k − k + ½ln(2πk))`.
///
/// Needed in its own right rather than as a difference — see
/// [the `f64` peer](crate::kernels::bessel::modified_first) for the argument; it applies with
/// more force here, since `f32` has only seven digits to lose.
fn stirling_remainder(k: f32) -> f32 {
    if k >= STIRLING_ASYMPTOTIC_MIN {
        let inv = 1.0 / k;
        let inv2 = inv * inv;
        inv * (1.0 / 12.0 + inv2 * (-1.0 / 360.0 + inv2 / 1260.0))
    } else {
        // Small k: the defining difference is between O(1) quantities, so it is safe, and the
        // asymptotic series would not yet have converged.
        ln_gamma(k + 1.0) - (k * k.ln() - k + 0.5 * (TAU * k).ln())
    }
}

/// `2k·ln h − 2·lnΓ(k+1) − 2h`, the shared log-domain core of both scaled peak terms.
///
/// Written so the large parts cancel symbolically:
///
/// ```text
/// 2k ln h − 2lnΓ(k+1) − 2h = 2k·ln(h/k) + 2(k − h) − ln(2πk) − 2μ(k)
/// ```
///
/// The literal form subtracts two `O(k ln k)` quantities, which in single precision is ruinous
/// rather than merely lossy: at `x = 1e6` both terms are about `1.1e6`, so `ε·1.1e6 ≈ 0.13` of
/// absolute error survives into the exponent — measured as **4.7% relative error in `I₀ᵉ` and
/// 42% in `I₁ᵉ`**, against a surface that claims to track `f64` within an ulp. The error is
/// invisible to a mixed absolute/relative test because `I₀ᵉ` is itself small.
fn ln_peak_core_scaled(h: f32, k: f32) -> f32 {
    debug_assert!(k >= 1.0);
    2.0 * k * ((h - k) / k).ln_1p() + 2.0 * (k - h) - (TAU * k).ln() - 2.0 * stirling_remainder(k)
}

/// `ln I₀(|x|) − |x|`, the scaled log from which every `I₀` form is derived.
fn ln_i0_minus_x(ax: f32) -> f32 {
    debug_assert!(ax >= 0.0);
    if ax == 0.0 {
        return 0.0; // I0(0) = 1
    }
    if ax >= I_ASYMPTOTIC_MIN {
        return i_scaled_asymptotic(0.0, ax).ln();
    }
    let h = ax * 0.5;
    let k_peak = h.round().max(0.0);
    if k_peak < 1.0 {
        // The peak is the k = 0 term, which is exactly 1, so `ln t_peak` is exactly 0.
        //
        // Spelling that out rather than letting the general expression produce it, because
        // `2k·ln h` is `0 × −∞ = NaN` once `h = x/2` underflows — which it does at the
        // smallest `f32` subnormal, taking `I₀`, `I₀ᵉ` and `ln I₀` to `NaN` with it.
        return i0_rel_sum(h, k_peak).ln() - ax;
    }
    ln_peak_core_scaled(h, k_peak) + i0_rel_sum(h, k_peak).ln()
}

/// `ln I₀(|x|)`.
fn ln_i0_sum(ax: f32) -> f32 {
    ln_i0_minus_x(ax) + ax
}

/// The `I₁` series summed relative to its peak term at the same index.
fn i1_rel_sum(h: f32, k_peak: f32) -> f32 {
    let mut sum = 1.0f32;

    // Upward: t_{k+1}/t_k = h^2/((k+1)(k+2)).
    let mut ratio = 1.0f32;
    let mut k = k_peak;
    for _ in 0..MAX_TERMS {
        ratio *= (h / (k + 1.0)) * (h / (k + 2.0));
        sum += ratio;
        k += 1.0;
        if ratio < sum * TERM_EPS {
            break;
        }
    }

    // Downward: t_{k-1}/t_k = k(k+1)/h^2.
    ratio = 1.0;
    k = k_peak;
    for _ in 0..MAX_TERMS {
        if k < 1.0 {
            break;
        }
        ratio *= (k / h) * ((k + 1.0) / h);
        k -= 1.0;
        sum += ratio;
        if ratio < sum * TERM_EPS {
            break;
        }
    }

    sum
}

/// `ln I₁(|x|) − |x|`.
///
/// `I₁`'s peak term shares `I₀`'s core: since `lnΓ(k+2) = lnΓ(k+1) + ln(k+1)`,
///
/// ```text
/// ln tₖ − 2h = ln h − ln(k+1) + [2k ln h − 2lnΓ(k+1) − 2h]
/// ```
fn ln_i1_minus_x(ax: f32) -> f32 {
    debug_assert!(ax >= 0.0);
    if ax == 0.0 {
        return f32::NEG_INFINITY; // I1(0) = 0
    }
    let h = ax * 0.5;
    let k_peak = h.round().max(0.0);
    if k_peak < 1.0 {
        return h.ln() + i1_rel_sum(h, k_peak).ln() - ax;
    }
    h.ln() - (k_peak + 1.0).ln() + ln_peak_core_scaled(h, k_peak) + i1_rel_sum(h, k_peak).ln()
}

/// `ln I₁(|x|)`.
fn ln_i1_sum(ax: f32) -> f32 {
    ln_i1_minus_x(ax) + ax
}

/// At or above this argument the scaled `I` functions use their asymptotic expansion rather
/// than the peak-factored series.
///
/// This closes a hole rather than saving time. The series walks `O(√x)` terms outward from the
/// peak, and two things go wrong as that count grows: the naive accumulation drifts like
/// `√n·ε`, and past `MAX_TERMS` the walk is simply cut short. Measured against the asymptotic
/// form, `I₀ᵉ` carried
///
/// ```text
/// x = 1e8   4.9e-14        x = 1e10   4.5e-2
/// x = 1e9   2.5e-10        x = 1e12   8.4e-1
/// ```
///
/// — the last two being truncation, not rounding, and silently finite.
///
/// `1e4` is where the expansion becomes the better of the two: its first omitted term is
/// `|a₄|/x⁴ ≈ 1.1e-17`, below half an ulp, while the series still has only ~200 terms there and
/// so has drifted by less than `1e-15`. Below the threshold the series is never long enough to
/// be a problem; above it, it is never used.
const I_ASYMPTOTIC_MIN: f32 = 1e2;

/// `Iνᵉ(x) = e^{−x}Iν(x)` from its asymptotic expansion, for large `x`.
///
/// ```text
/// Iνᵉ(x) ~ (2πx)^{−1/2} Σ (−1)^k a_k(ν)/x^k,   a_k(ν) = a_{k−1}·(4ν² − (2k−1)²)/(8k)
/// ```
///
/// The same coefficients the `K` expansion uses, with the signs alternating — so nothing is
/// transcribed here either. Divergent, and truncated at its minimum term; continuing past it
/// makes the result worse, which the growth check watches for.
fn i_scaled_asymptotic(nu: f32, x: f32) -> f32 {
    debug_assert!(x > 0.0);
    let four_nu_sq = 4.0 * nu * nu;
    let mut a = 1.0;
    let mut inv_power = 1.0;
    let mut total = 0.0;
    let mut previous = f32::INFINITY;
    for m in 0..ASYMPTOTIC_TERMS {
        if m > 0 {
            let mf = m as f32;
            let odd = 2.0 * mf - 1.0;
            a *= (four_nu_sq - odd * odd) / (8.0 * mf);
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
        if magnitude < ASYMPTOTIC_EPS {
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

/// Term cap for [`i_scaled_asymptotic`]; the minimum-term check normally stops it far sooner.
const ASYMPTOTIC_TERMS: u32 = 200;

/// Relative size at which an asymptotic term stops contributing.
const ASYMPTOTIC_EPS: f32 = 1e-12;

/// The modified Bessel function `I₀(x)` (even in `x`).
///
/// Overflows for `|x| ≳ 710`, as the true value does; use [`i0e`] or [`ln_i0`] there.
pub(crate) fn i0(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    ln_i0_sum(x.abs()).exp()
}

/// Below this argument the `k = 0` term is the peak, and `I₁` is formed without a logarithm.
///
/// `k_peak = round(x/2)` is zero exactly when `x < 1`, and there `t₀ = x/2` precisely — so
/// `I₁ = (x/2)·Σ` needs no `exp(ln t₀ + ⋯)` round trip at all.
///
/// Worth its own path because that round trip is lossy for small `x`: `ln(x/2)` is `−691` at
/// `x = 1e-300`, carrying `ε·691 ≈ 1.5e-13` of *absolute* error, which the final `exp` converts
/// into the same amount of *relative* error. Measured 3.5e-14 at `x = 1e-300` before this
/// existed — worse than the 1.7e-14 the surface documents — while `I₁(x) ~ x/2` is a perfectly
/// conditioned limit that ought to come back exact.
const I1_NO_LOG_MAX: f32 = 1.0;

/// `I₁(|x|)` for `|x| < I1_NO_LOG_MAX`, straight from the series with no logarithm.
fn i1_no_log(h: f32) -> f32 {
    h * i1_rel_sum(h, 0.0)
}

/// The modified Bessel function `I₁(x)` (odd in `x`).
pub(crate) fn i1(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let ax = x.abs();
    let magnitude = if ax < I1_NO_LOG_MAX {
        i1_no_log(ax * 0.5)
    } else {
        ln_i1_sum(ax).exp()
    };
    if x < 0.0 { -magnitude } else { magnitude }
}

/// The exponentially-scaled `I₀ᵉ(x) = e^{−|x|} I₀(x)`, finite for every argument.
pub(crate) fn i0e(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let ax = x.abs();
    if ax.is_infinite() {
        return 0.0;
    }
    if ax >= I_ASYMPTOTIC_MIN {
        return i_scaled_asymptotic(0.0, ax);
    }
    ln_i0_minus_x(ax).exp()
}

/// The exponentially-scaled `I₁ᵉ(x) = e^{−|x|} I₁(x)`.
pub(crate) fn i1e(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let ax = x.abs();
    if ax.is_infinite() {
        return 0.0;
    }
    let magnitude = if ax < I1_NO_LOG_MAX {
        i1_no_log(ax * 0.5) * (-ax).exp()
    } else if ax >= I_ASYMPTOTIC_MIN {
        i_scaled_asymptotic(1.0, ax)
    } else {
        ln_i1_minus_x(ax).exp()
    };
    if x < 0.0 { -magnitude } else { magnitude }
}

/// `I₁(x) / I₀(x)`, in `(−1, 1)`.
///
/// Both series peak at the same index `k`, and at that index the two peak terms differ by
/// exactly `ln h − ln(k+1)`. So the ratio is
///
/// ```text
/// I₁/I₀ = h/(k+1) · sum₁/sum₀
/// ```
///
/// where both sums are the peak-relative ones. Every quantity here is `O(1)`: the large
/// factors cancel *symbolically* rather than by subtracting two big logarithms. That matters
/// because those logarithms are `O(x)` — at `x = 400` they are around 400 while their
/// difference is around 0.0025, so even a relative error of `1e-7` in each would swamp the
/// answer. This form has no such cancellation at any argument or precision.
pub(crate) fn i1_over_i0(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let ax = x.abs();
    if ax == 0.0 {
        return 0.0;
    }
    if ax.is_infinite() {
        return if x < 0.0 { -1.0 } else { 1.0 };
    }
    if ax >= I_ASYMPTOTIC_MIN {
        let ratio = i_scaled_asymptotic(1.0, ax) / i_scaled_asymptotic(0.0, ax);
        return if x < 0.0 { -ratio } else { ratio };
    }
    let h = ax * 0.5;
    let k_peak = h.round().max(0.0);
    let ratio = (h / (k_peak + 1.0)) * (i1_rel_sum(h, k_peak) / i0_rel_sum(h, k_peak));
    if x < 0.0 { -ratio } else { ratio }
}

/// `ln I₀(x)`, the overflow-free log of `I₀`.
///
/// Taken straight from the peak-factored sum, so it needs no `e^{|x|}` round trip and stays
/// accurate for every argument.
pub(crate) fn ln_i0(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let ax = x.abs();
    if ax.is_infinite() {
        return f32::INFINITY;
    }
    ln_i0_sum(ax)
}

#[cfg(test)]
mod tests;
