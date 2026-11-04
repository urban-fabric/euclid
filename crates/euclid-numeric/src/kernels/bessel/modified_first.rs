//! The modified Bessel functions of the **first kind**, `I₀` and `I₁`, with their
//! exponentially-scaled forms `Iₙᵉ(x) = e^{−|x|}Iₙ(x)`, the ratio `I₁/I₀`, and `ln I₀`.
//!
//! # Method: a peak-factored power series
//!
//! These previously used the Abramowitz & Stegun polynomial approximations, which are accurate
//! to only about 1e-7 *by construction* — roughly `f32` resolution out of an `f64` function,
//! and eight orders of magnitude weaker than the gamma and beta families beside them.
//!
//! The defining series is used instead, which has two properties that make it the better
//! choice:
//!
//! - **Every term is positive**, so there is no cancellation at any argument.
//! - **The terms peak sharply.** For `I₀`, `tₖ = (x/2)^{2k}/(k!)²` is maximal near `k = x/2`
//!   with a Gaussian-like width of `O(√x)`, so only `O(√x)` terms matter.
//!
//! The one obstacle is scale: the sum is about `e^x/√(2πx)`, which overflows `f64` past
//! `x ≈ 710` even though the *scaled* result `I₀ᵉ` is a small, perfectly representable number.
//! So the peak term is factored out in the log domain and the rest accumulated **relative to
//! it**, walking outward by ratio:
//!
//! ```text
//! tₖ₊₁/tₖ = (x/2)² / (k+1)²        tₖ₋₁/tₖ = k² / (x/2)²
//! ```
//!
//! Every accumulated value is then at most 1, the sum is `O(√x)` well-conditioned additions,
//! and a single `exp(ln t_peak − |x|)` at the end carries the scale. That last exponent is
//! formed by [`ln_peak_core_scaled`] rather than written out literally: spelled directly it
//! subtracts two `O(x ln x)` logarithms, which costs `I₀ᵉ` nine digits by `x = 1e6`.
//!
//! The result is full double precision at any argument, with no approximation coefficients to
//! transcribe and no large-argument expansion to join on — the old branch switch at
//! `|x| = 3.75`, and the join error there, are gone. The cost is `O(√x)` multiply-adds rather
//! than a fixed-length polynomial.

use super::asymptotic;
use crate::kernels::gamma::ln_gamma;
use core::f64::consts::TAU;

/// Cap on the outward walk in either direction from the peak. The significant terms span
/// `O(√x)` indices, so this covers arguments past `1e8` before it binds.
const MAX_TERMS: u32 = 100_000;

/// Relative size at which an outward term stops contributing.
const TERM_EPS: f64 = 1e-22;

/// The `I₀` series summed **relative to its peak term** at index `k_peak`, so the result is
/// `O(√x)` and cannot overflow. Shared by `ln_i0_sum` and the ratio.
fn i0_rel_sum(h: f64, k_peak: f64) -> f64 {
    let mut sum = 1.0f64;

    // Upward: t_{k+1}/t_k = h^2/(k+1)^2.
    let mut ratio = 1.0f64;
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

/// Above this peak index the Stirling remainder is taken from its asymptotic series; below it,
/// from `ln_gamma` directly. At `k = 20` the first omitted term, `−691/(360360·k¹¹)`, is
/// `1e-17`.
const STIRLING_ASYMPTOTIC_MIN: f64 = 20.0;

/// The Stirling remainder `μ(k) = lnΓ(k+1) − (k ln k − k + ½ln(2πk))`.
///
/// Needed in its own right rather than as a difference, because `μ(k) ≈ 1/(12k)` is tiny
/// while the two terms defining it are `O(k ln k)` — forming it by subtraction at `k = 5e5`
/// would leave nothing. The asymptotic series `1/(12k) − 1/(360k³) + ⋯` has no such problem
/// and is exact to rounding for any `k` large enough to matter.
fn stirling_remainder(k: f64) -> f64 {
    if k >= STIRLING_ASYMPTOTIC_MIN {
        let inv = 1.0 / k;
        let inv2 = inv * inv;
        inv * (1.0 / 12.0
            + inv2
                * (-1.0 / 360.0 + inv2 * (1.0 / 1260.0 + inv2 * (-1.0 / 1680.0 + inv2 / 1188.0))))
    } else {
        // Small k: the defining difference is between O(1) quantities, so it is safe, and the
        // asymptotic series would not yet have converged.
        ln_gamma(k + 1.0) - (k * k.ln() - k + 0.5 * (TAU * k).ln())
    }
}

/// `2k·ln h − 2·lnΓ(k+1) − 2h`, the shared log-domain core of both scaled peak terms.
///
/// This is the whole reason the scaled forms can be accurate at large argument. Written
/// literally, that expression subtracts two `O(k ln k)` quantities to leave an `O(ln k)`
/// result: at `x = 1e6` both are `1.3e7`, so `ε·1.3e7 ≈ 1.5e-9` of relative error survives into
/// `I₀ᵉ` — which is a thousand times worse than the `1.7e-14` the surface promises.
///
/// Substituting Stirling's form for `lnΓ(k+1)` makes the large parts cancel *symbolically*:
///
/// ```text
/// 2k ln h − 2lnΓ(k+1) − 2h = 2k·ln(h/k) + 2(k − h) − ln(2πk) − 2μ(k)
/// ```
///
/// Since `k = round(h)`, both `ln(h/k)` and `k − h` are small, so every term on the right is
/// `O(ln k)` or smaller and nothing cancels. The `ln(h/k)` is taken with `ln_1p` for the same
/// reason — `h/k` is within half a unit of 1.
fn ln_peak_core_scaled(h: f64, k: f64) -> f64 {
    debug_assert!(k >= 1.0);
    2.0 * k * ((h - k) / k).ln_1p() + 2.0 * (k - h) - (TAU * k).ln() - 2.0 * stirling_remainder(k)
}

/// `ln I₀(|x|) − |x|`, the scaled log from which every `I₀` form is derived.
fn ln_i0_minus_x(ax: f64) -> f64 {
    debug_assert!(ax >= 0.0);
    if ax == 0.0 {
        return 0.0; // I0(0) = 1
    }
    if ax >= I_ASYMPTOTIC_MIN {
        return asymptotic::i_scaled(0.0, ax).ln();
    }
    let h = ax * 0.5;
    let k_peak = h.round().max(0.0);
    if k_peak < 1.0 {
        // The peak is the k = 0 term, which is exactly 1, so `ln t_peak` is exactly 0 and
        // there is nothing to factor out.
        //
        // Worth stating because the general expression `2k·ln h − 2lnΓ(k+1)` does *not*
        // evaluate to 0 here: at the smallest subnormal `x = 5e-324` the halving `h = x/2`
        // underflows to zero, making the first term `0 × −∞ = NaN` and taking `I₀`, and
        // everything built on it, with it.
        return i0_rel_sum(h, k_peak).ln() - ax;
    }
    ln_peak_core_scaled(h, k_peak) + i0_rel_sum(h, k_peak).ln()
}

/// `ln I₀(|x|)`.
fn ln_i0_sum(ax: f64) -> f64 {
    ln_i0_minus_x(ax) + ax
}

/// The `I₁` series summed relative to its peak term at the same index.
fn i1_rel_sum(h: f64, k_peak: f64) -> f64 {
    let mut sum = 1.0f64;

    // Upward: t_{k+1}/t_k = h^2/((k+1)(k+2)).
    let mut ratio = 1.0f64;
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
///
/// so the same [`ln_peak_core_scaled`] carries the scale and only two `O(ln k)` terms differ.
fn ln_i1_minus_x(ax: f64) -> f64 {
    debug_assert!(ax >= 0.0);
    if ax == 0.0 {
        return f64::NEG_INFINITY; // I1(0) = 0
    }
    let h = ax * 0.5;
    let k_peak = h.round().max(0.0);
    if k_peak < 1.0 {
        let ln_peak =
            (2.0 * k_peak + 1.0) * h.ln() - ln_gamma(k_peak + 1.0) - ln_gamma(k_peak + 2.0);
        return ln_peak + i1_rel_sum(h, k_peak).ln() - ax;
    }
    h.ln() - (k_peak + 1.0).ln() + ln_peak_core_scaled(h, k_peak) + i1_rel_sum(h, k_peak).ln()
}

/// `ln I₁(|x|)`.
fn ln_i1_sum(ax: f64) -> f64 {
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
pub(super) const I_ASYMPTOTIC_MIN: f64 = 1e4;

/// The modified Bessel function `I₀(x)` (even in `x`).
///
/// Overflows for `|x| ≳ 710`, as the true value does; use [`i0e`] or [`ln_i0`] there.
pub(crate) fn i0(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
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
pub(super) const I1_NO_LOG_MAX: f64 = 1.0;

/// `I₁(|x|)` for `|x| < I1_NO_LOG_MAX`, straight from the series with no logarithm.
fn i1_no_log(h: f64) -> f64 {
    h * i1_rel_sum(h, 0.0)
}

/// The modified Bessel function `I₁(x)` (odd in `x`).
pub(crate) fn i1(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
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
pub(crate) fn i0e(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    if ax.is_infinite() {
        return 0.0;
    }
    if ax >= I_ASYMPTOTIC_MIN {
        return asymptotic::i_scaled(0.0, ax);
    }
    ln_i0_minus_x(ax).exp()
}

/// The exponentially-scaled `I₁ᵉ(x) = e^{−|x|} I₁(x)`.
pub(crate) fn i1e(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    if ax.is_infinite() {
        return 0.0;
    }
    let magnitude = if ax < I1_NO_LOG_MAX {
        i1_no_log(ax * 0.5) * (-ax).exp()
    } else if ax >= I_ASYMPTOTIC_MIN {
        asymptotic::i_scaled(1.0, ax)
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
pub(crate) fn i1_over_i0(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    if ax == 0.0 {
        return 0.0;
    }
    if ax.is_infinite() {
        return if x < 0.0 { -1.0 } else { 1.0 };
    }
    if ax >= I_ASYMPTOTIC_MIN {
        let ratio = asymptotic::i_scaled(1.0, ax) / asymptotic::i_scaled(0.0, ax);
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
pub(crate) fn ln_i0(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    if ax.is_infinite() {
        return f64::INFINITY;
    }
    ln_i0_sum(ax)
}
