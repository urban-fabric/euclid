//! Single-precision peer of [`super::super::bessel`]. Same peak-factored power series; see
//! that module for why it replaced the Abramowitz & Stegun polynomials.
//!
//! The term cutoff is loosened to the `f32` noise floor, which shortens the outward walk.

use crate::constants::f32::SQRT_TAU;

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

/// `μ(k)` below [`STIRLING_ASYMPTOTIC_MIN`], correctly rounded from 256-bit MPFR.
///
/// The `f64` peer explains the defect this replaces; here is what it cost in this format. The
/// difference form `lnΓ(k+1) − (k ln k − k + ½ln(2πk))` subtracts two `O(k ln k)` quantities to
/// leave an `O(1/12k)` one, and measured in `f32`:
///
/// ```text
/// k                1       2       3       4       5       6       7
/// relative error  1.9e-7  3.1e-7  1.2e-6  4.6e-6  1.3e-5  1.2e-5  1.1e-4
/// ```
///
/// `1.1e-4` at `k = 7` against a format resolution of `1.19e-7` — **nine hundred times**. `μ` is
/// doubled and added to an exponent that `exp` consumes, so `2·1.1e-4·μ(7) = 2.6e-6` lands
/// straight in `I₀ᵉ` as relative error. Measured over `x ∈ [10, 40]`, which is where
/// `k = round(x/2)` takes the values 5 to 7: **`2.7e-6`**. The prediction and the measurement
/// agree to one significant figure, which is what identifies this as the whole of it.
const STIRLING_REMAINDER: [f32; 7] = [
    0.081_061_47,  // k = 1
    0.041_340_698, // k = 2
    0.027_677_925, // k = 3
    0.020_790_672, // k = 4
    0.016_644_692, // k = 5
    0.013_876_129, // k = 6
    0.011_896_71,  // k = 7
];

/// The Stirling remainder `μ(k) = lnΓ(k+1) − (k ln k − k + ½ln(2πk))`.
///
/// Needed in its own right rather than as a difference — see
/// [the `f64` peer](crate::kernels::bessel::modified_first) for the argument; it applies with
/// more force here, since `f32` has only seven digits to lose.
///
/// Two regimes, and the split is [`STIRLING_ASYMPTOTIC_MIN`]. Above it the asymptotic series is
/// exact to this format's rounding: the first omitted term `−1/(1680k⁷)` is `2.8e-10` at `k = 8`,
/// which is `2.7e-8` of `μ(8)` and so under a quarter-ulp. Below it the series has not converged
/// and the *difference* cannot be formed in this format at all, so the nine reachable indices
/// come from a table. `k` is `round(|x|/2)` and therefore always an integer.
fn stirling_remainder(k: f32) -> f32 {
    if k >= STIRLING_ASYMPTOTIC_MIN {
        let inv = 1.0 / k;
        let inv2 = inv * inv;
        return inv * (1.0 / 12.0 + inv2 * (-1.0 / 360.0 + inv2 / 1260.0));
    }
    debug_assert!(
        (1.0..STIRLING_ASYMPTOTIC_MIN).contains(&k) && k.fract() == 0.0,
        "the Stirling table covers integer k in 1..8; k = {k} is outside it"
    );
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let index = (k as usize).clamp(1, STIRLING_REMAINDER.len()) - 1;
    STIRLING_REMAINDER[index]
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
    2.0 * k * ((h - k) / k).ln_1p() + 2.0 * (k - h)
        - crate::math::log_domain::ln_tau_times(k)
        - 2.0 * stirling_remainder(k)
}

/// Which of the three forms of the `I` series a magnitude selects.
///
/// Named rather than left as a chain of comparisons because the choice *is* the numerics, and
/// because the same split was being re-tested in five functions — `ln_i0_minus_x`, [`i0e`],
/// [`i1`], [`i1e`] and [`i1_over_i0`] — against two constants, with each site spelling out only
/// the arms it happened to need. The three forms are algebraically equal and fail in different
/// places, so which one runs is the whole content of [`I1_NO_LOG_MAX`] and [`I_ASYMPTOTIC_MIN`].
///
/// The two boundaries are one split rather than two unrelated ones, which is not obvious from
/// the constants: `ln_i0_minus_x` used to test `k_peak < 1.0` separately, and since
/// `k_peak = round(|x|/2)`, that is below one exactly when `|x| < 1` — which is
/// [`I1_NO_LOG_MAX`] itself. The `I₀` small-argument case and the `I₁` no-logarithm case are
/// therefore the same regime, reached by two different routes through the old code.
///
/// Not every caller has all three arms, and the asymmetry is real rather than an oversight: see
/// [`i1`] for why the unscaled `I₁` has no asymptotic arm, and [`i1_over_i0`] for why the ratio
/// needs no no-logarithm arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IForm {
    /// `|x| < I1_NO_LOG_MAX`, where the peak is the `k = 0` term and no logarithm is needed.
    ///
    /// For `I₀` the peak term is exactly `1`, so `ln t_peak` is exactly `0`; for `I₁` it is
    /// exactly `x/2`. Both are perfectly conditioned limits that a `ln`/`exp` round trip would
    /// spoil, and the general expression is worse than unnecessary here — `2k·ln h` is
    /// `0 × −∞ = NaN` once `h = x/2` underflows, which it does at the smallest `f32` subnormal.
    NoLog,
    /// `I1_NO_LOG_MAX <= |x| < I_ASYMPTOTIC_MIN`: the power series summed relative to its peak
    /// term, with the large factors cancelled symbolically in the log domain.
    PeakFactored,
    /// `|x| >= I_ASYMPTOTIC_MIN`: the divergent asymptotic expansion, truncated at its minimum
    /// term. Closes the hole where the series' outward walk gets long enough to drift or to be
    /// cut short by `MAX_TERMS`.
    Asymptotic,
}

impl IForm {
    /// The regime `ax` falls in, for a non-negative magnitude.
    ///
    /// The boundaries are closed from above and open from below in the same direction as the
    /// constants' names: `NoLog` strictly below [`I1_NO_LOG_MAX`], `Asymptotic` at or above
    /// [`I_ASYMPTOTIC_MIN`]. `|x| = 1` is therefore [`Self::PeakFactored`], which is what
    /// `k_peak = round(0.5) = 1` already made it.
    fn for_magnitude(ax: f32) -> Self {
        debug_assert!(ax >= 0.0);
        if ax < I1_NO_LOG_MAX {
            Self::NoLog
        } else if ax >= I_ASYMPTOTIC_MIN {
            Self::Asymptotic
        } else {
            Self::PeakFactored
        }
    }
}

/// `ln I₀(|x|) − |x|`, the scaled log from which every `I₀` form is derived.
fn ln_i0_minus_x(ax: f32) -> f32 {
    debug_assert!(ax >= 0.0);
    if ax == 0.0 {
        return 0.0; // I0(0) = 1
    }
    match IForm::for_magnitude(ax) {
        IForm::Asymptotic => i_scaled_asymptotic(0.0, ax).ln(),
        // `k_peak` is provably `0` across this whole arm, so it is written as the literal it is
        // rather than recomputed: the peak is the `k = 0` term, which is exactly 1, so
        // `ln t_peak` is exactly 0. See [`IForm::NoLog`] for why the general expression must not
        // be allowed to produce that instead.
        IForm::NoLog => i0_rel_sum(ax * 0.5, 0.0).ln() - ax,
        IForm::PeakFactored => {
            let h = ax * 0.5;
            let k_peak = h.round().max(0.0);
            ln_peak_core_scaled(h, k_peak) + i0_rel_sum(h, k_peak).ln()
        }
    }
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
///
/// # Precondition: `ax >= I1_NO_LOG_MAX`
///
/// Same asserted precondition, and same reason, as the f64 peer: `k_peak = round(ax/2)` is below
/// 1 only when `ax < 1`, and both callers route those to `i1_no_log`. The arm that handled them
/// here was unreachable and is gone.
fn ln_i1_minus_x(ax: f32) -> f32 {
    debug_assert!(
        ax >= I1_NO_LOG_MAX,
        "ln_i1_minus_x needs ax >= {I1_NO_LOG_MAX}; smaller arguments belong in i1_no_log"
    );
    let h = ax * 0.5;
    let k_peak = h.round().max(0.0);
    h.ln() - (k_peak + 1.0).ln() + ln_peak_core_scaled(h, k_peak) + i1_rel_sum(h, k_peak).ln()
}

mod scaled;

// Re-exported so the family keeps one path per function: every caller reached
// `kernels::f32::bessel::modified_first::i0` before the split and still does.
pub(in crate::kernels::f32) use scaled::i_scaled_asymptotic;
pub(crate) use scaled::{i0, i0e, i1, i1_over_i0, i1e, ln_i0};
// `IForm` lives here but its two thresholds are documented beside the forms they select.
use scaled::{I_ASYMPTOTIC_MIN, I1_NO_LOG_MAX};

#[cfg(test)]
mod tests;
