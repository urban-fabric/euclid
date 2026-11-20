//! Single-precision peer of [`super::super::bessel`]. Same peak-factored power series; see
//! that module for why it replaced the Abramowitz & Stegun polynomials.
//!
//! The term cutoff is loosened to the `f32` noise floor, which shortens the outward walk.

use crate::constants::f32::SQRT_TAU;
use crate::kernels::f32::gamma::ln_gamma;

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
        ln_gamma(k + 1.0) - (k * k.ln() - k + 0.5 * crate::math::log_domain::ln_tau_times(k))
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

/// `ln I₁(|x|)`.
fn ln_i1_sum(ax: f32) -> f32 {
    ln_i1_minus_x(ax) + ax
}

/// At or above this argument the scaled `I` functions use their asymptotic expansion rather
/// than the peak-factored series.
///
/// This closes a hole rather than saving time. The series walks `O(√x)` terms outward from the
/// peak, and two things go wrong as that count grows: the naive accumulation drifts like
/// `√n·ε`, and past [`MAX_TERMS`] the walk is simply cut short.
///
/// **`1e2`, where the `f64` peer uses `1e4`, and the measurements below are this format's.** The
/// prose here used to be the peer's verbatim — it argued for `1e4` from `|a₄|/x⁴ ≈ 1.1e-17`
/// against "half an ulp", which are `f64`'s numbers, beside a constant that has been `1e2`
/// throughout. Both halves were right about a different format.
///
/// Two quantities set it, and both move by eight orders between the formats. The expansion's
/// first omitted term is `|a₄|/x⁴`, and `|a₄| = 0.112`:
///
/// ```text
///            |a₃|/x³   |a₄|/x⁴   half-ulp
/// x = 1e2    7.3e-8    1.1e-9    6.0e-8   (f32)
/// x = 1e4    7.3e-14   1.1e-17   1.1e-16  (f64)
/// ```
///
/// So `1e2` is already as deep for `f32` as `1e4` is for `f64`: four terms exhaust the format,
/// and the first dropped one is two orders below where it could matter.
///
/// The series, evaluated past the threshold where it is otherwise never used, against the
/// expansion — relative difference in `I₀ᵉ`:
///
/// ```text
/// x = 1e2   3.7e-7        x = 1e5   1.7e-6        x = 1e7   1.3e-5
/// x = 1e3   5.9e-7        x = 1e6   3.3e-6        x = 1e8   1.5e+1
/// ```
///
/// At the join the two agree to `3.7e-7`, a few `f32` ulps, so the handover costs nothing in
/// either direction — the same criterion the gamma and beta crossovers use. From `1e5` the
/// series is visibly drifting, and by `1e8` it is **wrong by a factor of 15** while remaining
/// finite and plausible: `6.4e-4` against a true `4.0e-5`. That is the hole, and it opens four
/// orders earlier here than the peer's table suggests, which is the other reason not to have
/// inherited the number.
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
pub(in crate::kernels::f32) fn i_scaled_asymptotic(nu: f32, x: f32) -> f32 {
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
    let ax = x.abs();
    // `I₀(±∞) = +∞`: the limit exists and is unambiguous, and without this the general route
    // reaches it through `∞ − ∞` in the peak-factored core and returns `NaN`. `I₀ᵉ`, `ln I₀` and
    // `I₁/I₀` each branch on infinity already; the plain forms were the only members of the
    // family answering `NaN` where the answer is a limit.
    if ax.is_infinite() {
        return f32::INFINITY;
    }
    ln_i0_sum(ax).exp()
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
    // `I₁(±∞) = ±∞`, for the same reason as `I₀` and signed because `I₁` is odd. `copysign`
    // rather than a comparison: no zero reaches this line, so the two agree, and this states the
    // intent directly. The unsigned-zero convention this family holds is about *zero*, not about
    // infinity — `I₁/I₀(±∞) = ±1` already signs its limit.
    if ax.is_infinite() {
        return f32::INFINITY.copysign(x);
    }
    let magnitude = match IForm::for_magnitude(ax) {
        IForm::NoLog => i1_no_log(ax * 0.5),
        // **No asymptotic arm, and that is not an omission.** `I₁` itself leaves `f32` by
        // `|x| ≈ 89` — `e^89/√(2π·89)` is already past `f32::MAX` — which is below
        // `I_ASYMPTOTIC_MIN`, so every argument that would reach the expansion has overflowed
        // before it gets there and the final `exp` returns `+∞` on either route. Switching forms
        // would cost a branch to compute the same infinity. The expansion earns its place in
        // [`i1e`], where the scaling keeps the result representable.
        IForm::PeakFactored | IForm::Asymptotic => ln_i1_sum(ax).exp(),
    };
    // `x < 0.0`, not `copysign`: this family returns an **unsigned** zero by convention, so
    // `-0.0` must take the `else` arm. See `zero_is_unsigned_and_the_first_kind_is_odd`, which
    // pins it because a `<`-to-`<=` mutation here flips `+0.0` to `-0.0` undetectably.
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
    match IForm::for_magnitude(ax) {
        IForm::Asymptotic => i_scaled_asymptotic(0.0, ax),
        // Both series forms go through `ln_i0_minus_x`, which makes the same `NoLog` distinction
        // internally — so this site does not repeat it.
        IForm::NoLog | IForm::PeakFactored => ln_i0_minus_x(ax).exp(),
    }
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
    let magnitude = match IForm::for_magnitude(ax) {
        IForm::NoLog => i1_no_log(ax * 0.5) * (-ax).exp(),
        IForm::PeakFactored => ln_i1_minus_x(ax).exp(),
        IForm::Asymptotic => i_scaled_asymptotic(1.0, ax),
    };
    // `x < 0.0`, not `copysign`: this family returns an **unsigned** zero by convention, so
    // `-0.0` must take the `else` arm. See `zero_is_unsigned_and_the_first_kind_is_odd`, which
    // pins it because a `<`-to-`<=` mutation here flips `+0.0` to `-0.0` undetectably.
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
        // An unsigned zero, deliberately: `i1_over_i0(±0) = +0` by this family's convention,
        // asserted in `i1_over_i0_keeps_the_sign_of_its_argument`.
        return 0.0;
    }
    if ax.is_infinite() {
        // `copysign` rather than a comparison: zero returned early above, so there is no
        // `-0.0` case to disagree about, and this states "reapply the argument's sign" directly
        // instead of encoding it as a branch a mutation can flip undetectably.
        return 1.0f32.copysign(x);
    }
    let ratio = match IForm::for_magnitude(ax) {
        IForm::Asymptotic => i_scaled_asymptotic(1.0, ax) / i_scaled_asymptotic(0.0, ax),
        // **No no-logarithm arm, and that is not an omission either.** The quotient never forms
        // a logarithm in the first place: the `h/(k_peak + 1)` factor already carries the `x/2`
        // that [`i1_no_log`] exists to keep exact, and at `k_peak = 0` the two peak-relative
        // sums both start from 1. So the small-argument limit comes back exact on this route
        // without a special case, which is the point of the form.
        IForm::NoLog | IForm::PeakFactored => {
            let h = ax * 0.5;
            let k_peak = h.round().max(0.0);
            (h / (k_peak + 1.0)) * (i1_rel_sum(h, k_peak) / i0_rel_sum(h, k_peak))
        }
    };
    ratio.copysign(x)
}

/// `ln I₀(x)`, the overflow-free log of `I₀`.
///
/// Taken straight from the peak-factored sum, so it needs no `e^{|x|}` round trip and stays
/// accurate for every argument.
/// `I₀(x) − 1 = Σ_{k≥1} (x²/4)^k/(k!)²`, summed **without ever forming the `1`**.
///
/// `h` is `x/2`, so the series variable is `t = h²`. Every term is positive, so there is no
/// cancellation at any argument and the sum carries full relative accuracy however small it is —
/// which is the whole point, because the quantity is `x²/4` at the origin and the `1` it would
/// otherwise be added to is overwhelmingly larger.
fn i0_minus_one(h: f32) -> f32 {
    crate::math::bessel::i0_minus_one_terms(h).0
}

/// Below this argument `ln I₀` is formed from [`i0_minus_one`] and `ln_1p` directly.
///
/// # Two compounding losses, not one
///
/// The general route is `ln_i0_minus_x(ax) + ax`, and for a small argument *both* halves of that
/// destroy the answer, which is why the fix has to sit at the entry point rather than inside.
///
/// 1. `ln I₀` was reached as `i0_rel_sum(h, 0).ln()` — the series summed **relative to its `k = 0`
///    term**, which is `1`. So it computes `1 + x²/4 + ⋯` and takes the logarithm of that,
///    progressively losing the small term to the rounding of `1`, and losing it completely once
///    `x²/4 < ε/2`.
/// 2. Then `ln_i0_minus_x` returns `ln I₀ − ax` and the caller adds `ax` back. But `ln I₀ ≈ x²/4`
///    is *far* smaller than `ax` near the origin, so the difference is `≈ −ax` and the round trip
///    cancels to nothing.
///
/// Measured before this branch existed, against the exact series: `ln I₀` was **100% wrong**
/// below `x ≈ 7e-4` — returning exactly `0` where the true value is a perfectly
/// representable `x²/4` — and still 4.6% wrong at `x = 1e-3`. The narrow format suffered four orders of
/// magnitude earlier than the wide one, which put the failure at ordinary arguments rather than
/// extreme ones.
///
/// # Why `1.0`
///
/// This is exactly [`IForm::NoLog`]'s range, where `k_peak` is `0` and the peak term is the `1`
/// at issue. Above it the series is no longer summed relative to `1` and the loss does not arise:
/// at `x = 1.5` the neglected information is `ε/0.65 ≈ 1.5ε`, already negligible. The branch is
/// also an improvement *at* the threshold — `s = 0.266` there, so the old route lost about `3.8ε`
/// to the `1 +` alone — and [`crate`]'s seam scan covers `ln_bessel_i0` at `1.0`, so the join is
/// checked rather than assumed.
const LN_I0_SERIES_MAX: f32 = 1.0;

pub(crate) fn ln_i0(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let ax = x.abs();
    if ax.is_infinite() {
        return f32::INFINITY;
    }
    match LnI0Form::for_magnitude(ax) {
        // Straight to `ln(1 + s)` with `s` summed on its own — see [`LN_I0_SERIES_MAX`] for the
        // two losses this bypasses. Deliberately not routed through `ln_i0_sum`, whose
        // `−ax`/`+ax` round trip is the second of them.
        LnI0Form::Series => i0_minus_one(ax * 0.5).ln_1p(),
        LnI0Form::Shifted => ln_i0_sum(ax),
    }
}

/// Which form of `ln I₀` the magnitude selects.
///
/// # Mutation testing is what asked for this
///
/// Narrowing the comparison survived, because the crossover is sited where the two forms agree —
/// which is what makes it a good crossover and what makes it invisible to a value assertion. The
/// decision is the thing worth pinning, following `IForm` in this same kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LnI0Form {
    /// Below [`LN_I0_SERIES_MAX`]: `ln(1 + (I₀ − 1))`, which avoids both of the losses the
    /// shifted form carries near the origin.
    Series,
    /// At or above it: `ln I₀ = (ln I₀ − x) + x`, whose cancellation `ln I₀` is by then large
    /// enough to absorb.
    Shifted,
}

impl LnI0Form {
    /// The boundary belongs to the shifted form.
    fn for_magnitude(ax: f32) -> Self {
        if ax < LN_I0_SERIES_MAX {
            Self::Series
        } else {
            Self::Shifted
        }
    }
}

#[cfg(test)]
mod tests;
