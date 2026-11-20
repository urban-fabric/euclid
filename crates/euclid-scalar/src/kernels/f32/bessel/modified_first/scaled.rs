//! The public modified-Bessel surface: `I₀`, `I₁`, their scaled forms, the ratio and `ln I₀`.
//!
//! A child module so the parent stays inside the crate's 400-line budget. The split is between
//! the *log-domain sums* in [`super`] — which is where the peak-shifted summation and the
//! Stirling remainder live — and the functions carved out of them here. Every one of these is a
//! thin wrapper that chooses a form and unscales; the arithmetic that earns the accuracy is in
//! the parent.

use super::*;

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
pub(super) const I_ASYMPTOTIC_MIN: f32 = 1e2;

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
pub(super) const I1_NO_LOG_MAX: f32 = 1.0;

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
pub(super) const LN_I0_SERIES_MAX: f32 = 1.0;

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
pub(super) enum LnI0Form {
    /// Below [`LN_I0_SERIES_MAX`]: `ln(1 + (I₀ − 1))`, which avoids both of the losses the
    /// shifted form carries near the origin.
    Series,
    /// At or above it: `ln I₀ = (ln I₀ − x) + x`, whose cancellation `ln I₀` is by then large
    /// enough to absorb.
    Shifted,
}

impl LnI0Form {
    /// The boundary belongs to the shifted form.
    pub(super) fn for_magnitude(ax: f32) -> Self {
        if ax < LN_I0_SERIES_MAX {
            Self::Series
        } else {
            Self::Shifted
        }
    }
}
