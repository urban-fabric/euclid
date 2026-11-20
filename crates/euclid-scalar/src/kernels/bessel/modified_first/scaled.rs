//! The public modified-Bessel surface: `I₀`, `I₁`, their scaled forms, the ratio and `ln I₀`.
//!
//! A child module so the parent stays inside the crate's 400-line budget, splitting where the
//! [single-precision peer](crate::kernels::f32::bessel::modified_first) already splits: the
//! *log-domain sums* stay in [`super`], which is where the peak-shifted summation and the
//! Stirling remainder live, and the functions carved out of them are here. Every one of these is
//! a thin wrapper that chooses a form and unscales; the arithmetic that earns the accuracy is in
//! the parent.
//!
//! The two crossover constants live here rather than beside the sums because they are what the
//! *surface* switches on, and [`super::IForm`] takes them back for its table.

use super::*;

/// The modified Bessel function `I₀(x)` (even in `x`).
///
/// Overflows for `|x| ≳ 710`, as the true value does; use [`i0e`] or [`ln_i0`] there.
pub(crate) fn i0(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    // `I₀(±∞) = +∞`: the limit exists and is unambiguous, and without this the general route
    // reaches it through `∞ − ∞` in the peak-factored core and returns `NaN`. `I₀ᵉ`, `ln I₀` and
    // `I₁/I₀` each branch on infinity already; the plain forms were the only members of the
    // family answering `NaN` where the answer is a limit.
    if ax.is_infinite() {
        return f64::INFINITY;
    }
    match IForm::for_magnitude(ax) {
        // The peak *is* the `k = 0` term, which is exactly 1, so the peak-relative sum is `I₀`
        // itself and no logarithm is formed at all. Reaching this through `ln_i0_sum(ax).exp()`
        // spent a `ln` and an `exp` to return the number it already had.
        IForm::NoLog => i0_rel_sum(ax * 0.5, 0.0),
        // The scaled form is the primitive one; see `scaling::unscale` for why applying the
        // exponential to it beats folding `ax` into the logarithm first.
        IForm::PeakFactored | IForm::Asymptotic => crate::math::scaling::unscale(i0e(ax), ax),
    }
}

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
    // `I₁(±∞) = ±∞`, for the same reason as `I₀` and signed because `I₁` is odd. `copysign`
    // rather than a comparison: no zero reaches this line, so the two agree, and this states the
    // intent directly. The unsigned-zero convention this family holds is about *zero*, not about
    // infinity — `I₁/I₀(±∞) = ±1` already signs its limit.
    if ax.is_infinite() {
        return f64::INFINITY.copysign(x);
    }
    let magnitude = match IForm::for_magnitude(ax) {
        IForm::NoLog => i1_no_log(ax * 0.5),
        // No asymptotic arm: `I₁` leaves `f64` by `|x| ≈ 710`, far below `I_ASYMPTOTIC_MIN`, so
        // every argument that would reach the expansion has already overflowed and `exp` returns
        // `+∞` either way. The expansion earns its place in `i1e`, where the scaling keeps the
        // result representable.
        IForm::PeakFactored | IForm::Asymptotic => crate::math::scaling::unscale(i1e(ax), ax),
    };
    // `x < 0.0`, not `copysign`: this family returns an **unsigned** zero by convention, so
    // `-0.0` must take the `else` arm. See `zero_is_unsigned_and_the_first_kind_is_odd`, which
    // pins it because a `<`-to-`<=` mutation here flips `+0.0` to `-0.0` undetectably.
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
    match IForm::for_magnitude(ax) {
        IForm::Asymptotic => asymptotic::i_scaled(0.0, ax),
        // Both series forms route through `ln_i0_minus_x`, which makes the same `NoLog`
        // distinction internally.
        IForm::NoLog | IForm::PeakFactored => ln_i0_minus_x(ax).exp(),
    }
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
    let magnitude = match IForm::for_magnitude(ax) {
        IForm::NoLog => i1_no_log(ax * 0.5) * (-ax).exp(),
        IForm::PeakFactored => ln_i1_minus_x(ax).exp(),
        IForm::Asymptotic => asymptotic::i_scaled(1.0, ax),
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
pub(crate) fn i1_over_i0(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
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
        return 1.0f64.copysign(x);
    }
    match IForm::for_magnitude(ax) {
        IForm::Asymptotic => {
            let ratio = asymptotic::i_scaled(1.0, ax) / asymptotic::i_scaled(0.0, ax);
            return ratio.copysign(x);
        }
        IForm::NoLog | IForm::PeakFactored => {}
    }
    let h = ax * 0.5;
    let k_peak = h.round().max(0.0);
    let ratio = (h / (k_peak + 1.0)) * (i1_rel_sum(h, k_peak) / i0_rel_sum(h, k_peak));
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
fn i0_minus_one(h: f64) -> f64 {
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
/// below `x ≈ 2.1e-8` — returning exactly `0` where the true value is a perfectly
/// representable `x²/4` — and still 2.3% wrong at `x = 1e-7`. The narrow format suffered four orders of
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
pub(in crate::kernels::bessel) const LN_I0_SERIES_MAX: f64 = 1.0;

pub(crate) fn ln_i0(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    if ax.is_infinite() {
        return f64::INFINITY;
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
pub(in crate::kernels::bessel) enum LnI0Form {
    /// Below [`LN_I0_SERIES_MAX`]: `ln(1 + (I₀ − 1))`, which avoids both of the losses the
    /// shifted form carries near the origin.
    Series,
    /// At or above it: `ln I₀ = (ln I₀ − x) + x`, whose cancellation `ln I₀` is by then large
    /// enough to absorb.
    Shifted,
}

impl LnI0Form {
    /// The boundary belongs to the shifted form.
    pub(in crate::kernels::bessel) fn for_magnitude(ax: f64) -> Self {
        if ax < LN_I0_SERIES_MAX {
            Self::Series
        } else {
            Self::Shifted
        }
    }
}
