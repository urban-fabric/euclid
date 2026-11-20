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

/// Cap on the outward walk in either direction from the peak. The significant terms span
/// `O(√x)` indices, so this covers arguments past `1e8` before it binds.
const MAX_TERMS: u32 = 100_000;

/// Relative size at which a term stops contributing.
///
/// **`1e-19`, which is deliberately only a little below `f64`'s resolution.** `ε = 2.22e-16`, so
/// a term at `1e-19` of the sum moves the result by about a thousandth of an ulp, and every term
/// after it moves it by less. The value was `1e-22`, which asked the peak-relative `I` sums —
/// which walk `O(√x)` terms — to converge three orders past the point where the format can
/// represent the difference: iterations bought and then discarded.
///
/// Raising it to `1e-19` is **free rather than a trade.** The outputs are bit-identical across a
/// 306,280-point sweep covering every kernel in this tree at both signs, while cutting `I₁ᵉ`
/// by **6.8%**.
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
        ln_gamma(k + 1.0) - (k * k.ln() - k + 0.5 * crate::math::log_domain::ln_tau_times(k))
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
    2.0 * k * ((h - k) / k).ln_1p() + 2.0 * (k - h)
        - crate::math::log_domain::ln_tau_times(k)
        - 2.0 * stirling_remainder(k)
}

/// Which of the three forms of the `I` series a magnitude selects.
///
/// Named rather than left as a chain of comparisons because the choice *is* the numerics, and
/// because the same split was re-tested in five functions — `ln_i0_minus_x`, [`i0e`], [`i1`],
/// [`i1e`] and [`i1_over_i0`] — against two constants, each site spelling out only the arms it
/// happened to need.
///
/// # Mutation testing is what asked for this
///
/// `replace < with <=` on `i1e`'s `ax < I1_NO_LOG_MAX` survived the entire suite, and it is not
/// an equivalent mutant: it moves `|x| = 1` from the peak-factored form to the no-logarithm one.
/// No *value* assertion can catch it, because the crossover is sited exactly where the two forms
/// agree — that is what makes it a good crossover. The decision is the thing worth pinning, so
/// it becomes a named function and the sibling test module asserts the routing directly. The `f32`
/// peer was given this treatment first and killed the same mutant.
///
/// The two boundaries are one split rather than two, which the constants do not show:
/// `ln_i0_minus_x` tested `k_peak < 1.0` separately, and since `k_peak = round(|x|/2)` that is
/// below one exactly when `|x| < 1` — which is [`I1_NO_LOG_MAX`] itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum IForm {
    /// `|x| < I1_NO_LOG_MAX`: the peak is the `k = 0` term, so no logarithm is formed.
    NoLog,
    /// `I1_NO_LOG_MAX <= |x| < I_ASYMPTOTIC_MIN`: the peak-relative power series.
    PeakFactored,
    /// `|x| >= I_ASYMPTOTIC_MIN`: the divergent asymptotic expansion.
    Asymptotic,
}

impl IForm {
    /// The regime `ax` falls in, for a non-negative magnitude.
    ///
    /// Open below [`I1_NO_LOG_MAX`] and closed at [`I_ASYMPTOTIC_MIN`], matching the constants'
    /// names. `|x| = 1` is [`Self::PeakFactored`], which is what `k_peak = round(0.5) = 1`
    /// already made it.
    pub(super) fn for_magnitude(ax: f64) -> Self {
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
fn ln_i0_minus_x(ax: f64) -> f64 {
    debug_assert!(ax >= 0.0);
    if ax == 0.0 {
        return 0.0; // I0(0) = 1
    }
    match IForm::for_magnitude(ax) {
        IForm::Asymptotic => return asymptotic::i_scaled(0.0, ax).ln(),
        IForm::NoLog | IForm::PeakFactored => {}
    }
    let h = ax * 0.5;
    let k_peak = h.round().max(0.0);
    // `NoLog` is exactly `k_peak < 1`; see `IForm`.
    if matches!(IForm::for_magnitude(ax), IForm::NoLog) {
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
///
/// # Precondition: `ax >= I1_NO_LOG_MAX`
///
/// Asserted rather than handled, and the assertion replaced a dead branch. `k_peak` is
/// `round(ax/2)`, which is below 1 exactly when `ax < 1` — and both callers send `ax < 1` to
/// [`i1_no_log`] instead, precisely because the logarithm round trip is lossy there. So the
/// `k_peak < 1` arm this function used to open with could not be reached, and a dense sweep of
/// 20,700 arguments at both precisions, both signs, across all four entry points confirmed it:
/// a `panic!()` in that arm never fired.
///
/// It was not free. Fifteen of this file's thirty-six surviving mutants lived in those three
/// lines — every operator in a `ln_peak` expression nothing evaluates — which is the same
/// bargain the f32 `erf_inv` Newton step turned out to be: unreachable arithmetic reads as a
/// test gap forever. Deleting it is the fix, and the `debug_assert` is what keeps it deleted,
/// since a future caller reaching for a small argument now fails loudly instead of silently
/// taking a path that no longer exists.
fn ln_i1_minus_x(ax: f64) -> f64 {
    debug_assert!(
        ax >= I1_NO_LOG_MAX,
        "ln_i1_minus_x needs ax >= {I1_NO_LOG_MAX}; smaller arguments belong in i1_no_log"
    );
    let h = ax * 0.5;
    let k_peak = h.round().max(0.0);
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
/// # `20`, lowered from `1e4`, because the old figure priced a fixed four-term expansion
///
/// The previous justification was that the expansion's "first omitted term is `|a₄|/x⁴ ≈ 1.1e-17`,
/// below half an ulp" at `1e4`. That is the right calculation for a **four-term** expansion, and
/// [`asymptotic::i_scaled`] is not one: it walks to the *minimum* term of the divergent series,
/// capped at `MAX_TERMS = 200` with a `TERM_EPS` of `1e-19` that is far below `ε` and so never
/// stops it early. The constant was set by a constraint the code had already stopped having.
///
/// Measured densely — 20,000 points per window — the expansion's optimal truncation against the
/// peak-factored series' actual error:
///
/// ```text
/// x            expansion truncation   series route
/// 12                 27212 ulp            9311 ulp
/// 15                    60                  53
/// 16                   8.7                  58
/// [20,  25]            0.125                 59
/// [25,  30]            0.125                 62
/// [30,  40]            0.125                 75
/// [40, 100]            0.125                 10
/// ```
///
/// Two things follow. The expansion is the better of the two from `x = 18` upward, by up to
/// **500×**, and the series is at its *worst* exactly where the old threshold kept it — `[20, 40]`
/// is both the series' worst region and deep inside the expansion's good one.
///
/// And the floor is a cliff rather than a slope, which is what sets the constant: `0.125` ulp at
/// `20`, `8.7` at `16`, `27212` at `12`. That is the signature of a divergent series' optimal
/// truncation, which falls off super-exponentially. `20` sits on the flat side with the cliff a
/// fifth of a decade below; going lower buys a region the series already handles adequately and
/// risks the part that is unusable.
///
/// After the change `I₀ᵉ` and `I₁ᵉ` are within **2.11 ulp uniformly from `x = 20` to `1e5`**,
/// against 57–75 ulp over `[20, 40]` and 4–8 over `[1e3, 1e4]` before.
///
/// Below the threshold the series is short and accurate; above it, it is never used.
pub(super) const I_ASYMPTOTIC_MIN: f64 = 20.0;

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
        IForm::PeakFactored | IForm::Asymptotic => ln_i1_sum(ax).exp(),
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
pub(super) const LN_I0_SERIES_MAX: f64 = 1.0;

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
    pub(super) fn for_magnitude(ax: f64) -> Self {
        if ax < LN_I0_SERIES_MAX {
            Self::Series
        } else {
            Self::Shifted
        }
    }
}
