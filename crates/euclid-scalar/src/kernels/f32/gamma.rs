//! Single-precision gamma family, native throughout.
//!
//! `digamma` and `trigamma` used to narrow the double-precision kernels "to retain accuracy near
//! roots and poles". They no longer do: an `f32` surface exists to be usable on a carrier that
//! has no `f64`, so a kernel that reaches for `f64` to be accurate has given up the only thing
//! it was for. Both are now the `f64` peer's algorithm — reflect, recur, then an asymptotic
//! series — with the recurrence threshold rederived at this format's resolution; see
//! [`PSI_ASYMPTOTIC_MIN`], where inheriting the `f64` value turns out to be actively worse
//! rather than merely wasteful.
//!
//! The old comment was right about where the difficulty is, and naming it is what made the port
//! work. `ψ` has a zero at `1.4616` and the plain recurrence cannot resolve it — it returned
//! **exactly `0.0`** at two of the three representable arguments either side — so `digamma`
//! factors the zero out the way `ln_gamma` factors out its two; see [`DIGAMMA_UNIT_H`].
//! `trigamma` needs no such treatment, having no zero on the positive axis.
//!
//! The result is that going native made these *more* accurate than the kernels they used to
//! narrow, not less: over the `[0.1, 20]` sweep `digamma`'s cross-precision budget fell from 31
//! to 3 ulp, because the factored form reproduces the zero where a narrowed `f64` value still
//! had to pay for rounding the argument. `trigamma` went the other way, 1 to 3. Measured at a
//! *common* argument — the only fair reading, since near a pole `ψ ≈ −1/(x−n)` is so sensitive
//! that rounding `x` to `f32` moves the answer by `6e-4` relative — both hold to `4.5` and `2.5`
//! times the format's resolution across `[−9.9, −0.1]`, and to `2.9e-7` and `8.7e-9` absolute
//! against the exact closed forms `ψ(n) = −γ + H_{n−1}` and `ψ₁(n) = ζ(2) − Σ_{k<n} k^{−2}`.

use crate::math::continued_fraction::LENTZ_TINY_F32;
use crate::math::gamma::{
    gamma_edgeworth, gamma_lower_series, gamma_reflection, gamma_upper_fraction, lanczos_gamma,
    lanczos_log_gamma, lanczos_reflected_gamma, log_gamma_reflection, log_scaled_gamma,
    reflection_sine, use_gamma_reflection, use_lower_gamma_series, valid_incomplete_gamma_inputs,
};

/// Lanczos `g` parameter and coefficients (g = 7, n = 9), narrowed to `f32`.
const G: f32 = 7.0;
const LANCZOS: [f32; 9] = [
    0.999_999_999_999_809_9,
    676.520_4,
    -1_259.139_2,
    771.323_4,
    -176.615_04,
    12.507_343,
    -0.138_571_1,
    9.984_369e-6,
    1.505_632_7e-7,
];

/// Above this argument `Γ(x)` exceeds the format's range, and ``+∞`` is returned directly.
///
/// `Γ` overflows where `lnΓ(x)` passes `ln(f32::MAX) = 88.72`, which is `x ≈ 35.04`. The
/// threshold sits just above that, so nothing representable is saturated: `Γ(35)` is
/// `2.95e38` and is returned in full.
///
/// Needed as an explicit test rather than left to the arithmetic. The Lanczos form below keeps
/// its intermediates in range only while the *result* is in range, which is exactly what makes
/// it accurate — but once `x` is far past the boundary the half-power overflows while `e^{−t/2}`
/// underflows, and `inf · 0` is `NaN`. Saturating deliberately is both correct and cheaper.
const GAMMA_OVERFLOW_MIN: f32 = 35.1;

/// `g(a) = lnΓ(1+a) / (a(a−1))` on `a ∈ [0, 1]`, ascending powers.
///
/// # Why `lnΓ` here is not the Lanczos log form
///
/// [`lanczos_log_gamma`] evaluates `½ln(2π) + power·ln(base) − base + ln(sum)` with
/// `base = x + g + ½`. That is a **cancelling sum**: at `x = 2.673` the terms are `0.919`,
/// `4.816` and `−10.17`, and they produce `lnΓ = 0.4135`. The absolute error of the result is
/// therefore set by `ulp(base)`, not by the size of the answer — `ulp(10.17)/2 = 4.8e-7` in
/// `f32` — and `lnΓ` is *small* across the whole region bounded by its two zeros at `1` and `2`.
/// Measured against the `f64` kernel, the native Lanczos route carried `6.04e-7` absolute at
/// `x = 2.673`, about **20 ulps** of its own result, where this module's header claims the
/// native/narrowed gap is "typically within an ULP".
///
/// No reassociation fixes that, because the lost bits are not in the summation — they are
/// missing from `base` itself. What fixes it is not forming the large terms at all.
///
/// `lnΓ(1+a)` has simple zeros at `a = 0` and `a = 1`, so `lnΓ(1+a) = a(a−1)·g(a)` with `g`
/// analytic and slowly varying — `γ = 0.5772` at `a = 0`, `1 − γ = 0.4228` at `a = 1`, and
/// monotone between. Factoring the zeros out means they are reproduced *exactly* rather than by
/// subtraction, and the only rounding left is the polynomial's own. Any argument is brought into
/// `[1, 2]` by the recurrence first; see [`ln_gamma_reduced`].
///
/// # Provenance
///
/// Degree 9, a Chebyshev least-squares fit on `[0, 1]` against `g` evaluated to 60 decimal
/// digits — `lnΓ(1+a) = −γa + Σ_{k≥2} (−1)^k ζ(k) a^k / k` summed in `Decimal`, with the upper
/// half reflected through `lnΓ(1+a) = lnΓ(1+e) + ln(1+e)`, `e = a − 1`, so the reference keeps
/// full relative accuracy at both endpoints. A reference built from `f64` `lgamma` instead will
/// *appear* to cap the achievable fit at `1.3e-7`, which is `lgamma`'s own relative error a
/// nanometre from its zeros rather than anything about the fit. Generated by a local tool; see
/// CONTRIBUTING.md, "Reference values are pasted in".
///
/// The fit itself reaches `4.0e-9` relative. Evaluated in `f32` by Horner the floor is
/// `1.97e-7` relative — **3.3 ulps** — which is the `f32` arithmetic in the recurrence and not
/// the coefficients; raising the degree does not move it. At `x = 2.673` that is `8.1e-8`
/// absolute against the Lanczos form's `6.04e-7`.
const LN_GAMMA1P_FACTORED: [f32; 10] = [
    // `g(0) = γ` exactly — `lnΓ(1+a) ≈ −γa` as `a → 0`, so this coefficient is the
    // Euler–Mascheroni constant rather than a fitted number, and the fit reproduces it to the
    // last `f32` bit. Taken from the constant so it reads as the identity it is.
    crate::constants::f32::EULER_GAMMA,
    -0.245_250_93,
    0.155_418_77,
    -0.114_933_98,
    0.090_727_13,
    -0.070_935_3,
    0.049_395_33,
    -0.026_635_678,
    0.009_306_42,
    -0.001_523_098_4,
];

/// Above this argument the Lanczos form's cancellation no longer dominates, and it takes over.
///
/// The two routes are compared by their measured gap to the `f64` kernel. Below about `10` the
/// Lanczos form's absolute error `≈ ulp(x + 7.5)` is large beside `lnΓ(x)`, which is `0` at
/// `x = 1` and `2` and only reaches `12.8` by `x = 10`; the reduced form is flat in *relative*
/// terms because it never forms a term larger than its answer. From `10` upward `lnΓ` grows
/// faster than the intermediates and the Lanczos form is already within a couple of ulps, while
/// the recurrence would have to accumulate eight or more logarithms to get there — so this is
/// where the better of the two changes hands.
const LN_GAMMA_RECURRENCE_MAX: f32 = 10.0;

/// Which numerical method evaluates `lnΓ` at a given argument.
///
/// Three regimes, split on two thresholds, and the split is the numerics rather than an
/// implementation detail — each route is unusable where another one runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LnGammaMethod {
    /// Negative arguments: reflect to the positive side through `lnΓ(1−x)`.
    Reflected,
    /// `0 < x < `[`LN_GAMMA_RECURRENCE_MAX`]: recurrence onto `[1, 2]` and the factored
    /// polynomial, which never forms a term larger than its answer. The only route that is
    /// accurate where `lnΓ` is small.
    Reduced,
    /// Large arguments: the Lanczos log form, whose cancellation `lnΓ` is by then big enough to
    /// absorb.
    Lanczos,
}

impl LnGammaMethod {
    fn for_argument(x: f32) -> Self {
        if x.is_sign_positive() && x < LN_GAMMA_RECURRENCE_MAX {
            Self::Reduced
        } else if use_gamma_reflection(x) {
            Self::Reflected
        } else {
            Self::Lanczos
        }
    }
}

/// `lnΓ(1+a)` for `a ∈ [0, 1]`, with both zeros factored out. See [`LN_GAMMA1P_FACTORED`].
fn ln_gamma1p_factored(a: f32) -> f32 {
    let mut sum = 0.0;
    for &coefficient in LN_GAMMA1P_FACTORED.iter().rev() {
        sum = sum * a + coefficient;
    }
    a * (a - 1.0) * sum
}

/// `lnΓ(x)` for `0 < x < LN_GAMMA_RECURRENCE_MAX`, by recurrence onto `[1, 2]`.
///
/// `lnΓ(x) = lnΓ(x−1) + ln(x−1)` walks the argument down; `lnΓ(x) = lnΓ(x+1) − ln(x)` walks it
/// up from below one. Every logarithm added on the way down takes an argument `≥ 1`, so the
/// accumulated sum is of non-negative terms and does not cancel — which is the property the
/// Lanczos form lacks and the reason this is worth the handful of extra `ln` calls.
fn ln_gamma_reduced(x: f32) -> f32 {
    if x < 1.0 {
        // Γ(x) = Γ(1+x)/x. For a tiny `x` the answer is dominated by `−ln x`, and the factored
        // term is `≈ −γx`, so nothing large is formed here either.
        return ln_gamma1p_factored(x) - x.ln();
    }
    // Bring `x` into `[1, 2)` and collect the logarithms shed on the way.
    let mut shifted = x;
    let mut logs = 0.0;
    while shifted >= 2.0 {
        shifted -= 1.0;
        logs += shifted.ln();
    }
    ln_gamma1p_factored(shifted - 1.0) + logs
}

/// Natural log of the gamma function on intervals where `Γ(x)` is positive.
pub(crate) fn ln_gamma(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { f32::INFINITY } else { f32::NAN };
    }
    // The non-positive integers are the poles of Γ. `ln|Γ|` does tend to `+∞` there, and
    // from both sides, so the limit is well defined and is returned.
    //
    // This has to be tested for rather than left to the reflection below, because
    // `sin(πx)` at an integer `x` does not evaluate to zero — it evaluates to the rounding
    // residue of `πx`, whose *sign* is then arbitrary. That made `ln_gamma` return `NaN` at
    // `−1` and `−5` (residue negative, so `ln` of a negative) but a plausible finite number at
    // `−2`, `−20` and `−1e5`: `36.4`, `−7.55`, `−1.05e6`. All three were noise.
    if x <= 0.0 && x == x.trunc() {
        return f32::INFINITY;
    }
    // Native throughout: `Reduced` replaces what used to be a narrowing of the `f64` kernel
    // below `WIDE_SHAPE_MAX`, and extends the accurate range from `0.01` up to `10`. An `f32`
    // surface exists to be usable where `f64` is not, so deferring to `f64` for accuracy would
    // defeat the point of the module.
    match LnGammaMethod::for_argument(x) {
        LnGammaMethod::Reduced => ln_gamma_reduced(x),
        LnGammaMethod::Reflected => log_gamma_reflection(reflection_sine(x), ln_gamma(1.0 - x)),
        LnGammaMethod::Lanczos => lanczos_log_gamma(x, G, &LANCZOS),
    }
}

/// The gamma function `Γ(x)`.
pub(crate) fn gamma(x: f32) -> f32 {
    // The non-positive integers are the poles of Γ. Unlike `ln|Γ|`, `Γ` itself has *opposite*
    // signs on the two sides of each pole, so no limit exists and `NaN` is the answer; at the
    // origin the sign of the zero says which side is meant, so `±∞` is.
    //
    // Tested for rather than left to the reflection, for the reason given in `ln_gamma`:
    // `sin(πx)` returns the rounding residue of `πx` at an integer argument, not zero. That
    // made `Γ(−1) = −2.57e16`, `Γ(−5) = −4.28e13` and — worst, because it looks entirely
    // plausible — `Γ(−20) = 5.27e-4`.
    if x <= 0.0 && x == x.trunc() {
        return if x == 0.0 {
            f32::INFINITY.copysign(x)
        } else {
            f32::NAN
        };
    }
    if use_gamma_reflection(x) {
        // Native, where this used to widen to `f64` and let the final narrowing round the
        // subnormal. The reflection is the same algebra in either format; only the clamp
        // differs, and it has to.
        gamma_reflection(x, |sin, reflected| {
            lanczos_reflected_gamma(sin, reflected, G, REFLECTED_ZERO_MIN, &LANCZOS)
        })
    } else if x > GAMMA_OVERFLOW_MIN {
        f32::INFINITY
    } else {
        lanczos_gamma(x, G, &LANCZOS)
    }
}

/// Reflected argument past which `Γ` on the negative side has certainly rounded to signed zero.
///
/// **`50`, where the `f64` peer uses `200`.** Two format-dependent quantities set it, and they
/// pull in opposite directions, so it cannot be inherited either way.
///
/// `|Γ(−N + d)| = π / (|sin(πd)|·Γ(1 + N − d))`, so the result is small when the reflected
/// `Γ` is large *and* `d` is not tiny. The denominator's floor is the spacing of this format's
/// inputs near the pole: at `|x| ≈ 50`, `f32`'s ulp is `2^-18 ≈ 3.8e-6`, far coarser than
/// `f64`'s `2^-46` — which works in favour of a *lower* clamp, since `|sin(πd)|` cannot be as
/// small here. Against that, `Γ(50) = 6.1e62` where `Γ(200)` is astronomically larger.
///
/// Taken together: `|Γ| ≤ 1/(3.8e-6 · 6.1e62) = 4.4e-58`, under `f32`'s least subnormal
/// `1.4e-45` by thirteen orders, so everything past `50` is signed zero. Clamping there rather
/// than at `200` also matters more than it does in `f64`: `Γ(200)` would overflow an `f32`
/// intermediate long before the clamp took effect, which is the `inf · 0 = NaN` the clamp exists
/// to prevent.
const REFLECTED_ZERO_MIN: f32 = 50.0;

/// Iteration cap for the incomplete-gamma series and continued fraction.
///
/// Both need roughly `O(sqrt(a))` terms near `x ~ a`, so a small cap silently truncates at
/// large shape: at 200 the series stopped short from about `a = 1e3`, leaving `P(1e5, median)`
/// at `0.237` instead of `0.5` and putting a visible discontinuity at the `x = a + 1` branch
/// switch. All the series terms are positive, so this is pure truncation with no cancellation —
/// raising the cap fixes it outright, and costs nothing at small shape because both loops
/// break as soon as the relative term falls below `EPS`.
const MAX_ITER: usize = 100_000;
// Loosened from the `f64` `1e-15`: a relative tolerance below `f32::EPSILON` (~1.2e-7) can
// never be met, so the series/continued fractions would spin to `MAX_ITER`. `1e-6` sits
// just above the noise floor — full `f32` accuracy, prompt convergence.
const EPS: f32 = 1e-6;

/// At or above this shape the incomplete gamma switches to an Edgeworth-corrected normal
/// approximation; the single-precision peer of
/// [`crate::kernels::gamma::GAMMA_EDGEWORTH_MIN`], which carries the full explanation.
///
/// Four orders of magnitude lower than the `f64` threshold, because single precision loses the
/// series that much sooner. Measured against the `f64` reference, worst case over
/// `z = (x−a)/√a ∈ [−1, 1]`:
///
/// ```text
///          series    Edgeworth              series    Edgeworth
/// a = 1e2  2.0e-6    4.3e-4       a = 1e4   1.5e-3    4.2e-6
/// a = 1e3  1.0e-4    4.1e-5       a = 1e5   3.5e-2    5.0e-7
///                                 a = 1e6   3.7e-1    2.8e-9
/// ```
///
/// so `1e3` is where the approximation overtakes the series, and from there the series only
/// gets worse — reaching `NaN` by `a = 1e8`.
pub(super) const GAMMA_EDGEWORTH_MIN: f32 = 1e3;

/// Shapes for which cancellation warrants a dedicated small-shape route.
/// Below this shape the complement needs a series that retains terms of order `a`.
///
/// Set to `0.01`, which is where the `f64` deferral this replaced used to begin — so the native
/// small-shape branch covers exactly the region that used to be widened, and the regular series
/// above it is untouched. The `f64` peer's own boundary is `1e-3`; the difference is not an
/// inheritance error but the same boundary measured against a coarser format, where the regular
/// series loses its grip on an `O(a)` term sooner.
pub(super) const SMALL_SHAPE: f32 = 0.01;

/// `ln Γ(1+a) / a`.
///
/// No separate Taylor series, unlike the `f64` peer: [`LN_GAMMA1P_FACTORED`] already expresses
/// `lnΓ(1+a)` as `a(a−1)·g(a)`, so dividing by `a` is just `(a−1)·g(a)` — the `a` cancels
/// *symbolically* rather than arithmetically, which is better than a series would be. It removes
/// the division entirely, so the quotient cannot round, and it stays exact as `a → 0` where
/// dividing a subnormal by a subnormal would not.
fn ln_gamma1p_over_a(a: f32) -> f32 {
    let mut g = 0.0;
    for &coefficient in LN_GAMMA1P_FACTORED.iter().rev() {
        g = g * a + coefficient;
    }
    (a - 1.0) * g
}

/// Terms of the small-shape series. The `f64` peer takes 24.
///
/// The series is `Σ (−x)^k / (k!(a+k))` and runs over `x < 1 + a`, so with `a` at most `0.01`
/// the ratio between terms is below `1/k`: the `k`-th term is bounded by `1/(k!·k)`. That passes
/// under half an `f32` ulp of the leading term at `k = 9` (`1/(9!·9) = 3.1e-7`) and under a
/// quarter of one by `k = 10`, so twelve is already well past the point where adding terms
/// changes nothing — and twenty-four would just be the `f64` count copied across.
const SMALL_SHAPE_TERMS: u32 = 12;

/// DLMF 8.7.1 with the constant term separated, for `x < 1+a`.
fn small_shape_log_lower(a: f32, x: f32) -> f32 {
    let mut power = -x;
    let mut sum = power / (a + 1.0);
    for k in 2..=SMALL_SHAPE_TERMS {
        let k = k as f32;
        power *= -x / k;
        sum += power / (a + k);
    }
    let v = a * sum;
    // ln1p(v)/v rounds near one at this cutoff. Avoid dividing subnormal values.
    let log_ratio = if v.abs() <= f32::EPSILON {
        1.0
    } else {
        v.ln_1p() / v
    };
    a * (x.ln() - ln_gamma1p_over_a(a) + sum * log_ratio)
}

/// Which series or approximation evaluates the regularized incomplete gamma.
///
/// Four regimes on three thresholds, and naming them is what makes the pair below readable:
/// `gamma_lr` and `gamma_ur` share every branch, differing only in which of the two tails each
/// forms directly and which it takes as a complement. Written as nested `if`s — which is how
/// this file had it — that structure is invisible, and the small-shape case in particular read
/// as an afterthought when it is the one regime where neither ordinary route works.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IncompleteGammaMethod {
    /// Not a positive finite shape with a non-negative argument.
    Invalid,
    /// `x` is `0` or `+∞`, where both tails are exactly `0` or `1`.
    Limit,
    /// `a ≤ `[`SMALL_SHAPE`] and `x < a + 1`: the DLMF 8.7.1 series, which keeps the `O(a)`
    /// terms the ordinary routes drop. Part of the branch that used to widen to `f64`.
    SmallShapeSeries,
    /// `a ≤ `[`SMALL_SHAPE`] and `x ≥ a + 1`: the upper continued fraction with the tiny shape
    /// applied **after** it, so the probability takes a single subnormal rounding.
    ///
    /// Separate from [`Self::SmallShapeSeries`] because the series above is only valid for
    /// `x < 1 + a` — it is a Taylor expansion in `x`. Collapsing the two into one small-shape
    /// arm is a mistake that reads as harmless and is not: it put the series to work at `x = 4`,
    /// where `P + Q` came back as `1.0000009537`. Caught by
    /// `the_native_f32_small_shape_gamma_tails_are_complementary`.
    SmallShapeFraction,
    /// `a ≥ `[`GAMMA_EDGEWORTH_MIN`]: the Edgeworth-corrected normal, where both series fail.
    Edgeworth,
    /// `x < a + 1`: the lower-tail series converges fastest here.
    LowerSeries,
    /// `x ≥ a + 1`: the upper-tail continued fraction does.
    UpperFraction,
}

impl IncompleteGammaMethod {
    fn for_arguments(a: f32, x: f32) -> Self {
        if !valid_incomplete_gamma_inputs(a, x) {
            Self::Invalid
        } else if x == 0.0 || x.is_infinite() {
            Self::Limit
        } else if a <= SMALL_SHAPE {
            // The small shape still has to pick a tail: the series is an expansion in `x`.
            if use_lower_gamma_series(a, x) {
                Self::SmallShapeSeries
            } else {
                Self::SmallShapeFraction
            }
        } else if a >= GAMMA_EDGEWORTH_MIN {
            Self::Edgeworth
        } else if use_lower_gamma_series(a, x) {
            Self::LowerSeries
        } else {
            Self::UpperFraction
        }
    }
}

/// Regularized **lower** incomplete gamma `P(a, x) = γ(a, x) / Γ(a)`, `a > 0`, `x ≥ 0`.
pub(crate) fn gamma_lr(a: f32, x: f32) -> f32 {
    match IncompleteGammaMethod::for_arguments(a, x) {
        IncompleteGammaMethod::Invalid => f32::NAN,
        // `P(a, 0) = 0` and `P(a, ∞) = 1`. Both are needed explicitly: without them the series
        // below return `NaN` at an infinite argument, which propagates into `erf` and the whole
        // normal family built on it.
        IncompleteGammaMethod::Limit => f32::from(u8::from(x.is_infinite())),
        IncompleteGammaMethod::SmallShapeSeries => small_shape_log_lower(a, x).exp(),
        // Past the series' radius the lower tail is the complement of a small upper tail, which
        // is the orientation that keeps its digits.
        IncompleteGammaMethod::SmallShapeFraction => 1.0 - gamma_ur(a, x),
        IncompleteGammaMethod::Edgeworth => gamma_edgeworth(a, x).0,
        IncompleteGammaMethod::LowerSeries => {
            let series = gamma_lower_series(a, x, EPS, MAX_ITER);
            log_scaled_gamma(a, x, series.ln()).exp()
        }
        IncompleteGammaMethod::UpperFraction => 1.0 - gamma_ur(a, x),
    }
}

/// Regularized **upper** incomplete gamma `Q(a, x) = Γ(a, x) / Γ(a) = 1 - P(a, x)`.
pub(crate) fn gamma_ur(a: f32, x: f32) -> f32 {
    match IncompleteGammaMethod::for_arguments(a, x) {
        IncompleteGammaMethod::Invalid => f32::NAN,
        IncompleteGammaMethod::Limit => f32::from(u8::from(!x.is_infinite())),
        // `exp_m1` rather than `1 - exp`: the small-shape lower tail is close to one here, so
        // the complement would cancel away exactly the digits the branch exists to keep.
        IncompleteGammaMethod::SmallShapeSeries => -small_shape_log_lower(a, x).exp_m1(),
        IncompleteGammaMethod::SmallShapeFraction => {
            let h = gamma_upper_fraction(a, x, LENTZ_TINY_F32, EPS, MAX_ITER);
            // Γ(1+a) = a Γ(a). Multiply the tiny shape only after evaluating the fraction so the
            // probability undergoes a single subnormal rounding.
            a * (-x + a * (x.ln() - ln_gamma1p_over_a(a)) + h.ln()).exp()
        }
        IncompleteGammaMethod::Edgeworth => gamma_edgeworth(a, x).1,
        IncompleteGammaMethod::LowerSeries => 1.0 - gamma_lr(a, x),
        IncompleteGammaMethod::UpperFraction => {
            log_scaled_gamma(a, x, 0.0).exp()
                * gamma_upper_fraction(a, x, LENTZ_TINY_F32, EPS, MAX_ITER)
        }
    }
}

/// Argument above which the Stirling-type asymptotic series for `digamma`/`trigamma` is
/// evaluated; below it the recurrence walks the argument up to here first.
///
/// **Derived at `f32`'s epsilon, not inherited.** The `f64` peer walks to `20`, and copying that
/// would be the mistake this tree keeps making in the other direction: at `f32` it is not merely
/// wasteful but actively *worse*, because every recurrence step adds a rounding of its own and
/// the series stopped being the binding constraint long before `20`.
///
/// The series are truncated after the `x^{-8}` and `x^{-9}` terms, so the first *omitted* term
/// sets the truncation floor — for `digamma`, `1/(132x^10)`, which is `7.2e-9` by `x = 4`,
/// already inside the format. Measured against the exact closed forms
/// `ψ(n) = −γ + H_{n−1}` and `ψ₁(n) = ζ(2) − Σ_{k<n} k^{−2}`, worst absolute error
/// over `n ∈ [1, 40]`:
///
/// | shift | 2 | 3 | 4 | **6** | 10 | 20 |
/// |---|---|---|---|---|---|---|
/// | `digamma` | `4.6e-6` | `2.9e-7` | `2.9e-7` | **`2.9e-7`** | `2.9e-7` | `4.8e-7` |
/// | `trigamma` | `2.1e-5` | `3.7e-7` | `1.1e-7` | **`8.7e-9`** | `2.8e-8` | `3.8e-8` |
///
/// Two things to read off it. `digamma` flattens at `3` and stays flat: `2.9e-7` is about one ulp
/// of `ψ(40) = 3.67`, so that column is the format's resolution at the top of the sweep rather
/// than the truncation — which is why shifting further cannot help it, and why `20` makes it
/// *worse* by adding steps. `trigamma` does keep improving as far as `6`, by a factor of
/// thirteen, and then degrades for the same reason.
///
/// So `6` is where the pair is jointly best, and a nonnegative argument takes at most six
/// recurrence steps rather than twenty.
const PSI_ASYMPTOTIC_MIN: f32 = 6.0;

/// `ψ`'s only positive zero, `x₀ = 1.4616321449710...`, as `a₀ = x₀ − 1` in two `f32` pieces.
///
/// Two pieces because one is not enough, and the failure is a *sign* error rather than a
/// precision one. `a₀` rounds to `f32` with a residual of `1.24e-8`, and the factored form below
/// multiplies the offset by `h ≈ 1.25` — so a single-`f32` `a₀` puts the computed zero
/// `1.6e-8` away from the true one. The three representable arguments straddling `x₀` are only
/// `1.15e-7` apart in `ψ`, so that shift is an eighth of a step and it flips the middle one's
/// sign. Subtracting the residual separately costs one operation and fixes it: `a − A0_HIGH` is
/// exact by Sterbenz near the zero, so the second subtraction is the only rounding.
const PSI_ZERO_HIGH: f32 = 0.461_632_13;
const PSI_ZERO_LOW: f32 = 1.244_079_95e-8;

/// `h(a) = ψ(1+a) / (a − a₀)` on `a ∈ [0, 1]`, ascending powers.
///
/// # Why `ψ` on `[1, 2]` is not the recurrence
///
/// The recurrence-plus-asymptotic route builds `ψ(x)` as a sum of terms of magnitude `1` to `2`
/// — the shed `−1/x` terms and the series' `ln x`. On `[1, 2]` the *answer* is at most `0.58`
/// and passes through zero at `x₀`, so the absolute error is set by `ulp(2)`, not by the result.
/// Measured: the recurrence returned **exactly `0.0`** at two of the three representable
/// arguments either side of `x₀`, where the true values are `−1.20e-8` and `+1.03e-7`. It is the
/// same defect as the Lanczos log form in [`LN_GAMMA1P_FACTORED`], and it has the same fix.
///
/// `ψ` has a simple zero at `x₀`, so `ψ(1+a) = (a − a₀)·h(a)` with `h` smooth and `O(1)` —
/// `1.2504` at `a = 0`, `0.7853` at `a = 1`. The zero is then reproduced by construction instead
/// of by cancellation, and the three arguments above come back at `−1.2739e-7`, `−1.2039e-8` and
/// `+1.0332e-7` against MPFR's `−1.2739e-7`, `−1.2036e-8` and `+1.0332e-7`.
///
/// # Provenance
///
/// Degree 8, Chebyshev least-squares on `[0, 1]` against `h` evaluated to 50 decimal digits:
/// `ψ(1+a) = −γ + Σ_{k≥2} (−1)^k ζ(k) a^{k−1}` in `Decimal`, with the upper half reflected
/// through `ψ(2+e) = ψ(1+e) + 1/(1+e)`, `e = a − 1`, because that series is at its radius of
/// convergence at `a = 1` and returns `−0.143` there instead of `0.785`. `a₀` itself is located
/// by bisecting the same reference. Generated by a local tool; see CONTRIBUTING.md, "Reference
/// values are pasted in". The fit evaluated in `f32` holds to `2.7e-7` relative (4.6 ulp), and
/// the resulting `ψ` to `1.36e-7` absolute over `[1, 2]` — about one ulp of `ψ(1) = −0.577`.
const DIGAMMA_UNIT_H: [f32; 9] = [
    1.250_38,
    -0.854_667_7,
    0.751_740_16,
    -0.706_409_7,
    0.651_346_86,
    -0.529_657_36,
    0.329_364_63,
    -0.130_719_65,
    0.023_930_421,
];

/// `ψ(1+a)` for `a ∈ [0, 1]`, with the zero factored out. See [`DIGAMMA_UNIT_H`].
fn digamma_on_unit(a: f32) -> f32 {
    let mut h = 0.0;
    for &coefficient in DIGAMMA_UNIT_H.iter().rev() {
        h = h * a + coefficient;
    }
    ((a - PSI_ZERO_HIGH) - PSI_ZERO_LOW) * h
}

/// How `ψ` and `ψ₁` reach an argument the asymptotic series can be evaluated at.
///
/// The series is only valid for a large positive argument, so everything else is a reduction
/// onto that case — and the reductions are different functions of the input, not variations of
/// one formula, which is why they are named.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PsiReduction {
    /// No answer: `NaN`, or `−∞` where the function oscillates without limit.
    Undefined,
    /// A pole of the function. Whether that has a limit is per-function — `ψ₁` tends to `+∞`
    /// from both sides, while `ψ` changes sign across one — so the variant names the *input*
    /// and each caller says what it means.
    Pole,
    /// Reflect across the poles with the cotangent (`ψ`) or cosecant-squared (`ψ₁`) identity.
    Reflected,
    /// Walk up to [`PSI_ASYMPTOTIC_MIN`] by the recurrence, accumulating the shed terms.
    Recurrence,
    /// Already large enough; evaluate the series directly.
    Asymptotic,
}

impl PsiReduction {
    /// For `digamma`, whose reflection applies to the whole negative axis.
    fn for_digamma(x: f32) -> Self {
        if !x.is_finite() {
            // `ψ(+∞) = +∞` is handled by the series; `NaN` and `−∞` have no answer.
            return if x == f32::INFINITY {
                Self::Asymptotic
            } else {
                Self::Undefined
            };
        }
        if x < 0.0 {
            // A negative integer is a pole, and `x - x.round()` is the distance to the nearest
            // one — formed before multiplying by `π` so rounding cannot erase it.
            return if x == x.round() {
                Self::Pole
            } else {
                Self::Reflected
            };
        }
        if x < PSI_ASYMPTOTIC_MIN {
            Self::Recurrence
        } else {
            Self::Asymptotic
        }
    }

    /// For `trigamma`, whose reflection is taken below one half — its own fixed point — because
    /// squaring the sine makes the integer shift immaterial.
    fn for_trigamma(x: f32) -> Self {
        if !x.is_finite() {
            return if x == f32::INFINITY {
                Self::Asymptotic
            } else {
                Self::Undefined
            };
        }
        if x <= 0.0 && x == x.floor() {
            Self::Pole
        } else if x < 0.5 {
            Self::Reflected
        } else if x < PSI_ASYMPTOTIC_MIN {
            Self::Recurrence
        } else {
            Self::Asymptotic
        }
    }
}

/// The digamma function `ψ(x) = d/dx ln Γ(x)`.
///
/// Native, where this used to widen to `f64`. The algorithm is the `f64` peer's — reflection,
/// then recurrence, then a Stirling-type asymptotic series — with the shift threshold rederived
/// at this format's resolution; see [`PSI_ASYMPTOTIC_MIN`].
pub(crate) fn digamma(x: f32) -> f32 {
    match PsiReduction::for_digamma(x) {
        // `ψ` has opposite signs on the two sides of a pole, so no limit exists there — unlike
        // `ψ₁`, which diverges to `+∞` from both. Hence the same answer as an unusable argument.
        PsiReduction::Undefined | PsiReduction::Pole => f32::NAN,
        PsiReduction::Reflected => {
            // DLMF 5.5.4, with the cotangent's argument reduced first.
            let offset = x - x.round();
            digamma(1.0 - x) - core::f32::consts::PI / (core::f32::consts::PI * offset).tan()
        }
        // Below the asymptotic threshold, reduce onto `[1, 2]` and use the factored polynomial
        // rather than the series: see [`DIGAMMA_UNIT_H`] for why the series cannot be used where
        // `ψ` is small.
        PsiReduction::Recurrence => digamma_reduced(x),
        PsiReduction::Asymptotic => digamma_asymptotic(x),
    }
}

/// `ψ(x)` for `0 < x < PSI_ASYMPTOTIC_MIN`, by recurrence onto `[1, 2)`.
///
/// `ψ(x) = ψ(x−1) + 1/(x−1)` walks down and `ψ(x) = ψ(x+1) − 1/x` walks up from below one. Every
/// shed term is positive on the way down, so the accumulation does not cancel; below one the
/// answer is dominated by `−1/x` and nothing large is formed either.
fn digamma_reduced(x: f32) -> f32 {
    if x < 1.0 {
        return digamma_on_unit(x) - 1.0 / x;
    }
    let mut shifted = x;
    let mut shed = 0.0;
    while shifted >= 2.0 {
        shifted -= 1.0;
        shed += 1.0 / shifted;
    }
    digamma_on_unit(shifted - 1.0) + shed
}

/// `ψ` from its asymptotic series, for an argument at or above [`PSI_ASYMPTOTIC_MIN`].
fn digamma_asymptotic(x: f32) -> f32 {
    let inv = 1.0 / x;
    x.ln() - 0.5 * inv - digamma_correction(inv)
}

/// The Stirling correction to `ψ`, a polynomial in `1/x²`.
///
/// # Mutation testing is what asked for this
///
/// Two of its signs survived as an expression inside [`digamma_asymptotic`]. The deeper a term
/// sits in the Horner nest the less it contributes, and by the third level the contribution is at
/// or below one `f32` ulp of `ψ` itself — `1.4e-7` relative at `x = 6`, the smallest argument the
/// series ever sees, and less above it. Against the polynomial's *own* value the same flips are
/// `7.2e-5` and `2.1e-6`, which is two to three orders clear of the format. The nest is the whole
/// answer here and a rounding correction there; that difference is the test.
fn digamma_correction(inv: f32) -> f32 {
    let inv2 = inv * inv;
    inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 / 240.0)))
}

/// Walk `x` up to [`PSI_ASYMPTOTIC_MIN`], accumulating `shed` at each step.
///
/// Shared by both functions because the walk is identical and only the term differs; returning
/// the pair keeps the loop in one place rather than transcribed twice with one sign changed.
fn psi_recurrence(mut x: f32, term: impl Fn(f32) -> f32) -> (f32, f32) {
    let mut accumulated = 0.0;
    while x < PSI_ASYMPTOTIC_MIN {
        accumulated += term(x);
        x += 1.0;
    }
    (x, accumulated)
}

/// The trigamma function `ψ₁(x) = d²/dx² ln Γ(x)` — the derivative of [`digamma`].
pub(crate) fn trigamma(x: f32) -> f32 {
    match PsiReduction::for_trigamma(x) {
        PsiReduction::Undefined => f32::NAN,
        // `ψ₁` diverges to `+∞` at its poles from both sides, so the limit is well defined.
        PsiReduction::Pole => f32::INFINITY,
        PsiReduction::Reflected => {
            // DLMF 5.15.6. Squaring the sine makes its integer shift immaterial.
            let offset = x - x.round();
            let cosecant = core::f32::consts::PI / (core::f32::consts::PI * offset).sin();
            cosecant * cosecant - trigamma(1.0 - x)
        }
        PsiReduction::Recurrence | PsiReduction::Asymptotic => {
            let (shifted, shed) = psi_recurrence(x, |x| 1.0 / (x * x));
            shed + trigamma_asymptotic(shifted)
        }
    }
}

/// `ψ₁` from its asymptotic series, for an argument at or above [`PSI_ASYMPTOTIC_MIN`].
fn trigamma_asymptotic(x: f32) -> f32 {
    let r = 1.0 / x;
    let r2 = r * r;
    r + 0.5 * r2 + trigamma_tail(r)
}

/// The Bernoulli tail of `ψ₁`, past its two leading terms.
///
/// Named for the reason [`digamma_correction`] is: its innermost sign is a rounding correction to
/// `ψ₁` and the whole of its own value, so only a direct call separates the two.
fn trigamma_tail(r: f32) -> f32 {
    let r2 = r * r;
    r2 * r * (1.0 / 6.0 - r2 * (1.0 / 30.0 - r2 * (1.0 / 42.0 - r2 / 30.0)))
}

/// Single-precision [`crate::kernels::gamma::ln_rising_factorial`]; see that function for the
/// regime split.
pub(crate) fn ln_rising_factorial(a: f32, n: f32) -> f32 {
    if n == 0.0 {
        return 0.0;
    }
    if !a.is_finite() || !n.is_finite() {
        return ln_gamma(a + n) - ln_gamma(a);
    }
    if (1.0..=64.0).contains(&n) && n.fract() == 0.0 {
        let mut total = 0.0f32;
        for k in 0..n as u32 {
            total += (a + k as f32).ln();
        }
        return total;
    }
    // The threshold is lower than the f64 one because f32 loses the difference much sooner.
    if a.min(a + n) > 1e2 {
        let r = n / a;
        let l1p = r.ln_1p();
        return n * a.ln() + (a + n) * l1p - n - 0.5 * l1p + 1.0 / (12.0 * (a + n))
            - 1.0 / (12.0 * a);
    }
    ln_gamma(a + n) - ln_gamma(a)
}

#[cfg(test)]
mod tests;
