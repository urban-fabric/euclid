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
    gamma_edgeworth, gamma_lower_series, gamma_reflection, gamma_upper_fraction, lanczos_log_gamma,
    lanczos_reflected_gamma, log_gamma_reflection, log_scaled_gamma, reflection_sine,
    use_gamma_reflection, use_lower_gamma_series, valid_incomplete_gamma_inputs,
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
    } else if x >= 1.0 {
        reduced_gamma(x)
    } else {
        // `Γ(x) = Γ(1+x)/x` on the rest of `(0, 1)`. `ln_gamma1p_factored` takes `x` itself and
        // returns `lnΓ(1+x)`, so the shifted argument is never formed and never rounded — and
        // `|lnΓ(1+x)| ≤ 0.1214` there, so `exp` of it costs a hundredth of an ulp against the
        // several the Lanczos product was spending.
        ln_gamma1p_factored(x).exp() / x
    }
}

/// `Γ(x)` for `1 ≤ x ≤ `[`GAMMA_OVERFLOW_MIN`], by reduction to `[1, 2)`.
///
/// `Γ(x) = (x−1)(x−2)⋯r·Γ(r)` with `r ∈ [1, 2)`. Every subtraction is exact — the result lands in
/// a binade no coarser than `x`'s — so the only error is the product's roundings and `Γ(r)`'s,
/// and `Γ(r)` is `exp` of a logarithm bounded by `0.1214`, which costs a hundredth of an ulp.
///
/// # The Lanczos product's error here is arithmetic, not truncation
///
/// This arm is the `f64` peer's and arrives for a different reason. There, Lanczos's *truncation*
/// was the problem — `456` ulp by `x = 170`, reproduced exactly at 60 digits. In `f32` that same
/// truncation is about `1e-15` relative and so is three orders *below* `ε`; what costs here is
/// the product form's own arithmetic. [`lanczos_gamma`] takes a `powf`, an `exp`, a nine-term
/// rational sum and three multiplications, and a dozen `f32` roundings at half an ulp each is
/// several ulp of result.
///
/// It shows up most clearly where the answer is an exactly representable integer, since there is
/// then no reference to argue about. Measured against `Γ`'s exact values, in `f32` ulp:
///
/// ```text
/// x             1     2     3     4     5     6     10     12
/// Γ(x)          1     1     2     6    24   120 362880 39916800
/// Lanczos     3.0   3.0   1.0   5.3  10.0   0.0   17.3   10.9   ulp
/// reduced       0     0     0     0     0     0      0      0
/// ```
///
/// The `f64` kernel returns every one of those exactly, and `gamma`'s own `fixed_point` property
/// asserts `Γ(1) = 1` at zero ulp — for `f64`. Nothing had ever asked `f32`, because the
/// single-precision surface had never been swept; this is the first thing that sweep found.
///
/// # The product has to be accumulated *upward*, which `f64` never had to care about
///
/// Taking the factors as they fall — `(x−1)(x−2)⋯r` and then one multiplication by `Γ(r)` —
/// builds `Γ(x)/Γ(r)`, and `Γ(r) ≤ 1` on `[1, 2)`, so that partial product is **larger than the
/// answer**. In `f64` the arm stops at `STIRLING_MIN = 20` where `Γ(20)` is `1.2e17` against a
/// format limit of `1.8e308`, so it cannot matter. Here the arm runs to `35.1` and `Γ(35.03)` is
/// `3.34e38` against `f32`'s `3.40e38`: the first version overflowed to `inf` for every argument
/// in `[35.0347, 35.1]`, where the true value fits.
///
/// `no_f32_kernel_saturates_where_the_wide_peer_does_not` caught it, which is what that property
/// is for — it compares *classification* against the wide peer, so it sees a finite value turned
/// infinite even where it would not argue about digits.
///
/// Seeding with `Γ(r)` and multiplying by `r, r+1, …, x−1` in ascending order fixes it
/// structurally: every intermediate is `Γ(r+k)`, which increases monotonically to `Γ(x)` and so
/// never exceeds the answer. `remaining -= 1.0` and `factor += 1.0` are both exact here, since
/// every value involved is under `2^24`.
///
/// The loop runs at most thirty-four times, since it is entered only at or below
/// [`GAMMA_OVERFLOW_MIN`] of `35.1`, and a `NaN` cannot enter it — the comparison above is false
/// for one.
fn reduced_gamma(x: f32) -> f32 {
    let mut remaining = x;
    let mut steps = 0u32;
    while remaining >= 2.0 {
        remaining -= 1.0;
        steps += 1;
    }
    let mut product = ln_gamma(remaining).exp();
    let mut factor = remaining;
    for _ in 0..steps {
        product *= factor;
        factor += 1.0;
    }
    product
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

mod incomplete;
mod psi;

// Re-exported so the family keeps one path per function: callers reached
// `kernels::f32::gamma::gamma_lr` before the split and still do. `REFLECTED_ZERO_MIN`
// stays in this file because `gamma` itself uses it.
pub(super) use incomplete::GAMMA_EDGEWORTH_MIN;
pub(crate) use incomplete::{gamma_lr, gamma_ur};
pub(crate) use psi::{digamma, trigamma};

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
