//! The modified Bessel functions of the **second kind**, `K₀` and `K₁`, with their
//! exponentially-scaled forms `Kₙᵉ(x) = e^{x}Kₙ(x)`.
//!
//! # Method: the trapezoidal rule on `Kν(x) = ∫₀^∞ e^{−x cosh t} cosh(νt) dt`
//!
//! `K` is the awkward member of the family. Its power series (below) carries a `ln(x/2)`
//! factor and is exact only for small argument; its asymptotic expansion needs `x ≳ 20` before
//! it reaches machine precision, because an optimally-truncated asymptotic series for `K`
//! errs by about `e^{−2x}`. That leaves a wide middle range — exactly the range most callers
//! use — where neither is good enough.
//!
//! The integral representation closes it, and does so *uniformly*. Three properties make the
//! trapezoidal rule the right quadrature for it:
//!
//! - **The integrand is even in `t`**, so `∫₀^∞ = ½∫_{−∞}^{∞}` and the rule inherits the
//!   exponential convergence the trapezoidal rule enjoys on the whole line — error `≈ e^{−2πd/h}`
//!   for an integrand analytic in the strip `|Im t| < d` — rather than the `O(h²)` an endpoint
//!   would impose.
//! - **The integrand is positive**, so there is no cancellation at any argument. The result is
//!   a sum of positive terms, which is the opposite of the `Si`/`Ci` situation.
//! - **Both the width and the decay scale with `x` in the same way**, so the node count stays
//!   bounded (~40) rather than growing. For small `x` the integrand is a broad plateau out to
//!   `t ≈ ln(2/x)` with double-exponential tails, and `d → π/2` bounds the step. For large `x`
//!   it is a narrow Gaussian of width `1/√x`, which instead bounds the step by `≈0.66/√x`;
//!   taking the smaller of the two is correct in both regimes.
//!
//! Writing it for the **scaled** function `Kνᵉ(x) = e^{x}Kν(x) = ∫₀^∞ e^{−x(cosh t − 1)}cosh(νt)dt`
//! makes the scaled forms primitive and overflow-free, with the unscaled ones recovered by a
//! single multiplication that is *allowed* to underflow — as the true `K₀(746)` does.
//!
//! `ν` is a parameter rather than fixed at 0 and 1 because the same quadrature is the
//! machine-precision route to the Airy functions on the positive axis, through
//! `Ai(x) = π^{−1}√(x/3)·K_{1/3}(⅔x^{3/2})`.

use super::ln_half_x;
use super::modified_first::{i0, i1};
use crate::constants::f64::EULER_GAMMA;

/// Iteration cap for the power series here.
const MAX_TERMS: u32 = 1_000;

/// Relative size at which a series term stops contributing.
const TERM_EPS: f64 = 1e-20;

/// `pub(super)` rather than private: the unit tests live in a sibling `tests` module because
/// implementation plus tests exceeded this crate's 400-line file budget, and they compare the
/// two routes against each other directly.
///
/// At or below this argument `K₀`/`K₁` use their power series; above it, the quadrature.
///
/// The series is exact for small `x` and the quadrature's node count grows like `ln(1/x)` as
/// `x → 0`, so the series takes the small end. At `x = 2` the series has lost about one digit
/// to the `ln(x/2)·I₀` subtraction and the quadrature is already at machine precision.
pub(super) const K_SERIES_MAX: f64 = 2.0;

/// Largest trapezoid step, set by the strip half-width `d → π/2` that bounds the small-`x`
/// regime: the error is `≈ e^{−2πd/h} = e^{−π²/h}`, and `e^{−π²/0.2} ≈ 4e-22`.
const K_STEP_MAX: f64 = 0.2;

/// Step coefficient for the large-`x` regime, where the integrand is a Gaussian of width
/// `σ = 1/√x` and the trapezoid error is `≈ e^{−2π²σ²/h²}`. That needs `h < 0.66σ`; `0.5`
/// leaves margin.
const K_STEP_COEFF: f64 = 0.5;

/// Truncation level: integrate while `x(cosh t − 1) < 45`, so the discarded tail is below
/// `e^{−45} ≈ 3e-20` relative to an `O(1)` result.
const K_DECAY: f64 = 45.0;

/// `Kνᵉ(x) = e^{x}Kν(x)` for `x > 0` and any `ν ≥ 0`, by the trapezoidal rule.
///
/// # Cost at small argument
///
/// Accurate at every argument — measured `3e-16` relative down to `x = 1e-300` — but *slow*
/// there: the integrand is a plateau out to `t ≈ ln(2/x)`, so the node count grows like
/// `ln(1/x)` and reaches ~6300 at `x = 1e-300`, about 390× the ~40 nodes a moderate argument
/// needs. Callers on this crate's surface never see that, because `k0`/`k1` take the power
/// series below [`K_SERIES_MAX`]; it is worth knowing before reaching for `ν` outside
/// `{0, 1}` at a tiny argument.
///
/// `ν` above 1 works and is exact — the half-integer closed forms `K_{1/2}`, `K_{3/2}` and
/// `K_{5/2}` are checked against it in the relationship suite — since `cosh(νt)` is always
/// dominated by `e^{−x cosh t}` at the truncation point.
///
/// See the discussion above. `cosh t − 1` is formed as `2 sinh²(t/2)`: the direct subtraction
/// cancels for small `t`, and small `t` is precisely the regime the large-`x` branch lives in,
/// where that error would then be amplified by the factor `x` in the exponent.
pub(crate) fn k_nu_scaled(nu: f64, x: f64) -> f64 {
    debug_assert!(x > 0.0 && x.is_finite());
    let step = K_STEP_MAX.min(K_STEP_COEFF / x.sqrt());
    // Solve `cosh T − 1 = K_DECAY/x` for the truncation point. Spelling that as
    // `acosh(1 + K_DECAY/x)` would be a cancellation bug rather than a mere loss of digits:
    // once `x > K_DECAY/ε ≈ 2e17` the sum `1 + K_DECAY/x` rounds to exactly 1, `acosh` returns
    // 0, and the quadrature silently integrates nothing. Using the half-angle form
    // `cosh T − 1 = 2 sinh²(T/2)` instead gives `T = 2 asinh(√(K_DECAY/2x))`, which is exact
    // across the whole range.
    let t_max = 2.0 * (0.5 * K_DECAY / x).sqrt().asinh();
    let nodes = (t_max / step).ceil();

    // The t = 0 node, halved as the trapezoidal rule requires; the integrand there is 1.
    let mut total = 0.5;
    let mut j = 1.0f64;
    while j <= nodes {
        let t = step * j;
        let half = t * 0.5;
        let cosh_minus_one = 2.0 * half.sinh() * half.sinh();
        total += (-x * cosh_minus_one).exp() * (nu * t).cosh();
        j += 1.0;
    }
    step * total
}

/// `K₀(x)` by its power series, for small `x`.
///
/// `K₀(x) = −(ln(x/2) + γ)I₀(x) + Σ_{k≥1} (x²/4)^k/(k!)²·H_k`, with `H_k` the `k`th harmonic
/// number. Every term of the sum is positive; the only subtraction is the leading one, which
/// costs about a digit by `x = 2` and nothing below it.
pub(super) fn k0_series(x: f64) -> f64 {
    let q = x * x * 0.25;
    let mut term = 1.0f64; // (x²/4)^k/(k!)²
    let mut harmonic = 0.0f64;
    let mut sum = 0.0f64;
    for k in 1..MAX_TERMS {
        let kf = f64::from(k);
        term *= q / (kf * kf);
        harmonic += 1.0 / kf;
        let t = term * harmonic;
        sum += t;
        if t < sum * TERM_EPS {
            break;
        }
    }
    // `ln(x/2)` is formed as `ln x − ln 2` rather than `(x*0.5).ln()`. The product form
    // underflows for the smallest subnormals — `x*0.5` rounds to zero at `x = 5e-324`, making
    // `ln` return `−∞` where the true value is a perfectly representable `−474` — and the
    // difference is marginally more accurate everywhere else too.
    -(ln_half_x(x) + EULER_GAMMA) * i0(x) + sum
}

/// `K₁(x)` by its power series, for small `x`.
///
/// `K₁(x) = 1/x + ln(x/2)·I₁(x) − (x/4)Σ_{k≥0}(H_k + H_{k+1} − 2γ)(x²/4)^k/(k!(k+1)!)`.
/// The `1/x` carries the pole, so it is kept as a separate exact term rather than folded in.
pub(super) fn k1_series(x: f64) -> f64 {
    let q = x * x * 0.25;
    let mut term = 1.0f64; // (x²/4)^k/(k!(k+1)!)
    let mut h_k = 0.0f64; // H_k
    let mut sum = 0.0f64;
    for k in 0..MAX_TERMS {
        let kf = f64::from(k);
        if k > 0 {
            term *= q / (kf * (kf + 1.0));
            h_k += 1.0 / kf;
        }
        let h_k1 = h_k + 1.0 / (kf + 1.0);
        let t = term * (h_k + h_k1 - 2.0 * EULER_GAMMA);
        sum += t;
        if t.abs() < sum.abs() * TERM_EPS && k > 1 {
            break;
        }
    }
    1.0 / x + ln_half_x(x) * i1(x) - x * 0.25 * sum
}

/// The modified Bessel function of the second kind `K₀(x)`, for `x > 0`.
///
/// Underflows to `0` for `x ≳ 746`, as the true value does; use [`k0e`] there.
pub(crate) fn k0(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return f64::NAN; // the real K is defined for positive argument
    }
    if x == 0.0 {
        return f64::INFINITY; // logarithmic singularity
    }
    if x.is_infinite() {
        return 0.0;
    }
    if x <= K_SERIES_MAX {
        return k0_series(x);
    }
    k_nu_scaled(0.0, x) * (-x).exp()
}

/// The modified Bessel function of the second kind `K₁(x)`, for `x > 0`.
pub(crate) fn k1(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY; // simple pole, K₁ ~ 1/x
    }
    if x.is_infinite() {
        return 0.0;
    }
    if x <= K_SERIES_MAX {
        return k1_series(x);
    }
    k_nu_scaled(1.0, x) * (-x).exp()
}

/// The exponentially-scaled `K₀ᵉ(x) = e^{x}K₀(x)`, finite for every positive argument.
pub(crate) fn k0e(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x.is_infinite() {
        return 0.0; // K₀ᵉ ~ sqrt(pi/2x) -> 0
    }
    if x <= K_SERIES_MAX {
        return k0_series(x) * x.exp();
    }
    k_nu_scaled(0.0, x)
}

/// The exponentially-scaled `K₁ᵉ(x) = e^{x}K₁(x)`.
pub(crate) fn k1e(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x.is_infinite() {
        return 0.0;
    }
    if x <= K_SERIES_MAX {
        return k1_series(x) * x.exp();
    }
    k_nu_scaled(1.0, x)
}
