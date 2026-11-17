//! The gamma function family: `ln_gamma`/`gamma` (Lanczos), the regularized lower/upper
//! incomplete gamma `gamma_lr`/`gamma_ur` (the Poisson/Gamma/chi-squared CDF kernels), and
//! the polygamma `digamma`/`trigamma`.

use crate::math::continued_fraction::LENTZ_TINY_F64;
use crate::math::gamma::{
    gamma_edgeworth, gamma_lower_series, gamma_reflection, gamma_upper_fraction, lanczos_gamma,
    lanczos_log_gamma, lanczos_reflected_gamma, log_gamma_reflection, log_scaled_gamma,
    reflection_sine, use_gamma_reflection, use_lower_gamma_series, valid_incomplete_gamma_inputs,
};
use core::f64::consts::PI;

/// Lanczos `g` parameter and coefficients (g = 7, n = 9) — ~15-digit accuracy.
const G: f64 = 7.0;
const LANCZOS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Above this argument `Γ(x)` exceeds the format's range, and ``+∞`` is returned directly.
///
/// `Γ` overflows where `lnΓ(x)` passes `ln(f64::MAX) = 709.78`, which is `x ≈ 171.62`. The
/// threshold sits just above that, so nothing representable is saturated: `Γ(171)` is
/// `7.26e306` and is returned in full.
///
/// Needed as an explicit test rather than left to the arithmetic. The Lanczos form below keeps
/// its intermediates in range only while the *result* is in range, which is exactly what makes
/// it accurate — but once `x` is far past the boundary the half-power overflows while `e^{−t/2}`
/// underflows, and `inf · 0` is `NaN`. Saturating deliberately is both correct and cheaper.
const GAMMA_OVERFLOW_MIN: f64 = 171.7;

/// Natural log of the gamma function on intervals where `Γ(x)` is positive.
pub(crate) fn ln_gamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // ln Gamma grows without bound, so the limit at +inf is +inf rather than NaN.
    if x.is_infinite() {
        return if x > 0.0 { f64::INFINITY } else { f64::NAN };
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
        return f64::INFINITY;
    }
    if x.is_sign_positive() && x <= SMALL_SHAPE {
        return x * ln_gamma1p_over_a(x) - x.ln();
    }
    if use_gamma_reflection(x) {
        log_gamma_reflection(reflection_sine(x), ln_gamma(1.0 - x))
    } else {
        lanczos_log_gamma(x, G, &LANCZOS)
    }
}

/// The gamma function `Γ(x)`.
pub(crate) fn gamma(x: f64) -> f64 {
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
            f64::INFINITY.copysign(x)
        } else {
            f64::NAN
        };
    }
    if use_gamma_reflection(x) {
        gamma_reflection(x, |sin, reflected| {
            lanczos_reflected_gamma(sin, reflected, G, &LANCZOS)
        })
    } else if x > GAMMA_OVERFLOW_MIN {
        f64::INFINITY
    } else {
        lanczos_gamma(x, G, &LANCZOS)
    }
}

/// Iteration cap for the incomplete-gamma series and continued fraction.
///
/// Both need roughly `O(sqrt(a))` terms near `x ~ a`, so a small cap silently truncates at
/// large shape: at 200 the series stopped short from about `a = 1e3`, leaving `P(1e5, median)`
/// at `0.237` instead of `0.5` and putting a visible discontinuity at the `x = a + 1` branch
/// switch. All the series terms are positive, so this is pure truncation with no cancellation —
/// raising the cap fixes it outright, and costs nothing at small shape because both loops
/// break as soon as the relative term falls below `EPS`.
const MAX_ITER: usize = 100_000;
const EPS: f64 = 1e-15;

/// Below this shape the complement needs a series that retains terms of order `a`.
const SMALL_SHAPE: f64 = 1e-3;

/// `ln Γ(1+a) / a` from the Taylor series at zero. The omitted term is
/// smaller than 2e-19 throughout the small-shape branch.
fn ln_gamma1p_over_a(a: f64) -> f64 {
    -crate::constants::f64::EULER_GAMMA
        + a * (0.822_467_033_424_113_2
            + a * (-0.400_685_634_386_531_4
                + a * (0.270_580_808_427_784_55
                    + a * (-0.207_385_551_028_674 + a * 0.169_557_176_997_408_2))))
}

/// DLMF 8.7.1 with the constant term separated, for `x < 1+a`.
fn small_shape_log_lower(a: f64, x: f64) -> f64 {
    let mut power = -x;
    let mut sum = power / (a + 1.0);
    for k in 2..=24 {
        let k = f64::from(k);
        power *= -x / k;
        sum += power / (a + k);
    }
    let v = a * sum;
    // ln1p(v)/v rounds near one at this cutoff. Avoid dividing subnormal values.
    let log_ratio = if v.abs() <= f64::EPSILON {
        1.0
    } else {
        v.ln_1p() / v
    };
    a * (x.ln() - ln_gamma1p_over_a(a) + sum * log_ratio)
}

/// At or above this shape the incomplete gamma switches to an Edgeworth-corrected normal
/// approximation.
///
/// Both of the series routes fail above it, and they fail *quietly*. Measured `P(a, a)`
/// against its known asymptote `½ + 1/(3√(2πa))`:
///
/// ```text
/// a = 1e7   2.9e-8        a = 1e9.5  7.5e-2
/// a = 1e8   3.4e-7        a = 1e10   3.2e-1
/// a = 1e9   1.6e-3        a = 1e11   7.5e-1
/// ```
///
/// and further out it stops being a probability at all: `P(1e15, a + √a)` came back as
/// `−9.06`, and from `a = 1e16` as `NaN`. Two mechanisms compound — the series needs `O(√a)`
/// terms and runs into [`MAX_ITER`], and the front factor `exp(a ln x − x − lnΓ(a))` subtracts
/// three `O(a ln a)` quantities, which past `a ≈ 1e14` leaves nothing at all.
///
/// A `Gamma(a, 1)` standardised by its own mean and variance has skewness `2/√a`, so the
/// Edgeworth expansion
///
/// ```text
/// P(a, x) ≈ Φ(z) − φ(z)·(z² − 1)/(3√a),      z = (x − a)/√a
/// ```
///
/// has absolute error `O(1/a)` — *improving* exactly where the series collapses, and exact in
/// the limit. Verified against the series over `a ∈ [1e6, 1e8]` and `z ∈ [−3, 6]`, where the
/// measured discrepancy is `1.8e-5`, `1.8e-6`, `3.4e-7` — one order per decade of `a`, as
/// `O(1/a)` predicts. At the threshold both routes carry about `3e-8`, so the join costs
/// nothing; above it the approximation is the more accurate of the two.
///
/// No tail accuracy is given up. `Q` decays like `e^{−a·D(x/a)}`, which underflows for any
/// `a` past a few thousand once `x` is away from `a` — `Q(1e6, 2e6)` is already exactly zero.
pub(crate) const GAMMA_EDGEWORTH_MIN: f64 = 3e7;

/// Regularized **lower** incomplete gamma `P(a, x) = γ(a, x) / Γ(a)`, `a > 0`, `x ≥ 0`.
pub(crate) fn gamma_lr(a: f64, x: f64) -> f64 {
    if !valid_incomplete_gamma_inputs(a, x) {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    // P(a, x) -> 1 as x -> inf. Without this the series/continued fraction below produce NaN
    // at an infinite argument, which then propagates into everything built on `gamma_lr`
    // (`erf`, and through it the whole normal family).
    if x.is_infinite() {
        return 1.0;
    }
    if a >= GAMMA_EDGEWORTH_MIN {
        return gamma_edgeworth(a, x).0;
    }
    if use_lower_gamma_series(a, x) {
        if a <= SMALL_SHAPE {
            return small_shape_log_lower(a, x).exp();
        }
        let series = gamma_lower_series(a, x, EPS, MAX_ITER);
        log_scaled_gamma(a, x, series.ln()).exp()
    } else {
        1.0 - gamma_ur(a, x)
    }
}

/// Regularized **upper** incomplete gamma `Q(a, x) = Γ(a, x) / Γ(a) = 1 - P(a, x)`.
pub(crate) fn gamma_ur(a: f64, x: f64) -> f64 {
    if !valid_incomplete_gamma_inputs(a, x) {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    // Q(a, x) -> 0 as x -> inf, the complement of `gamma_lr`'s limit.
    if x.is_infinite() {
        return 0.0;
    }
    if a >= GAMMA_EDGEWORTH_MIN {
        return gamma_edgeworth(a, x).1;
    }
    if use_lower_gamma_series(a, x) {
        if a <= SMALL_SHAPE {
            return -small_shape_log_lower(a, x).exp_m1();
        }
        return 1.0 - gamma_lr(a, x);
    }
    let h = gamma_upper_fraction(a, x, LENTZ_TINY_F64, EPS, MAX_ITER);
    if a <= SMALL_SHAPE {
        // Γ(1+a) = a Γ(a). Multiply the tiny shape only after evaluating the
        // fraction so the probability undergoes a single subnormal rounding.
        return a * (-x + a * (x.ln() - ln_gamma1p_over_a(a)) + h.ln()).exp();
    }
    log_scaled_gamma(a, x, 0.0).exp() * h
}

/// Argument above which the Stirling-type asymptotic series for `digamma`/`trigamma` is
/// evaluated; below it the recurrence walks the argument up to here first.
///
/// This was `10`, which left both functions a thousand times weaker than the rest of the crate
/// for no good reason. Their series are truncated after the `x^{-8}` and `x^{-9}` terms, so the
/// first *omitted* term sets the floor: for `digamma` that is `1/(132x^10)`, which is `7.6e-13`
/// at `x = 10` — and the measured error was `7.4e-13`, i.e. entirely this.
///
/// Walking to `20` instead puts the omitted term at `1/(132·20^10) = 7.4e-16`, below `ε`.
/// Measured against the exact closed forms `ψ(n) = −γ + H_{n−1}` and
/// `ψ₁(n) = ζ(2) − Σ_{k<n} k^{−2}`:
///
/// | shift | 10 | 15 | **20** | 25 | 30 |
/// |---|---|---|---|---|---|
/// | `digamma` | `7.4e-13` | `1.3e-14` | **`8.8e-16`** | `4.4e-16` | `8.9e-16` |
/// | `trigamma` | `7.0e-12` | `1.2e-13` | **`8.5e-15`** | `8.5e-15` | `8.5e-15` |
///
/// It flattens at 20 — past that the series is at its own floor and further shifting only costs
/// recurrence steps. A nonnegative argument takes at most twenty steps.
const PSI_ASYMPTOTIC_MIN: u8 = 20;

/// The digamma function `ψ(x) = d/dx ln Γ(x)`.
pub(crate) fn digamma(mut x: f64) -> f64 {
    if x < 0.0 {
        // DLMF 5.5.4. Reduce the cotangent's argument before multiplying by π
        // so rounding cannot erase the distance to the nearest pole.
        let offset = x - x.round();
        if offset == 0.0 || !offset.is_finite() {
            return f64::NAN;
        }
        return digamma(1.0 - x) - PI / (PI * offset).tan();
    }
    let mut result = 0.0;
    for _ in 0..PSI_ASYMPTOTIC_MIN {
        if x < f64::from(PSI_ASYMPTOTIC_MIN) {
            result -= 1.0 / x;
            x += 1.0;
        } else {
            break;
        }
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + x.ln()
        - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 / 240.0)))
}

/// The trigamma function `ψ₁(x) = d²/dx² ln Γ(x)` — the derivative of [`digamma`].
pub(crate) fn trigamma(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() {
        return f64::INFINITY;
    }
    if x < 0.5 {
        // DLMF 5.15.6. Squaring the sine makes its integer shift immaterial.
        let offset = x - x.round();
        let cosecant = PI / (PI * offset).sin();
        return cosecant * cosecant - trigamma_positive(1.0 - x);
    }
    trigamma_positive(x)
}

/// Trigamma after reflection, where the recurrence moves towards positive infinity.
fn trigamma_positive(mut x: f64) -> f64 {
    let mut result = 0.0;
    for _ in 0..PSI_ASYMPTOTIC_MIN {
        if x < f64::from(PSI_ASYMPTOTIC_MIN) {
            result += 1.0 / (x * x);
            x += 1.0;
        } else {
            break;
        }
    }
    let r = 1.0 / x;
    let r2 = r * r;
    result + r + 0.5 * r2 + r2 * r * (1.0 / 6.0 - r2 * (1.0 / 30.0 - r2 * (1.0 / 42.0 - r2 / 30.0)))
}
mod rising_factorial;
pub(crate) use rising_factorial::ln_rising_factorial;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests;
