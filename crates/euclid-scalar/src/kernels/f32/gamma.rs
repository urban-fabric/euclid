//! Single-precision gamma family. The Lanczos and incomplete-gamma kernels
//! use native `f32`. Digamma and trigamma reuse the double-precision kernels
//! to retain accuracy near roots and poles.

use crate::math::continued_fraction::LENTZ_TINY_F32;
use crate::math::gamma::{
    gamma_edgeworth, gamma_lower_series, gamma_reflection, gamma_upper_fraction, lanczos_gamma,
    lanczos_log_gamma, log_gamma_reflection, log_scaled_gamma, reflection_sine,
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
    if x.is_sign_positive() && x <= WIDE_SHAPE_MAX {
        return super::super::gamma::ln_gamma(f64::from(x)) as f32;
    }
    if use_gamma_reflection(x) {
        log_gamma_reflection(reflection_sine(x), ln_gamma(1.0 - x))
    } else {
        lanczos_log_gamma(x, G, &LANCZOS)
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
        // When the reflected gamma overflows, retain extra precision in the
        // negative tail and let the final narrowing round the subnormal.
        gamma_reflection(x, |_, _| super::super::gamma::gamma(f64::from(x)) as f32)
    } else if x > GAMMA_OVERFLOW_MIN {
        f32::INFINITY
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
const GAMMA_EDGEWORTH_MIN: f32 = 1e3;

/// Shapes for which cancellation warrants evaluating the gamma family in f64.
pub(super) const WIDE_SHAPE_MAX: f32 = 0.01;

/// Regularized **lower** incomplete gamma `P(a, x) = γ(a, x) / Γ(a)`, `a > 0`, `x ≥ 0`.
pub(crate) fn gamma_lr(a: f32, x: f32) -> f32 {
    if a <= WIDE_SHAPE_MAX {
        return super::super::gamma::gamma_lr(f64::from(a), f64::from(x)) as f32;
    }
    if !valid_incomplete_gamma_inputs(a, x) {
        return f32::NAN;
    }
    // P(a, x) -> 1 as x -> inf; see the f64 peer.
    if x.is_infinite() {
        return 1.0;
    }
    if x == 0.0 {
        return 0.0;
    }
    if a >= GAMMA_EDGEWORTH_MIN {
        return gamma_edgeworth(a, x).0;
    }
    if use_lower_gamma_series(a, x) {
        let series = gamma_lower_series(a, x, EPS, MAX_ITER);
        log_scaled_gamma(a, x, series.ln()).exp()
    } else {
        1.0 - gamma_ur(a, x)
    }
}

/// Regularized **upper** incomplete gamma `Q(a, x) = Γ(a, x) / Γ(a) = 1 - P(a, x)`.
pub(crate) fn gamma_ur(a: f32, x: f32) -> f32 {
    if a <= WIDE_SHAPE_MAX {
        return super::super::gamma::gamma_ur(f64::from(a), f64::from(x)) as f32;
    }
    if !valid_incomplete_gamma_inputs(a, x) {
        return f32::NAN;
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
        return 1.0 - gamma_lr(a, x);
    }
    log_scaled_gamma(a, x, 0.0).exp() * gamma_upper_fraction(a, x, LENTZ_TINY_F32, EPS, MAX_ITER)
}

/// The digamma function `ψ(x) = d/dx ln Γ(x)`.
pub(crate) fn digamma(x: f32) -> f32 {
    super::super::gamma::digamma(f64::from(x)) as f32
}

/// The trigamma function `ψ₁(x) = d²/dx² ln Γ(x)` — the derivative of [`digamma`].
pub(crate) fn trigamma(x: f32) -> f32 {
    super::super::gamma::trigamma(f64::from(x)) as f32
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
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    /// Cross-check the native `f32` result against the `f64` reference, narrowed — the guard
    /// against the two implementations silently drifting apart.
    fn agrees(a: f32, b: f64, tol: f32) {
        let b = b as f32;
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn gamma_matches_f64_reference() {
        for &x in &[0.3, 0.7, 1.5, 3.2, 5.0, 10.0] {
            agrees(gamma(x), crate::kernels::gamma::gamma(x as f64), 1e-5);
            agrees(ln_gamma(x), crate::kernels::gamma::ln_gamma(x as f64), 1e-5);
        }
    }

    #[test]
    fn incomplete_gamma_matches_f64_reference() {
        for &(a, x) in &[(1.0, 0.5), (2.5, 3.0), (5.0, 2.0), (10.0, 12.0)] {
            let p = gamma_lr(a, x);
            let q = gamma_ur(a, x);
            agrees(p, crate::kernels::gamma::gamma_lr(a as f64, x as f64), 1e-5);
            agrees(p + q, 1.0, 1e-6);
        }
    }

    #[test]
    fn polygamma_matches_f64_reference() {
        for &x in &[0.5, 1.0, 2.0, 3.3, 7.5] {
            agrees(digamma(x), crate::kernels::gamma::digamma(x as f64), 1e-5);
            agrees(trigamma(x), crate::kernels::gamma::trigamma(x as f64), 1e-5);
        }
    }
}
