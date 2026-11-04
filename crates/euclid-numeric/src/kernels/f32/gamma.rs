//! Native `f32` gamma family — the single-precision port of [`mod@super::super::gamma`]. Same
//! Lanczos / Numerical-Recipes algorithms, evaluated in `f32` throughout (no `f64`
//! round-trip), with the continued-fraction tolerance loosened to the `f32` noise floor.

use crate::constants::f32::SQRT_TAU;
use core::f32::consts::PI;

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

/// Natural log of the gamma function, `ln Γ(x)`, for `x > 0` (and, via reflection,
/// `x < 0` away from the non-positive integers).
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
    if x < 0.5 {
        (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = LANCZOS[0];
        let t = x + G + 0.5;
        for (i, &c) in LANCZOS.iter().enumerate().skip(1) {
            a += c / (x + i as f32);
        }
        0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
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
    if x < 0.5 {
        PI / ((PI * x).sin() * gamma(1.0 - x))
    } else if x > GAMMA_OVERFLOW_MIN {
        f32::INFINITY
    } else {
        let x = x - 1.0;
        let mut a = LANCZOS[0];
        let t = x + G + 0.5;
        for (i, &c) in LANCZOS.iter().enumerate().skip(1) {
            a += c / (x + i as f32);
        }
        // `t^(x+0.5)·e^{−t}` is formed as the square of its own square root, which looks
        // gratuitous and is not. Written directly, `t^(x+0.5)` overflows *long* before the
        // product does, because `e^{−t}` is a large compensating underflow — so the
        // intermediate leaves the format's range while the answer is still comfortably
        // inside it. Measured against exact factorials, the direct form returned:
        //
        //   `gamma(26)` through `gamma(35)` as `inf`, where the true values
        //   `1.55e25` through `2.95e38` are all representable
        //
        // and once `e^{−t}` underflowed to zero as well, `inf · 0 = NaN` — for `x ≥ 98`,
        // and for large negative arguments through the reflection below.
        //
        // Each half-power is the geometric mean of the two extremes, so it stays in range
        // until the result itself does; squaring then costs one extra rounding and turns the
        // overflow into a clean `inf · inf = inf` rather than a `NaN`.
        let half = t.powf((x + 0.5) * 0.5) * (-t * 0.5).exp();
        (2.0 * PI).sqrt() * half * half * a
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

/// `(P, Q)` from the Edgeworth-corrected normal approximation; see
/// [`GAMMA_EDGEWORTH_MIN`].
fn gamma_edgeworth(a: f32, x: f32) -> (f32, f32) {
    let root = a.sqrt();
    let z = (x - a) / root;
    let density = (-0.5 * z * z).exp() / SQRT_TAU;
    let correction = density * (z * z - 1.0) / (3.0 * root);
    let lower = 0.5 * super::erf::erfc(-z * core::f32::consts::FRAC_1_SQRT_2) - correction;
    let upper = 0.5 * super::erf::erfc(z * core::f32::consts::FRAC_1_SQRT_2) + correction;
    (lower.clamp(0.0, 1.0), upper.clamp(0.0, 1.0))
}

/// Regularized **lower** incomplete gamma `P(a, x) = γ(a, x) / Γ(a)`, `a > 0`, `x ≥ 0`.
pub(crate) fn gamma_lr(a: f32, x: f32) -> f32 {
    if a.is_nan() || x.is_nan() {
        return f32::NAN;
    }
    // P(a, x) -> 1 as x -> inf; see the f64 peer.
    if x.is_infinite() && x > 0.0 {
        return 1.0;
    }
    if a <= 0.0 || x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if a >= GAMMA_EDGEWORTH_MIN {
        return gamma_edgeworth(a, x).0;
    }
    if x < a + 1.0 {
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..MAX_ITER {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * EPS {
                break;
            }
        }
        (sum.ln() - x + a * x.ln() - ln_gamma(a)).exp()
    } else {
        1.0 - gamma_ur(a, x)
    }
}

/// Regularized **upper** incomplete gamma `Q(a, x) = Γ(a, x) / Γ(a) = 1 - P(a, x)`.
pub(crate) fn gamma_ur(a: f32, x: f32) -> f32 {
    if a <= 0.0 || x < 0.0 || a.is_nan() || x.is_nan() {
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
    if x < a + 1.0 {
        return 1.0 - gamma_lr(a, x);
    }
    let tiny = 1e-30;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..MAX_ITER {
        let an = -(i as f32) * (i as f32 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// The digamma function `ψ(x) = d/dx ln Γ(x)`.
pub(crate) fn digamma(mut x: f32) -> f32 {
    let mut result = 0.0;
    while x < 10.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + x.ln()
        - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 / 240.0)))
}

/// The trigamma function `ψ₁(x) = d²/dx² ln Γ(x)` — the derivative of [`digamma`].
pub(crate) fn trigamma(x: f32) -> f32 {
    if x <= 0.0 && x == x.floor() {
        return f32::INFINITY;
    }
    if x < 0.5 {
        let s = (PI * x).sin();
        return PI * PI / (s * s) - trigamma(1.0 - x);
    }
    let mut x = x;
    let mut result = 0.0;
    while x < 10.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let r = 1.0 / x;
    let r2 = r * r;
    result + r + 0.5 * r2 + r2 * r * (1.0 / 6.0 - r2 * (1.0 / 30.0 - r2 * (1.0 / 42.0 - r2 / 30.0)))
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
    if n > 0.0 && n <= 64.0 && n.fract() == 0.0 {
        let mut total = 0.0f32;
        let mut k = 0.0f32;
        while k < n {
            total += (a + k).ln();
            k += 1.0;
        }
        return total;
    }
    // The threshold is lower than the f64 one because f32 loses the difference much sooner.
    if a > 1e2 && a + n > 1e2 {
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
