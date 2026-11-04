//! Native `f32` gamma family — the single-precision port of [`mod@super::super::gamma`]. Same
//! Lanczos / Numerical-Recipes algorithms, evaluated in `f32` throughout (no `f64`
//! round-trip), with the continued-fraction tolerance loosened to the `f32` noise floor.

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

/// Natural log of the gamma function, `ln Γ(x)`, for `x > 0` (and, via reflection,
/// `x < 0` away from the non-positive integers).
pub(crate) fn ln_gamma(x: f32) -> f32 {
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
    if x < 0.5 {
        PI / ((PI * x).sin() * gamma(1.0 - x))
    } else {
        let x = x - 1.0;
        let mut a = LANCZOS[0];
        let t = x + G + 0.5;
        for (i, &c) in LANCZOS.iter().enumerate().skip(1) {
            a += c / (x + i as f32);
        }
        (2.0 * PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a
    }
}

const MAX_ITER: usize = 200;
// Loosened from the `f64` `1e-15`: a relative tolerance below `f32::EPSILON` (~1.2e-7) can
// never be met, so the series/continued fractions would spin to `MAX_ITER`. `1e-6` sits
// just above the noise floor — full `f32` accuracy, prompt convergence.
const EPS: f32 = 1e-6;

/// Regularized **lower** incomplete gamma `P(a, x) = γ(a, x) / Γ(a)`, `a > 0`, `x ≥ 0`.
pub(crate) fn gamma_lr(a: f32, x: f32) -> f32 {
    if a <= 0.0 || x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 {
        return 0.0;
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
    if a <= 0.0 || x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 {
        return 1.0;
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

/// Single-precision [`ln_rising_factorial`](super::gamma::ln_rising_factorial); see that
/// function for the regime split.
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
            agrees(gamma(x), crate::special::gamma::gamma(x as f64), 1e-5);
            agrees(ln_gamma(x), crate::special::gamma::ln_gamma(x as f64), 1e-5);
        }
    }

    #[test]
    fn incomplete_gamma_matches_f64_reference() {
        for &(a, x) in &[(1.0, 0.5), (2.5, 3.0), (5.0, 2.0), (10.0, 12.0)] {
            let p = gamma_lr(a, x);
            let q = gamma_ur(a, x);
            agrees(p, crate::special::gamma::gamma_lr(a as f64, x as f64), 1e-5);
            agrees(p + q, 1.0, 1e-6);
        }
    }

    #[test]
    fn polygamma_matches_f64_reference() {
        for &x in &[0.5, 1.0, 2.0, 3.3, 7.5] {
            agrees(digamma(x), crate::special::gamma::digamma(x as f64), 1e-5);
            agrees(trigamma(x), crate::special::gamma::trigamma(x as f64), 1e-5);
        }
    }
}
