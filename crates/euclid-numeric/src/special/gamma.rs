//! The gamma function family: `ln_gamma`/`gamma` (Lanczos), the regularized lower/upper
//! incomplete gamma `gamma_lr`/`gamma_ur` (the Poisson/Gamma/chi-squared CDF kernels), and
//! the polygamma `digamma`/`trigamma`.

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

/// Natural log of the gamma function, `ln Γ(x)`, for `x > 0` (and, via reflection,
/// `x < 0` away from the non-positive integers).
pub(crate) fn ln_gamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // ln Gamma grows without bound, so the limit at +inf is +inf rather than NaN.
    if x.is_infinite() {
        return if x > 0.0 { f64::INFINITY } else { f64::NAN };
    }
    if x < 0.5 {
        (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = LANCZOS[0];
        let t = x + G + 0.5;
        for (i, &c) in LANCZOS.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// The gamma function `Γ(x)`.
pub(crate) fn gamma(x: f64) -> f64 {
    if x < 0.5 {
        PI / ((PI * x).sin() * gamma(1.0 - x))
    } else {
        let x = x - 1.0;
        let mut a = LANCZOS[0];
        let t = x + G + 0.5;
        for (i, &c) in LANCZOS.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        (2.0 * PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a
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

/// Regularized **lower** incomplete gamma `P(a, x) = γ(a, x) / Γ(a)`, `a > 0`, `x ≥ 0`.
pub(crate) fn gamma_lr(a: f64, x: f64) -> f64 {
    if a <= 0.0 || x < 0.0 || a.is_nan() || x.is_nan() {
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
pub(crate) fn gamma_ur(a: f64, x: f64) -> f64 {
    if a <= 0.0 || x < 0.0 || a.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    // Q(a, x) -> 0 as x -> inf, the complement of `gamma_lr`'s limit.
    if x.is_infinite() {
        return 0.0;
    }
    if x < a + 1.0 {
        return 1.0 - gamma_lr(a, x);
    }
    let tiny = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..MAX_ITER {
        let an = -(i as f64) * (i as f64 - a);
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
pub(crate) fn digamma(mut x: f64) -> f64 {
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
pub(crate) fn trigamma(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() {
        return f64::INFINITY;
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

/// `ln Γ(a + n) − ln Γ(a)`, the log rising factorial, computed without the cancellation of
/// subtracting two nearly-equal `ln_gamma` values.
///
/// Three regimes:
///
/// 1. **`n` a small non-negative integer** — sum `ln(a+k)` directly. All terms are the same
///    sign, so there is no cancellation at any `a`, and this is the dominant use (binomial and
///    beta-binomial weights, Dirichlet updates).
/// 2. **large `a`** — integrate the digamma asymptotic, `lnΓ(a+n) − lnΓ(a) = ∫ₐ^{a+n} ψ(t)dt`
///    with `ψ(t) ≈ ln t − 1/(2t) − 1/(12t²)`. Rearranged so every term is formed with `ln_1p`
///    on `n/a` rather than by subtracting two large logarithms:
///    `n·ln a + (a+n)·ln1p(n/a) − n − ½·ln1p(n/a) + 1/(12(a+n)) − 1/(12a)`.
/// 3. **otherwise** — the direct difference, which is accurate when the two terms are not
///    nearly equal.
pub(crate) fn ln_rising_factorial(a: f64, n: f64) -> f64 {
    if n == 0.0 {
        return 0.0;
    }
    if !a.is_finite() || !n.is_finite() {
        return ln_gamma(a + n) - ln_gamma(a);
    }
    // (1) small integer n: an exact product in the log domain.
    if n > 0.0 && n <= 64.0 && n.fract() == 0.0 {
        let mut total = 0.0f64;
        let mut k = 0.0f64;
        while k < n {
            total += (a + k).ln();
            k += 1.0;
        }
        return total;
    }
    // (2) large a: integrated digamma asymptotic, cancellation-free in n/a.
    if a > 1e4 && a + n > 1e4 {
        let r = n / a;
        let l1p = r.ln_1p();
        return n * a.ln() + (a + n) * l1p - n - 0.5 * l1p + 1.0 / (12.0 * (a + n))
            - 1.0 / (12.0 * a);
    }
    // (3) the terms are not nearly equal; the direct difference is fine.
    ln_gamma(a + n) - ln_gamma(a)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn gamma_matches_factorials() {
        close(gamma(1.0), 1.0, 1e-12);
        close(gamma(5.0), 24.0, 1e-10);
        close(gamma(6.0), 120.0, 1e-10);
        close(gamma(0.5), core::f64::consts::PI.sqrt(), 1e-12);
    }

    #[test]
    fn ln_gamma_matches_gamma() {
        for &x in &[0.3, 0.7, 1.5, 3.2, 10.0, 50.0] {
            close(ln_gamma(x).exp(), gamma(x), 1e-9);
        }
    }

    #[test]
    fn regularized_incomplete_gamma_bounds_and_complement() {
        for &(a, x) in &[(1.0, 0.5), (2.5, 3.0), (5.0, 2.0), (10.0, 12.0)] {
            let p = gamma_lr(a, x);
            let q = gamma_ur(a, x);
            assert!((0.0..=1.0).contains(&p), "P out of range: {p}");
            close(p + q, 1.0, 1e-10);
        }
        close(gamma_lr(1.0, 2.0), 1.0 - (-2.0f64).exp(), 1e-10);
    }

    #[test]
    fn digamma_known_values() {
        // Tightened to 1e-11 after adding the 1/(240·x⁸) asymptotic term: the prior
        // truncation reached only ~1e-10 (vs the ~15-digit intent of the coefficients).
        close(digamma(1.0), -0.577_215_664_901_532_9, 1e-11);
        close(digamma(2.0), 1.0 - 0.577_215_664_901_532_9, 1e-11);
        // ψ(3) = 3/2 − γ.
        close(digamma(3.0), 1.5 - 0.577_215_664_901_532_9, 1e-11);
    }

    #[test]
    fn trigamma_known_values() {
        let pi2 = core::f64::consts::PI * core::f64::consts::PI;
        close(trigamma(1.0), pi2 / 6.0, 1e-10);
        close(trigamma(2.0), pi2 / 6.0 - 1.0, 1e-10);
        close(trigamma(0.5), pi2 / 2.0, 1e-10);
        close(trigamma(3.3) - trigamma(4.3), 1.0 / (3.3 * 3.3), 1e-10);
    }
}
