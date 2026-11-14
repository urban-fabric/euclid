//! The gamma function family: `ln_gamma`/`gamma` (Lanczos), the regularized lower/upper
//! incomplete gamma `gamma_lr`/`gamma_ur` (the Poisson/Gamma/chi-squared CDF kernels), and
//! the polygamma `digamma`/`trigamma`.

use crate::constants::f64::SQRT_TAU;
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
    if x < 0.5 {
        PI / ((PI * x).sin() * gamma(1.0 - x))
    } else if x > GAMMA_OVERFLOW_MIN {
        f64::INFINITY
    } else {
        let x = x - 1.0;
        let mut a = LANCZOS[0];
        let t = x + G + 0.5;
        for (i, &c) in LANCZOS.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        // `t^(x+0.5)·e^{−t}` is formed as the square of its own square root, which looks
        // gratuitous and is not. Written directly, `t^(x+0.5)` overflows *long* before the
        // product does, because `e^{−t}` is a large compensating underflow — so the
        // intermediate leaves the format's range while the answer is still comfortably
        // inside it. Measured against exact factorials, the direct form returned:
        //
        //   `gamma(144)` through `gamma(171)` as `inf`, where the true values
        //   `3.9e247` through `7.3e306` are all representable
        //
        // and once `e^{−t}` underflowed to zero as well, `inf · 0 = NaN` — for `x ≥ 739`,
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
const EPS: f64 = 1e-15;

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

/// `(P, Q)` from the Edgeworth-corrected normal approximation; see [`GAMMA_EDGEWORTH_MIN`].
///
/// `Q` is formed as `Φ(−z)` rather than `1 − Φ(z)`, so a small upper tail keeps its relative
/// accuracy instead of being lost to the complement.
///
/// The `erfc` call here reaches back into [`super::erf`], which is itself built on
/// `gamma_lr` — but the recursion terminates immediately: `erfc` asks for shape `½`, far below
/// this threshold, so it takes the series branch and never returns here.
fn gamma_edgeworth(a: f64, x: f64) -> (f64, f64) {
    let root = a.sqrt();
    let z = (x - a) / root;
    let density = (-0.5 * z * z).exp() / SQRT_TAU;
    let correction = density * (z * z - 1.0) / (3.0 * root);
    let lower = 0.5 * super::erf::erfc(-z * core::f64::consts::FRAC_1_SQRT_2) - correction;
    let upper = 0.5 * super::erf::erfc(z * core::f64::consts::FRAC_1_SQRT_2) + correction;
    (lower.clamp(0.0, 1.0), upper.clamp(0.0, 1.0))
}

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
    if a >= GAMMA_EDGEWORTH_MIN {
        return gamma_edgeworth(a, x).1;
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
/// recurrence steps. Those steps are one division and one add each, at most ten of them.
const PSI_ASYMPTOTIC_MIN: f64 = 20.0;

/// The digamma function `ψ(x) = d/dx ln Γ(x)`.
pub(crate) fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < PSI_ASYMPTOTIC_MIN {
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
    while x < PSI_ASYMPTOTIC_MIN {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let r = 1.0 / x;
    let r2 = r * r;
    result + r + 0.5 * r2 + r2 * r * (1.0 / 6.0 - r2 * (1.0 / 30.0 - r2 * (1.0 / 42.0 - r2 / 30.0)))
}
/// Lower edge of the differenced-Stirling route in [`ln_rising_factorial`].
///
/// Below it the direct `lnΓ(a+n) − lnΓ(a)` is used, where the cancellation costs about
/// `ε·a·ln a` — `1.9e-13` absolute at the threshold, and shrinking from there. Above it the
/// differenced series is exact to rounding.
///
/// It was `1e4`, with a single correction pair. That left `a = 1e4` itself on the *direct*
/// side, where the two log-gammas are both `82108` and their difference is `4.6`: `1.8e-11` of
/// absolute error, which showed up as `4.7e-13` in `ln B(½, 1e4)`. Three correction pairs
/// bring the threshold down to where the direct route is still harmless.
const STIRLING_RATIO_MIN: f64 = 300.0;

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
    if a > STIRLING_RATIO_MIN && a + n > STIRLING_RATIO_MIN {
        let r = n / a;
        let l1p = r.ln_1p();
        // Stirling's series differenced term by term. Each correction appears as a *pair*,
        // `c(a+n) − c(a)`, so the large parts never meet: what is formed is only the small
        // difference. Carrying three pairs rather than one is what lets the threshold come
        // down to 300 — the remainder is then `~1/(1260a⁵)`, which is `3e-16` there.
        let correction = |z: f64| {
            let inv = 1.0 / z;
            let inv2 = inv * inv;
            inv * (1.0 / 12.0 + inv2 * (-1.0 / 360.0 + inv2 / 1260.0))
        };
        return n * a.ln() + (a + n) * l1p - n - 0.5 * l1p + correction(a + n) - correction(a);
    }
    // (3) the terms are not nearly equal; the direct difference is fine.
    ln_gamma(a + n) - ln_gamma(a)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests;
