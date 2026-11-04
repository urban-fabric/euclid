//! The polygamma function `ψ⁽ⁿ⁾(x)` for arbitrary order, and the reciprocal gamma `1/Γ(x)`.
//!
//! # Polygamma by Euler–Maclaurin
//!
//! For `n ≥ 1`, `ψ⁽ⁿ⁾(x) = (−1)^{n+1} n! ζ(n+1, x)`, where `ζ(s, a) = Σ_{k≥0} (a+k)^{−s}` is
//! the Hurwitz zeta. That series converges like `k^{−(n+1)}`, which is far too slow to use
//! directly: reaching double precision for `n = 1` would take about `1e8` terms.
//!
//! Euler–Maclaurin fixes that by summing a handful of terms exactly and replacing the rest
//! with an integral plus a correction series in Bernoulli numbers:
//!
//! ```text
//! ζ(s, a) ≈ Σ_{k<N} (a+k)^{−s}
//!         + (a+N)^{1−s}/(s−1)
//!         + ½ (a+N)^{−s}
//!         + Σ_{j≥1} c_j · (s)_{2j−1} · (a+N)^{−s−2j+1}
//! ```
//!
//! with `c_j = B_{2j}/(2j)!` and `(s)_m` the rising factorial. With `N = 16` and five
//! correction terms this reaches full double precision for every order used in practice, in a
//! fixed amount of work rather than `O(1e8)`.
//!
//! The Bernoulli coefficients are written as exact rationals rather than decimal digits, so
//! there is nothing to transcribe wrongly.

use super::gamma::{digamma, ln_gamma, trigamma};

/// Terms summed exactly before the Euler–Maclaurin tail takes over.
const EM_SPLIT: u32 = 16;

/// `B_{2j} / (2j)!` for `j = 1..=6`, as exact rationals.
///
/// `B₂ = 1/6`, `B₄ = −1/30`, `B₆ = 1/42`, `B₈ = −1/30`, `B₁₀ = 5/66`, `B₁₂ = −691/2730`.
const EM_COEFFS: [f64; 6] = [
    1.0 / 12.0,                      // (1/6)/2!
    -1.0 / 720.0,                    // (-1/30)/4!
    1.0 / 30_240.0,                  // (1/42)/6!
    -1.0 / 1_209_600.0,              // (-1/30)/8!
    1.0 / 47_900_160.0,              // (5/66)/10!
    -691.0 / 2730.0 / 479_001_600.0, // (-691/2730)/12!
];

/// The Hurwitz zeta `ζ(s, a)` for real `s > 1` and `a > 0`, by Euler–Maclaurin.
///
/// `pub(crate)` because the Riemann zeta is `ζ(s, 1)` and reuses this directly rather than
/// reimplementing the same summation.
pub(crate) fn hurwitz_zeta(s: f64, a: f64) -> f64 {
    debug_assert!(s > 1.0);
    debug_assert!(a > 0.0);

    // Exact head.
    let mut total = 0.0f64;
    for k in 0..EM_SPLIT {
        total += (a + f64::from(k)).powf(-s);
    }

    let b = a + f64::from(EM_SPLIT);
    // Integral term and the half-endpoint.
    total += b.powf(1.0 - s) / (s - 1.0);
    total += 0.5 * b.powf(-s);

    // Bernoulli corrections. The j-th needs the rising factorial (s)_{2j-1}, built
    // incrementally so no factorial is ever formed.
    let mut rising = s; // (s)_1
    let mut power = b.powf(-s - 1.0); // b^{-s-2j+1} at j = 1
    for (j, &c) in EM_COEFFS.iter().enumerate() {
        total += c * rising * power;
        // Advance to j+1: (s)_{2j+1} = (s)_{2j-1} · (s+2j-1)(s+2j)
        let m = 2.0 * (j as f64 + 1.0);
        rising *= (s + m - 1.0) * (s + m);
        power /= b * b;
    }
    total
}

/// `ψ⁽ⁿ⁾(x)`, the `n`-th derivative of `digamma`.
pub(crate) fn polygamma(order: u32, x: f64) -> f64 {
    match order {
        0 => return digamma(x),
        1 => return trigamma(x),
        _ => {}
    }
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 && x.fract() == 0.0 {
        return f64::NAN; // poles at the non-positive integers
    }
    if x < 0.0 {
        // No reflection formula is used here; the function is only defined for x > 0 in this
        // implementation, which is where every caller needs it.
        return f64::NAN;
    }
    let n = f64::from(order);
    let s = n + 1.0;
    // (-1)^{n+1} n! zeta(n+1, x)
    let sign = if order.is_multiple_of(2) { -1.0 } else { 1.0 };
    sign * (ln_gamma(n + 1.0)).exp() * hurwitz_zeta(s, x)
}

/// The reciprocal gamma `1/Γ(x)`, entire and zero at the non-positive integers.
pub(crate) fn rgamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        // 1/Gamma -> 0 at +inf; the negative direction oscillates without limit.
        return if x > 0.0 { 0.0 } else { f64::NAN };
    }
    if x <= 0.0 && x.fract() == 0.0 {
        return 0.0; // the poles of Gamma are the zeros of 1/Gamma
    }
    if x > 0.0 {
        // exp(-lnGamma) rather than gamma(x).recip(): Gamma overflows from about x = 172,
        // while 1/Gamma stays representable well past that.
        return (-ln_gamma(x)).exp();
    }
    // Reflection: 1/Gamma(x) = Gamma(1-x) sin(pi x) / pi.
    let s = (core::f64::consts::PI * x).sin();
    let magnitude = (ln_gamma(1.0 - x) + s.abs().ln() - core::f64::consts::PI.ln()).exp();
    if s < 0.0 { -magnitude } else { magnitude }
}

/// `(eˣ − 1)/x`, continuous at the origin.
pub(crate) fn exprel(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { f64::INFINITY } else { 0.0 };
    }
    // The series is better conditioned than expm1(x)/x for tiny x and avoids the 0/0 form.
    if x.abs() < 1e-5 {
        return 1.0 + x / 2.0 + x * x / 6.0 + x * x * x / 24.0;
    }
    x.exp_m1() / x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol * b.abs().max(1.0),
            "{a:e} vs {b:e} (tol {tol:e})"
        );
    }

    #[test]
    fn polygamma_reduces_to_digamma_and_trigamma() {
        for i in 1..40 {
            let x = f64::from(i) * 0.25;
            close(polygamma(0, x), digamma(x), 1e-14);
            close(polygamma(1, x), trigamma(x), 1e-14);
        }
    }

    #[test]
    fn polygamma_matches_its_known_values_at_one() {
        // psi^(n)(1) = (-1)^{n+1} n! zeta(n+1)
        // zeta(3) = 1.2020569031595942854, zeta(4) = pi^4/90, zeta(5) = 1.0369277551433699263
        let pi4_90 = core::f64::consts::PI.powi(4) / 90.0;
        close(polygamma(2, 1.0), -2.0 * 1.202_056_903_159_594_3, 1e-12);
        close(polygamma(3, 1.0), 6.0 * pi4_90, 1e-12);
        // zeta(5), written to the digits f64 can hold.
        close(polygamma(4, 1.0), -24.0 * 1.036_927_755_143_37, 1e-12);
    }

    #[test]
    fn polygamma_satisfies_its_recurrence() {
        // psi^(n)(x+1) = psi^(n)(x) + (-1)^n n! / x^{n+1}
        for order in 2..7u32 {
            let n = f64::from(order);
            let factorial = (1..=order).map(f64::from).product::<f64>();
            let sign = if order.is_multiple_of(2) { 1.0 } else { -1.0 };
            for i in 1..30 {
                let x = f64::from(i) * 0.5;
                let lhs = polygamma(order, x + 1.0);
                let rhs = polygamma(order, x) + sign * factorial / x.powf(n + 1.0);
                close(lhs, rhs, 1e-10);
            }
        }
    }

    #[test]
    fn polygamma_is_the_derivative_of_the_previous_order() {
        for order in 1..5u32 {
            for i in 2..20 {
                let x = f64::from(i) * 0.5;
                let h = 1e-5 * x;
                let d = (polygamma(order, x + h) - polygamma(order, x - h)) / (2.0 * h);
                close(d, polygamma(order + 1, x), 1e-7);
            }
        }
    }

    #[test]
    fn rgamma_is_zero_at_the_poles_and_inverts_gamma_elsewhere() {
        for n in 0..10 {
            assert_eq!(rgamma(-f64::from(n)), 0.0, "1/Gamma(-{n}) = 0");
        }
        for i in 1..60 {
            let x = f64::from(i) * 0.25;
            close(rgamma(x), (-ln_gamma(x)).exp(), 1e-14);
        }
        // Negative, non-integer arguments through the reflection formula.
        for &x in &[-0.5f64, -1.5, -2.5, -3.5, -0.25, -4.75] {
            let g = super::super::gamma::gamma(x);
            close(rgamma(x), 1.0 / g, 1e-11);
        }
    }

    #[test]
    fn rgamma_is_defined_where_gamma_overflows() {
        // Gamma(175) overflows; 1/Gamma(175) is a representable subnormal-range value.
        assert!(super::super::gamma::gamma(175.0).is_infinite());
        let r = rgamma(175.0);
        assert!(r > 0.0 && r.is_finite(), "rgamma(175) = {r}");
        // 1/Gamma(175) is about 1e-316, which is *subnormal* — below f64::MIN_POSITIVE at
        // 2.2e-308 — so it carries well under 53 bits of mantissa. The value is recoverable,
        // but a log round trip through it cannot be checked at full precision.
        assert!(
            r < f64::MIN_POSITIVE,
            "expected a subnormal result, got {r}"
        );
        close(r.ln(), -ln_gamma(175.0), 1e-10);

        // Where the result is normal, the round trip is tight.
        let r160 = rgamma(160.0);
        assert!(r160 > f64::MIN_POSITIVE);
        close(r160.ln(), -ln_gamma(160.0), 1e-14);
    }

    #[test]
    fn exprel_is_continuous_at_the_origin() {
        assert_eq!(exprel(0.0), 1.0);
        // `1 + x/2` is leading-order only; the x^2/6 term is 1.7e-3 at x = 0.1, so compare
        // against it only where it is negligible.
        for i in 6..40 {
            let x = 10f64.powi(-i);
            close(exprel(x), 1.0 + x / 2.0, 1e-12);
        }
        for i in 1..40 {
            let x = f64::from(i) * 0.25;
            close(exprel(x), x.exp_m1() / x, 1e-14);
            close(exprel(-x), (-x).exp_m1() / -x, 1e-14);
        }
    }
}
