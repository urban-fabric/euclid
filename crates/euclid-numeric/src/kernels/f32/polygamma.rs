//! Single-precision peer of [`super::super::polygamma`]. Same Euler–Maclaurin scheme with a
//! shorter exact head, since `f32` needs far fewer terms to reach its own noise floor.

use super::gamma::{digamma, ln_gamma, trigamma};

/// Terms summed exactly before the Euler–Maclaurin tail takes over.
const EM_SPLIT: u32 = 12;

/// `B_{2j} / (2j)!` for `j = 1..=6`, as exact rationals.
///
/// `B₂ = 1/6`, `B₄ = −1/30`, `B₆ = 1/42`, `B₈ = −1/30`, `B₁₀ = 5/66`, `B₁₂ = −691/2730`.
const EM_COEFFS: [f32; 6] = [
    1.0 / 12.0,                      // (1/6)/2!
    -1.0 / 720.0,                    // (-1/30)/4!
    1.0 / 30_240.0,                  // (1/42)/6!
    -1.0 / 1_209_600.0,              // (-1/30)/8!
    1.0 / 47_900_160.0,              // (5/66)/10!
    -691.0 / 2730.0 / 479_001_600.0, // (-691/2730)/12!
];

/// The Hurwitz zeta `ζ(s, a)` for integer `s ≥ 2` and `a > 0`, by Euler–Maclaurin.
fn hurwitz_zeta(s: f32, a: f32) -> f32 {
    debug_assert!(s >= 2.0);
    debug_assert!(a > 0.0);

    // Exact head.
    let mut total = 0.0f32;
    for k in 0..EM_SPLIT {
        total += (a + (k as f32)).powf(-s);
    }

    let b = a + (EM_SPLIT as f32);
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
        let m = 2.0 * (j as f32 + 1.0);
        rising *= (s + m - 1.0) * (s + m);
        power /= b * b;
    }
    total
}

/// `ψ⁽ⁿ⁾(x)`, the `n`-th derivative of `digamma`.
pub(crate) fn polygamma(order: u32, x: f32) -> f32 {
    match order {
        0 => return digamma(x),
        1 => return trigamma(x),
        _ => {}
    }
    if x.is_nan() {
        return f32::NAN;
    }
    if x <= 0.0 && x.fract() == 0.0 {
        return f32::NAN; // poles at the non-positive integers
    }
    if x < 0.0 {
        // No reflection formula is used here; the function is only defined for x > 0 in this
        // implementation, which is where every caller needs it.
        return f32::NAN;
    }
    let n = order as f32;
    let s = n + 1.0;
    // (-1)^{n+1} n! zeta(n+1, x)
    let sign = if order.is_multiple_of(2) { -1.0 } else { 1.0 };
    sign * (ln_gamma(n + 1.0)).exp() * hurwitz_zeta(s, x)
}

/// The reciprocal gamma `1/Γ(x)`, entire and zero at the non-positive integers.
pub(crate) fn rgamma(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x.is_infinite() {
        // 1/Gamma -> 0 at +inf; the negative direction oscillates without limit.
        return if x > 0.0 { 0.0 } else { f32::NAN };
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
    let s = (core::f32::consts::PI * x).sin();
    let magnitude = (ln_gamma(1.0 - x) + s.abs().ln() - core::f32::consts::PI.ln()).exp();
    if s < 0.0 { -magnitude } else { magnitude }
}

/// `(eˣ − 1)/x`, continuous at the origin.
pub(crate) fn exprel(x: f32) -> f32 {
    if x == 0.0 {
        return 1.0;
    }
    if x.is_nan() {
        return f32::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { f32::INFINITY } else { 0.0 };
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

    fn close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol * b.abs().max(1.0),
            "{a:e} vs {b:e} (tol {tol:e})"
        );
    }

    #[test]
    fn polygamma_reduces_and_recurs_in_f32() {
        for i in 1..20 {
            let x = (i as f32) * 0.5;
            close(polygamma(0, x), digamma(x), 1e-5);
            close(polygamma(1, x), trigamma(x), 1e-5);
        }
        // psi^(2)(x+1) = psi^(2)(x) + 2/x^3
        for i in 1..20 {
            let x = (i as f32) * 0.5;
            close(
                polygamma(2, x + 1.0),
                polygamma(2, x) + 2.0 / x.powi(3),
                1e-4,
            );
        }
    }

    #[test]
    fn rgamma_and_exprel_in_f32() {
        assert_eq!(rgamma(0.0f32), 0.0);
        assert_eq!(rgamma(-3.0f32), 0.0);
        for i in 1..20 {
            let x = (i as f32) * 0.5;
            close(rgamma(x), (-ln_gamma(x)).exp(), 1e-5);
        }
        assert_eq!(exprel(0.0f32), 1.0);
        for i in 1..20 {
            let x = (i as f32) * 0.25;
            close(exprel(x), x.exp_m1() / x, 1e-5);
        }
    }
}
