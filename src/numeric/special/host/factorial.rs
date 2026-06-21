//! Combinatorics over structural integer counts: `factorial`, `ln_factorial`, and the
//! binomial coefficient — the host companions to the gamma family
//! (`ln(n!) = ln Γ(n+1)`), used by the discrete-distribution PMFs (Poisson, Binomial, …).
//!
//! These take an integer **count** and return `f64`, so they are not part of the lazy
//! [`LazyReal`](crate::numeric::LazyReal) surface — a count is structural, not a tensor
//! value, and a lazy density kernel takes a caller-supplied `ln_factorial`/`ln_binomial`
//! instead.

use crate::numeric::special::{f32::gamma::ln_gamma as ln_gamma_f32, gamma::ln_gamma};

/// Cutoff for the exact loop: 20! is the largest factorial that fits an `f64` exactly.
const SMALL: usize = 21;

/// `n!` as `f64`. Exact through 20!, then via `exp(ln_factorial)` (so it overflows to
/// `+∞` like the true factorial beyond ~170!).
pub(crate) fn factorial(n: u64) -> f64 {
    if (n as usize) < SMALL {
        let mut acc = 1.0_f64;
        for k in 2..=n {
            acc *= k as f64;
        }
        acc
    } else {
        ln_factorial(n).exp()
    }
}

/// `ln(n!) = ln Γ(n+1)`. Exact small sum below 21, Lanczos `ln_gamma` above.
pub(crate) fn ln_factorial(n: u64) -> f64 {
    if (n as usize) < SMALL {
        let mut s = 0.0_f64;
        for k in 2..=n {
            s += (k as f64).ln();
        }
        s
    } else {
        ln_gamma(n as f64 + 1.0)
    }
}

/// The binomial coefficient `C(n, k)` as `f64`.
pub(crate) fn binomial(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }
    ln_binomial(n, k).exp().round()
}

/// `ln C(n, k)`.
pub(crate) fn ln_binomial(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
}

/// `n!` as `f32`. Exact through 20!, then via `exp(ln_factorial_u32)`.
pub(crate) fn factorial_u32(n: u32) -> f32 {
    if (n as usize) < SMALL {
        let mut acc = 1.0_f32;
        for k in 2..=n {
            acc *= k as f32;
        }
        acc
    } else {
        ln_factorial_u32(n).exp()
    }
}

/// `ln(n!) = ln Γ(n+1)`, native to the `u32 -> f32` natural surface.
pub(crate) fn ln_factorial_u32(n: u32) -> f32 {
    if (n as usize) < SMALL {
        let mut s = 0.0_f32;
        for k in 2..=n {
            s += (k as f32).ln();
        }
        s
    } else {
        ln_gamma_f32(n as f32 + 1.0)
    }
}

/// The binomial coefficient `C(n, k)` as `f32`.
pub(crate) fn binomial_u32(n: u32, k: u32) -> f32 {
    if k > n {
        return 0.0;
    }
    ln_binomial_u32(n, k).exp().round()
}

/// `ln C(n, k)` in the `u32 -> f32` natural surface.
pub(crate) fn ln_binomial_u32(n: u32, k: u32) -> f32 {
    if k > n {
        return f32::NEG_INFINITY;
    }
    ln_factorial_u32(n) - ln_factorial_u32(k) - ln_factorial_u32(n - k)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn factorials_exact() {
        assert_eq!(factorial(0), 1.0);
        assert_eq!(factorial(5), 120.0);
        assert_eq!(factorial(10), 3_628_800.0);
    }

    #[test]
    fn ln_factorial_matches() {
        for n in [0u64, 1, 5, 20, 25, 100] {
            let expect = (1..=n).map(|k| (k as f64).ln()).sum::<f64>();
            assert!((ln_factorial(n) - expect).abs() <= 1e-7 * expect.abs().max(1.0));
        }
    }

    #[test]
    fn binomials() {
        assert_eq!(binomial(5, 2), 10.0);
        assert_eq!(binomial(10, 5), 252.0);
        assert_eq!(binomial(3, 4), 0.0);
    }

    #[test]
    fn u32_factorials_are_native_f32() {
        assert_eq!(factorial_u32(0), 1.0);
        assert_eq!(factorial_u32(5), 120.0);
        assert_eq!(factorial_u32(10), 3_628_800.0);
        assert!((ln_factorial_u32(25) - ln_factorial(25) as f32).abs() < 1e-4);
    }

    #[test]
    fn u32_binomials_are_native_f32() {
        assert_eq!(binomial_u32(5, 2), 10.0);
        assert_eq!(binomial_u32(10, 5), 252.0);
        assert_eq!(binomial_u32(3, 4), 0.0);
    }
}
