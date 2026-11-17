//! Combinatorics over structural integer counts: `factorial`, `ln_factorial`, and the
//! binomial coefficient — the host companions to the gamma family
//! (`ln(n!) = ln Γ(n+1)`), used by the discrete-distribution PMFs (Poisson, Binomial, …).
//!
//! These take an integer **count** and return `f64`, so they are not part of the generic
//! [`Real`](crate::Real) surface. A generic density kernel should take any needed
//! `ln_factorial`/`ln_binomial` value from the caller.

use crate::kernels::{f32::gamma::ln_gamma as ln_gamma_f32, gamma::ln_gamma};

/// Cutoff for the exact integer loop (`n < SMALL`). `22!` is in fact the largest factorial
/// exactly representable in `f64`; `20!` is a conservative cutoff that also keeps the
/// `f32` companion loop (`factorial_u32`) honest within its much narrower `2^24` range.
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

/// The exact integer `C(n, k)`, or `None` when the recurrence leaves `u128`.
///
/// The multiplicative recurrence `C(n, i) = C(n, i-1)·(n-k+i)/i`, carried in integers.
/// Every step is exact by construction: the partial `C(n-k+i, i)` is an integer, and `i`
/// divides `acc·(n-k+i)` exactly, so the truncating divide loses nothing.
///
/// Correct for any `k <= n`. The callers still pass `k.min(n - k)`, but that is an
/// *optimisation* and not a precondition: the recurrence telescopes to `C(n, k)` whichever
/// of the two symmetric forms it is given, and in integers both land on the same exact
/// value. Taking the smaller `k` halves the worst-case iteration count and, because the
/// partials stay smaller, keeps more arguments inside `u128` before the fallback is needed.
///
/// That distinction is why `binomial_u128`'s `n - k` is pinned by a test at the very top of
/// the `u128` range rather than by an ordinary value: mutating it to `n + k` changes nothing
/// about the answer and only shows up where the addition itself overflows.
///
/// # Why integers, and not an `f64` accumulator
///
/// Because the exactness bound is set by the *intermediate*, not the answer.
/// `acc·(n-k+i)` equals `i·C(n-k+i, i)` — a factor of `i` above the partial it produces —
/// so an `f64` accumulator begins rounding a factor of roughly `k` below `2^53`, not at it.
/// That is a real and reachable error, not a corner: `C(56, 29)` is
/// `7_384_942_649_010_080`, well inside `2^53`, and the `f64` recurrence returned one less.
/// Twelve of the 1,840 coefficients with `n <= 60` whose true value is exactly
/// representable came out wrong that way, and the docs here claimed exactness through
/// `2^53` throughout. `exactness_holds_wherever_the_answer_is_representable` is the pin.
fn binomial_exact(n: u128, k: u128) -> Option<u128> {
    let mut acc: u128 = 1;
    for i in 1..=k {
        acc = acc.checked_mul(n - k + i)? / i;
    }
    Some(acc)
}

/// The `f64` recurrence, for the arguments whose coefficient does not fit in `u128` at all.
///
/// Inexact in the low digits — unavoidably, since the answer is past `2^53` by a wide
/// margin — and `+∞` once it leaves `f64`, which is the true limit rather than an artefact.
fn binomial_saturating(n: u128, k: u128) -> f64 {
    let mut acc = 1.0_f64;
    for i in 1..=k {
        acc = acc * (n - k + i) as f64 / i as f64;
        if acc.is_infinite() {
            return f64::INFINITY;
        }
    }
    acc.round()
}

/// The binomial coefficient `C(n, k)` as `f64`, over `u128` counts — the widest natural
/// carrier, and the one body the `u64` and `u32` entry points and `Natural for u128` all
/// route through, so every carrier answers a combinatorial question identically.
///
/// **Exact whenever the true coefficient is exactly representable**, via
/// [`binomial_exact`]; the nearest representable `f64` above that, and `+∞` once the
/// coefficient leaves `f64`.
pub(crate) fn binomial_u128(n: u128, k: u128) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    match binomial_exact(n, k) {
        Some(c) => c as f64,
        None => binomial_saturating(n, k),
    }
}

/// The binomial coefficient `C(n, k)` as `f64`. See [`binomial_u128`].
pub(crate) fn binomial(n: u64, k: u64) -> f64 {
    binomial_u128(n as u128, k as u128)
}

/// `ln(n!) = ln Γ(n+1)` over `u128` counts. Exact small sum below 21, Lanczos above —
/// the body [`ln_factorial`] and `Natural for u128` share.
pub(crate) fn ln_factorial_u128(n: u128) -> f64 {
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

/// `n!` as `f64` over `u128` counts. See [`factorial`].
pub(crate) fn factorial_u128(n: u128) -> f64 {
    if (n as usize) < SMALL {
        let mut acc = 1.0_f64;
        for k in 2..=n {
            acc *= k as f64;
        }
        acc
    } else {
        ln_factorial_u128(n).exp()
    }
}

/// `ln C(n, k)` over `u128` counts.
///
/// Differenced from [`ln_factorial_u128`] rather than from `ln_gamma` directly, so that the
/// exact small sum is used wherever it applies and every natural carrier agrees.
pub(crate) fn ln_binomial_u128(n: u128, k: u128) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    ln_factorial_u128(n) - ln_factorial_u128(k) - ln_factorial_u128(n - k)
}

/// `ln C(n, k)`.
pub(crate) fn ln_binomial(n: u64, k: u64) -> f64 {
    ln_binomial_u128(n as u128, k as u128)
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
///
/// The same exact integer recurrence as [`binomial_u128`], rounded once into `f32` — so it
/// is exact wherever the coefficient is representable in `f32` (`2^24`) and the nearest
/// representable value beyond. An `f32` accumulator went wrong from `C(26, 13)` upwards,
/// where `i·C(n-k+i, i)` first clears `2^24` while the coefficient itself is still well
/// inside it.
pub(crate) fn binomial_u32(n: u32, k: u32) -> f32 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    match binomial_exact(n as u128, k as u128) {
        Some(c) => c as f32,
        None => binomial_saturating(n as u128, k as u128) as f32,
    }
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
mod tests;
