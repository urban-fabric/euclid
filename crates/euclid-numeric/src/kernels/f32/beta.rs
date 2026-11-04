//! The `f32` beta family. `ln_beta` and `beta` are native single precision — they are
//! well-conditioned at every magnitude, having been rewritten around the rising factorial to
//! remove the subtraction that used to wreck them. `beta_reg` is **not**: it narrows the `f64`
//! implementation, for the reason set out on that function.

use super::gamma::{ln_gamma, ln_rising_factorial};

/// `ln B(a, b) = ln Γ(a) + ln Γ(b) − ln Γ(a + b)`.
///
/// # Why it is not written that way
///
/// Written literally, that expression subtracts two nearly-equal large logarithms whenever
/// `a + b ≫ min(a, b)` — and the result is `O(ln)` while the terms are `O(a ln a)`. The damage
/// is total, not gradual. Against the exact `ln B(a, 1) = −ln a`:
///
/// ```text
/// a = 1e2   1.7e-14 relative        a = 1e15   4e-2 relative
/// a = 1e8   1.4e-9  relative        a = 1e16   ln B returned 0, and B returned 1
/// ```
///
/// Past `a = 1/ε` the sum `a + b` rounds back to `a`, so `lnΓ(a + b)` and `lnΓ(a)` become
/// *bit-identical* and the whole answer cancels away. `B(1e300, 1)` came back as `1` where the
/// true value is `1e-300`.
///
/// Rewriting it around the rising factorial removes the subtraction entirely:
///
/// ```text
/// ln B(a, b) = ln Γ(b) − ln (a)_b,     (a)_b = Γ(a+b)/Γ(a)
/// ```
///
/// which is the same identity read differently — and
/// [`ln_rising_factorial`] already computes
/// that ratio without ever forming the two logs apart, by an exact product for small integer
/// `b` and an integrated-digamma asymptotic for large `a`. The larger argument is passed as
/// the base so the increment is the short one.
pub(crate) fn ln_beta(a: f32, b: f32) -> f32 {
    let (base, increment) = if a >= b { (a, b) } else { (b, a) };
    ln_gamma(increment) - ln_rising_factorial(base, increment)
}

/// The beta function `B(a, b)`.
pub(crate) fn beta(a: f32, b: f32) -> f32 {
    ln_beta(a, b).exp()
}

/// The regularized incomplete beta `Iₓ(a, b)`, `x ∈ [0, 1]`, `a, b > 0`.
///
/// **Evaluated in `f64` and narrowed once**, unlike `ln_beta` and `beta` above, which stay
/// native. This is the exception the [module header](super) describes, and `beta_reg` is the
/// clearest case of it in the crate: its error floor is set by the *working precision*, so a
/// single-precision port is not merely less accurate but unusable.
///
/// The reason is the `exp` prefactor, `exp(a ln x + b ln(1−x) − ln B(a,b))`. Those three terms
/// each have magnitude `≈ 1.386(a+b)` and sum to an `O(1)` exponent, so the exponent carries
/// `≈ 1.386(a+b)·ε` of absolute error and `exp` returns it as *relative* error. In `f64` that is
/// tolerable to `a + b ≈ 1e7`; in `f32`, with `ε = 1.2e-7`, it is `1.65e-7(a+b)` — already
/// `3e-4` at `a = b = 1e3`. The native port measured, against the exact `I_{1/2}(a,a) = ½`:
///
/// | `a = b` | 1e4 | 1e5 | 1e6 | 1e7 | 1e8 | 1e10 |
/// |---|---|---|---|---|---|---|
/// | native `f32` error | `3.0e-3` | `3.7e-2` | `0.319` | `0.4995` | `6.9e23` | `inf` |
///
/// The last two columns are not probabilities at all — it returned a large negative number and
/// then `−inf`. Narrowing replaces the whole table with a single rounding of an `f64` value
/// that is itself good to `≈1e-9` or better everywhere, so the `f32` surface inherits the
/// Edgeworth handover and the convergence guard rather than needing its own.
pub(crate) fn beta_reg(a: f32, b: f32, x: f32) -> f32 {
    crate::kernels::beta::beta_reg(f64::from(a), f64::from(b), f64::from(x)) as f32
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn agrees(a: f32, b: f64, tol: f32) {
        let b = b as f32;
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn beta_matches_f64_reference() {
        for &(a, b) in &[(1.0, 1.0), (2.0, 3.0), (0.5, 0.5), (5.0, 2.0)] {
            agrees(
                beta(a, b),
                crate::kernels::beta::beta(a as f64, b as f64),
                1e-5,
            );
        }
    }

    #[test]
    fn beta_reg_matches_f64_reference() {
        assert_eq!(beta_reg(2.0, 3.0, 0.0), 0.0);
        assert_eq!(beta_reg(2.0, 3.0, 1.0), 1.0);
        for &(a, b, x) in &[(2.0, 3.0, 0.4), (0.5, 0.5, 0.3), (5.0, 2.0, 0.7)] {
            agrees(
                beta_reg(a, b, x),
                crate::kernels::beta::beta_reg(a as f64, b as f64, x as f64),
                1e-5,
            );
        }
    }
}
