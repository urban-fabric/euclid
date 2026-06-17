//! The numeric **field** traits: a single source of truth for *what arithmetic a
//! formula needs*, so each formula is written **once** and runs on every numeric
//! surface — scalar `f64`/`f32`, complex scalars, and differentiable tensors.
//!
//! See `plans/sci-comp/supercomputer.md` §3. A formula declares the minimum it
//! needs: an inherently-real density or geometry kernel takes `T: RealField`; a
//! linear-algebra / FFT / complex-network kernel takes `T: ComplexField`. The
//! same real formula still runs on the real component of complex data.
//!
//! Unlike euclid's [`Quantity`](crate::Quantity) traits, these are **not**
//! `Copy`-bound. A lazy tensor handle (`lebesgue::Tensor`) is `Clone` but not
//! `Copy`, so the operators are taken by value over cheap-to-clone operands.
//!
//! euclid provides the `f64`/`f32` implementations (the scalar reference). The
//! complex implementations live with their owning crate: `ComplexField` for
//! `jacobi::Complex{32,64}` in `jacobi`, and `Field` for `lebesgue::Tensor`
//! (real *and* complex dtype) in `lebesgue` — each by the orphan rule, since the
//! concrete type is local there.

use core::ops::{Add, Div, Mul, Neg, Sub};

/// The arithmetic substrate: `Clone` plus the four field operations and
/// negation, with a constructor from a real literal. The `Output` of every
/// operator is `Self`, and operands are taken by value (cheap for a tensor
/// handle). This is the bound behind `+ - * /` in a generic formula.
pub trait Scalar:
    Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Lift a real constant into the field (`f64` identity; `x + 0i` for complex;
    /// a scalar tensor for `Tensor`). The single entry point for literals so a
    /// generic formula never names a concrete numeric type.
    fn from_f64(x: f64) -> Self;

    /// The additive identity.
    fn zero() -> Self {
        Self::from_f64(0.0)
    }

    /// The multiplicative identity.
    fn one() -> Self {
        Self::from_f64(1.0)
    }
}

/// A real field: a [`Scalar`] plus the real transcendentals that inherently-real
/// formulas (probability densities, geometry kernels) are written in.
pub trait RealField: Scalar {
    /// `e^x`.
    fn exp(&self) -> Self;
    /// Natural logarithm.
    fn ln(&self) -> Self;
    /// Square root.
    fn sqrt(&self) -> Self;
    /// Absolute value.
    fn abs(&self) -> Self;
    /// Raise to a real power.
    fn powf(&self, p: f64) -> Self;
    /// Numerically-stable `ln(1 + e^x)` (softplus / `log1pexp`) — the canonical
    /// stable primitive for logistic losses and log-sum-exp.
    fn ln_1p_exp(&self) -> Self;
}

/// A complex field: a [`Scalar`] plus complex structure, parameterized by its
/// real-component field. `re`/`im`/`norm_sqr` project to `Self::Real`;
/// `from_parts` assembles a value from real and imaginary components.
pub trait ComplexField: Scalar {
    /// The real-component field — `f64` for `Complex64`, and `Self` for a
    /// dtype-erased `Tensor` (whose real part is still a `Tensor`).
    type Real: RealField;

    /// Complex conjugate (identity on a real value).
    fn conj(&self) -> Self;
    /// Real part.
    fn re(&self) -> Self::Real;
    /// Imaginary part (zero for a real value).
    fn im(&self) -> Self::Real;
    /// Squared magnitude `re² + im²`.
    fn norm_sqr(&self) -> Self::Real;
    /// Assemble from real and imaginary components.
    fn from_parts(re: Self::Real, im: Self::Real) -> Self;
}

// --- f64 / f32: the scalar reference implementations ----------------------
//
// Stable softplus `ln_1p_exp`: for x > 0 factor out x to avoid overflow in e^x;
// for x <= 0 evaluate ln(1 + e^x) directly.

impl Scalar for f64 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
}

impl RealField for f64 {
    #[inline]
    fn exp(&self) -> Self {
        f64::exp(*self)
    }
    #[inline]
    fn ln(&self) -> Self {
        f64::ln(*self)
    }
    #[inline]
    fn sqrt(&self) -> Self {
        f64::sqrt(*self)
    }
    #[inline]
    fn abs(&self) -> Self {
        f64::abs(*self)
    }
    #[inline]
    fn powf(&self, p: f64) -> Self {
        f64::powf(*self, p)
    }
    #[inline]
    fn ln_1p_exp(&self) -> Self {
        let x = *self;
        if x > 0.0 {
            x + f64::ln_1p((-x).exp())
        } else {
            f64::ln_1p(x.exp())
        }
    }
}

impl Scalar for f32 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
}

impl RealField for f32 {
    #[inline]
    fn exp(&self) -> Self {
        f32::exp(*self)
    }
    #[inline]
    fn ln(&self) -> Self {
        f32::ln(*self)
    }
    #[inline]
    fn sqrt(&self) -> Self {
        f32::sqrt(*self)
    }
    #[inline]
    fn abs(&self) -> Self {
        f32::abs(*self)
    }
    #[inline]
    fn powf(&self, p: f64) -> Self {
        f32::powf(*self, p as f32)
    }
    #[inline]
    fn ln_1p_exp(&self) -> Self {
        let x = *self;
        if x > 0.0 {
            x + f32::ln_1p((-x).exp())
        } else {
            f32::ln_1p(x.exp())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A formula written once, generic over the real field.
    fn quadratic<T: RealField>(x: T) -> T {
        x.clone() * x.clone() + x + T::one()
    }

    /// `-½·((x-µ)/σ)² − ln σ − ½·ln(2π)` — the Gaussian log-density, generic.
    fn normal_log_prob<T: RealField>(x: T, mu: T, sigma: T) -> T {
        let z = (x - mu) / sigma.clone();
        let half = T::from_f64(0.5);
        let log_2pi = T::from_f64((2.0 * core::f64::consts::PI).ln());
        -(half.clone() * z.clone() * z) - sigma.ln() - half * log_2pi
    }

    #[test]
    fn one_formula_runs_on_f64_and_f32() {
        assert!((quadratic(2.5_f64) - 9.75).abs() < 1e-12);
        assert!((quadratic(2.5_f32) - 9.75).abs() < 1e-5);
    }

    #[test]
    fn normal_log_prob_matches_closed_form() {
        // Standard normal at 0: -½·ln(2π).
        let expected = -0.5 * (2.0 * core::f64::consts::PI).ln();
        let got = normal_log_prob(0.0_f64, 0.0, 1.0);
        assert!((got - expected).abs() < 1e-12, "{got} vs {expected}");
    }

    #[test]
    fn softplus_is_stable_at_extremes() {
        assert!((f64::ln_1p_exp(&0.0) - core::f64::consts::LN_2).abs() < 1e-12);
        // Large positive argument must not overflow and is ~x.
        let big = 50.0_f64;
        assert!((big.ln_1p_exp() - big).abs() < 1e-10);
    }

    #[test]
    fn zero_and_one_defaults() {
        assert_eq!(<f64 as Scalar>::zero(), 0.0);
        assert_eq!(<f64 as Scalar>::one(), 1.0);
    }
}
