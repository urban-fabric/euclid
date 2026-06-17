//! The **numeric analysis** traits: a single source of truth for *what
//! element-wise arithmetic and analysis a formula needs*, so each formula is
//! written **once** and runs on every backend — scalar `f64`/`f32`, complex
//! scalars, differentiable tensors, and columnar expressions.
//!
//! These are the **analysis** view of a number (`exp`, `ln`, `sqrt`, `conj`,
//! magnitude), as opposed to [`crate::algebra`], which is the **structure** view
//! (ring/field laws, `div_rem`, ordering). The word *field* is reserved for the
//! algebraic [`Field`](crate::algebra::Field); these traits are named `*Scalar` to
//! avoid that overload.
//!
//! They are **not** `Copy`-bound: a lazy tensor handle is `Clone` but not `Copy`,
//! so operators are taken by value over cheap-to-clone operands.
//!
//! Real and complex are kept distinct: `f64`/`f32` are [`RealScalar`]s, not
//! complex. Code that must treat real and complex scalars uniformly (dense linear
//! algebra) does so through its own scalar trait that both kinds implement, rather
//! than by forcing a real to masquerade as a complex.

use core::ops::{Add, Div, Mul, Neg, Sub};

/// The arithmetic substrate: `Clone` plus the four operations and negation, with a
/// constructor from a real literal. The bound behind `+ - * /` in a generic
/// formula; implemented by `f64`/`f32`, complex scalars, `Tensor`, and `Expr`.
pub trait Scalar:
    Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Lift a real constant into the type (`f64` identity; `x + 0i` for complex; a
    /// scalar tensor for `Tensor`).
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

/// A real-valued numeric scalar: a [`Scalar`] plus the real transcendentals that
/// inherently-real formulas (probability densities, geometry kernels) are written
/// in.
pub trait RealScalar: Scalar {
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
    /// Numerically-stable `ln(1 + e^x)` (softplus / `log1pexp`).
    fn ln_1p_exp(&self) -> Self;
}

/// A complex-valued numeric scalar: a [`Scalar`] plus complex structure,
/// parameterized by its real-component type. Implemented by genuinely-complex
/// types — `Complex{32,64}` and a complex-dtype `Tensor` — **not** by the real
/// scalars (`f64`/`f32` are [`RealScalar`]s).
///
/// Construction from explicit `(re, im)` parts is **not** here; it lives on the
/// concrete complex constructors (e.g. `Complex64::new`,
/// `Tensor::complex_from_parts`).
pub trait ComplexScalar: Scalar {
    /// The real-component type — `f64` for `Complex64`, `Self` for a dtype-erased
    /// complex `Tensor`.
    type Real: RealScalar;

    /// Complex conjugate.
    fn conj(&self) -> Self;
    /// Real part.
    fn re(&self) -> Self::Real;
    /// Imaginary part.
    fn im(&self) -> Self::Real;
    /// Squared magnitude `re² + im²`.
    fn norm_sqr(&self) -> Self::Real;
    /// The magnitude `|z| = √(re² + im²)`.
    fn magnitude(&self) -> Self::Real {
        self.norm_sqr().sqrt()
    }
}

// --- f64 / f32: the real reference (also degenerate complex) ---------------

impl Scalar for f64 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
}

impl RealScalar for f64 {
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
        // Stable softplus: for x > 0 factor out x to avoid overflow in e^x; for
        // x <= 0 evaluate ln(1 + e^x) directly.
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

impl RealScalar for f32 {
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

    /// A formula written once over the real scalars.
    fn quadratic<T: RealScalar>(x: T) -> T {
        x.clone() * x.clone() + x + T::one()
    }

    #[test]
    fn real_formula_runs_on_f64_and_f32() {
        assert!((quadratic(2.5_f64) - 9.75).abs() < 1e-12);
        assert!((quadratic(2.5_f32) - 9.75).abs() < 1e-5);
    }

    #[test]
    fn softplus_is_stable_at_extremes() {
        assert!((f64::ln_1p_exp(&0.0) - core::f64::consts::LN_2).abs() < 1e-12);
        let big = 50.0_f64;
        assert!((big.ln_1p_exp() - big).abs() < 1e-10);
    }
}
