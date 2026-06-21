//! Scalar numeric contracts.
//!
//! The traits are grouped by the structure they require:
//!
//! - [`LazySemiring`] and [`LazyNatural`]: `+`, `*`, identities, natural-number lifting,
//!   Euclidean division, and count combinatorics.
//! - [`LazyRing`] and [`LazyInteger`]: subtraction, negation, signed integer lifting,
//!   Euclidean division, and conversion to real carriers.
//! - [`Arithmetic`]: division on top of the ring operations.
//! - [`LazyReal`]: real analysis, lifted constants, square/powers, rounding/sign
//!   operations, comparison-free selectors, masks, special functions, and stable log-domain
//!   helpers.
//! - [`NormedDivisionAlgebra`], [`LazyComplex`], and [`LazyQuaternion`]: conjugation,
//!   magnitude, real part, and complex/quaternionic construction.
//! - [`LazyBool`] and [`LazyCompare`]: mask algebra, comparisons, `select`, and lazy
//!   control flow.
//!
//! Use the narrowest trait bound that contains the operations a formula needs. Nothing in
//! this module returns a scalar `bool` from a lazy comparison; comparisons return masks.

use core::ops::{Add, Div, Mul, Neg, Sub};

/// The **semiring** substrate: `Clone` plus `+ *` (with identities `0`/`1`), subtraction and
/// negation withheld — the lazy peer of the algebraic `Semiring`, and the tier the naturals
/// (`u32`/`u64`/`u128`, an XLA `u64` count/index array) sit at. Blanket-implemented for every
/// carrier with `Add` and `Mul`, so every ring/field carrier joins it too; [`LazyRing`]
/// refines it with `Sub`/`Neg`, [`LazyNatural`] with the natural surface.
pub trait LazySemiring: Clone + Add<Output = Self> + Mul<Output = Self> {}
impl<T> LazySemiring for T where T: Clone + Add<Output = T> + Mul<Output = T> {}

/// The **ring** substrate: a [`LazySemiring`] plus `-` and negation, `÷` withheld — the lazy
/// peer of the algebraic `Ring`. Blanket-implemented for every carrier with the four ring
/// operators, so the division-bearing scalars (`f64`/`f32`, complex, `Tensor`) join it
/// alongside `i32`/`i64`/`i128` and integer tensors. (Unsigned carriers stop at
/// [`LazySemiring`], having no `Neg`.) [`Arithmetic`] refines it with `Div`;
/// [`LazyInteger`] with the integer surface.
pub trait LazyRing: LazySemiring + Sub<Output = Self> + Neg<Output = Self> {}
impl<T> LazyRing for T where T: LazySemiring + Sub<Output = T> + Neg<Output = T> {}

/// A [`LazyRing`] plus division — the field-tier operator bound behind `+ - * /`,
/// implemented by `f64`/`f32`, complex scalars, and `Tensor`. Integer carriers stop at
/// [`LazyRing`]/[`LazyInteger`]. (Constant lifting lives on [`LazyReal`].)
pub trait Arithmetic: LazyRing + Div<Output = Self> {}

/// A **normed division algebra over ℝ** — the number abstraction the array contracts
/// ([`LazyVector`](crate::array::LazyVector), [`LazyMatrix`](crate::array::LazyMatrix),
/// [`LazyTensor`](crate::array::LazyTensor)) are generic over. By Frobenius/Hurwitz the
/// associative ones are exactly ℝ, ℂ, ℍ, so one matrix/vector kernel runs over all three.
///
/// It names the real field magnitudes, norms, and SPD quantities live in (`Self` for a
/// real number, the real part otherwise), the `conjugate` every Hermitian formula needs,
/// and real-axis constant lifting through [`field_scalar`](Self::field_scalar),
/// [`field_zero`](Self::field_zero), and [`field_one`](Self::field_one). Refined by
/// [`LazyReal`], [`LazyComplex`], [`LazyQuaternion`].
pub trait NormedDivisionAlgebra: Arithmetic {
    /// The real field of magnitudes / norms — `Self` for a real scalar, the real
    /// component for a complex one.
    type RealPart: LazyReal;

    /// The complex conjugate; the identity for a real scalar.
    fn conjugate(&self) -> Self;
    /// The squared magnitude (`x²`, or `re² + im²`) — no `sqrt`.
    fn norm_sqr(&self) -> Self::RealPart;
    /// The real part (the value itself for a real scalar; `re` for complex).
    fn real_part(&self) -> Self::RealPart;
    /// Embed a real value (`re + 0i` for complex; the identity for a real scalar).
    fn from_real(re: Self::RealPart) -> Self;
    /// Lift a host scalar into this division algebra's real axis.
    fn field_scalar(x: f64) -> Self {
        Self::from_real(Self::RealPart::scalar(x))
    }
    /// The additive identity, embedded through the real axis.
    fn field_zero() -> Self {
        Self::field_scalar(0.0)
    }
    /// The multiplicative identity, embedded through the real axis.
    fn field_one() -> Self {
        Self::field_scalar(1.0)
    }
    /// The magnitude `|x|`.
    fn magnitude(&self) -> Self::RealPart;
}

mod boolean;
mod complex;
mod integer;
mod natural;
mod real;
mod real_impl_f32;
mod real_impl_f64;
#[cfg(test)]
mod tests;

/// Crate-internal reference implementations of the special functions the lazy surface
/// names (`erf`/`ln_gamma`/`beta_reg`/`bessel_i0e`/…).
pub(crate) mod special;

pub use boolean::{LazyBool, LazyCompare};
pub use complex::{LazyComplex, LazyQuaternion};
pub use integer::LazyInteger;
pub use natural::LazyNatural;
pub use real::LazyReal;
