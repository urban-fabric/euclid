//! The number traits, in five tiers.
//!
//! - **Lazy** ([`Arithmetic`] → [`LazyReal`]/[`LazyComplex`]) — the elementwise analysis
//!   surface over `+ - * /`, `Clone` not `Copy`: transcendentals, rounding/sign, the
//!   comparison-free `min`/`max`/`clamp` selectors, `conj`/magnitude, and scalar-literal
//!   lifting. Every op lowers to a tensor compiler (`Sin`/`Max`/`Select`/…), so a pure
//!   formula runs on every backend. Nothing here returns a scalar `bool` or needs the
//!   concrete number line.
//! - **Semiring** ([`LazySemiring`] → [`LazyNatural`]) — the carriers with `+ *` but no
//!   subtraction, the naturals (a `u64` non-negative count/index/shape array). One rung
//!   below the ring; cross into the signed integers via [`LazyNatural::to_integer`] when a
//!   formula needs subtraction.
//! - **Ring** ([`LazyRing`] → [`LazyInteger`]) — the semiring refined with `-` and negation,
//!   the integers (an `s64` count/index/shape array). The bridge a discrete log-density
//!   crosses into real analysis is [`LazyInteger::to_real`].
//! - **Mask** ([`LazyBool`] + [`LazyCompare`]) — the comparison/`select` surface. Booleans
//!   are a Boolean algebra, not a ring, so they sit beside the tower: a comparison returns
//!   [`LazyCompare::Mask`] (`bool` eagerly, a `pred` tensor on a backend) and `select`
//!   consumes it, expressing comparisons without the scalar `bool` a backend cannot return.
//! - **Concrete** (`Real`/`Complex`, in `napier`) — a `Copy` number with the full API,
//!   fusing this lazy analysis with the algebraic field. `napier::Real: LazyReal`, so a
//!   concrete number satisfies a lazy formula — but a `Tensor` (no `Copy`) is *only* lazy.
//!
//! The word *field* stays reserved for the algebraic `Field`.

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
/// alongside `i64`/`i128` and integer tensors. (Unsigned carriers stop at [`LazySemiring`],
/// having no `Neg`.) [`Arithmetic`] refines it with `Div`; [`LazyInteger`] with the integer
/// surface.
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
/// real number, the real part otherwise) and the `conjugate` every Hermitian formula
/// needs — the identity on a real number, so a `conjugate`/`magnitude` kernel specializes
/// to the real case for free. Refined by [`LazyReal`], [`LazyComplex`], [`LazyQuaternion`].
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
    /// The magnitude `|x|`. Defaults to `√(norm_sqr)`; a real scalar overrides with `abs`.
    fn magnitude(&self) -> Self::RealPart {
        self.norm_sqr().sqrt()
    }
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
