//! Scalar numeric contracts.
//!
//! The traits are grouped by the structure they require:
//!
//! - [`Element`]: the canonical constants `0` and `1` every carrier defines directly — the
//!   tier booleans, naturals, integers, and the real/complex/quaternion division algebras all
//!   unify into, so any of them can be a [`Tensor`](crate::tensor::Tensor) element.
//! - [`Semiring`] and [`Natural`]: `+`, `*`, identities, natural-number lifting,
//!   Euclidean division, and count combinatorics.
//! - [`Ring`] and [`Integer`]: subtraction, negation, signed integer lifting,
//!   Euclidean division, and conversion to real carriers.
//! - [`Arithmetic`]: division on top of the ring operations.
//! - [`Real`]: real analysis, lifted constants, square/powers, rounding/sign
//!   operations, comparison-free selectors, masks, special functions, and stable log-domain
//!   helpers.
//! - [`VectorSpace`], [`InnerProductSpace`], and [`NormedVectorSpace`]:
//!   scalar multiplication, dot products, and norms over a real field.
//! - [`Interpolate`]: scalar-parameterized interpolation (`lerp`/`midpoint`), kept separate
//!   from [`VectorSpace`] so a carrier can supply a stable or domain-specific path.
//! - [`NormedDivisionAlgebra`], [`Complex`], and [`Quaternion`]: conjugation,
//!   magnitude, real part, and complex/quaternionic construction.
//! - [`Boolean`] and [`Compare`]: mask algebra, comparisons, `select`, and carrier-level
//!   control flow.
//!
//! Use the narrowest trait bound that contains the operations a formula needs. Nothing in
//! this module returns a scalar `bool` from a generic comparison; comparisons return masks.

use core::ops::{Add, Div, Mul, Neg, Sub};

/// The carrier's canonical constants — the additive identity `0` and the multiplicative
/// identity `1`. The tier every [`Tensor`](crate::tensor::Tensor) element kind unifies
/// into: booleans (`0`/`1` are `false`/`true`), naturals, integers, and the real/complex/
/// quaternion division algebras all implement it, so structural tensor constructors and
/// reductions can name a zero/one for any of them through one bound.
///
/// This is the supertrait that gives [`Semiring`] its guaranteed identities — factored one
/// tier *below* the operators so booleans, which carry `0`/`1` but not `+`/`*`, can share the
/// identities without being a semiring. Everything above [`Semiring`] (ring, field,
/// integer, natural, division algebra) therefore inherits `0`/`1` for free.
///
/// Both are **required**, not defaulted through a host-`f64` lift: a carrier defines its own
/// exact `0`/`1` representation. (Arbitrary real lifting stays on [`Real::scalar`].)
pub trait Element: Clone {
    /// The additive identity — `0` for numbers, `false` for booleans.
    fn zero() -> Self;
    /// The multiplicative identity — `1` for numbers, `true` for booleans.
    fn one() -> Self;
}

/// The **semiring** substrate: a [`Element`] (which guarantees its identities `0`/`1`) plus
/// `+` and `*`, with subtraction and negation withheld. Naturals sit at this tier.
/// Blanket-implemented for every carrier with [`Element`] + `Add` + `Mul`, so every
/// ring/field carrier joins it too; [`Ring`] refines it with `Sub`/`Neg`, [`Natural`]
/// with the natural surface.
pub trait Semiring: Element + Add<Output = Self> + Mul<Output = Self> {}
impl<T> Semiring for T where T: Element + Add<Output = T> + Mul<Output = T> {}

/// The **ring** substrate: a [`Semiring`] plus `-` and negation, with division withheld.
/// Blanket-implemented for every carrier with the four ring operators, so division-bearing
/// scalars and ring-valued arrays join it alongside signed integers. (Unsigned carriers stop at
/// [`Semiring`], having no `Neg`.) [`Arithmetic`] refines it with `Div`;
/// [`Integer`] with the integer surface.
pub trait Ring: Semiring + Sub<Output = Self> + Neg<Output = Self> {}
impl<T> Ring for T where T: Semiring + Sub<Output = T> + Neg<Output = T> {}

/// A [`Ring`] plus division — the field-tier operator bound behind `+ - * /`,
/// implemented by `f64`/`f32`, complex scalars, and `Tensor`. Integer carriers stop at
/// [`Ring`]/[`Integer`]. (Constant lifting lives on [`Real`].)
pub trait Arithmetic: Ring + Div<Output = Self> {}

/// A module over a real scalar field: an additive abelian group plus multiplication by a scalar.
///
/// The supertrait bound is the additive group only — [`Element`] (for the `0` vector) plus
/// `+`/`-`/negation — **not** [`Ring`]. A vector space has no vector×vector product; requiring
/// [`Ring`] would wrongly force one (that is an *algebra*, a ring that is also a vector space).
///
/// Real, complex, quaternion, and tensor carriers are all vector spaces over their real
/// field — ℝ one-dimensional over itself, ℂ two-dimensional, ℍ four-dimensional. This is a
/// required-method contract: an implementer supplies `scale` over its
/// [`Scalar`](Self::Scalar) field. (euclid does not derive it from
/// [`NormedDivisionAlgebra`]; a carrier that wants the standard division-algebra structure
/// implements `scale` as `self * Self::from_real(factor)`.)
pub trait VectorSpace:
    Element + Add<Output = Self> + Sub<Output = Self> + Neg<Output = Self>
{
    /// The real scalar field this vector space is over.
    type Scalar: Real;

    /// Scalar multiplication. (The additive identity is the `0` vector, [`Element::zero`].)
    fn scale(self, factor: Self::Scalar) -> Self;
}

/// A [`VectorSpace`] with an inner product.
///
/// This is a required-method contract. Implementers choose the numerics for the dot product;
/// euclid does not define it as repeated multiplication and addition.
pub trait InnerProductSpace: VectorSpace {
    /// The inner product in the vector space's scalar field.
    fn dot(self, rhs: Self) -> Self::Scalar;
}

/// An [`InnerProductSpace`] with explicit norm operations.
///
/// Both methods are required so an implementation can use the numerically stable path for its
/// representation instead of inheriting `dot(x, x)` or `sqrt(dot(x, x))`.
pub trait NormedVectorSpace: InnerProductSpace {
    /// The squared norm.
    fn norm_squared(self) -> Self::Scalar;

    /// The norm.
    fn norm(self) -> Self::Scalar;
}

/// A **normed division algebra over ℝ** — the number abstraction the
/// [`Tensor`](crate::tensor::Tensor) array contract is generic over. By Frobenius/Hurwitz the
/// associative ones are exactly ℝ, ℂ, ℍ, so one tensor kernel runs over all three.
///
/// It names the real field magnitudes, norms, and SPD quantities live in (`Self` for a
/// real number, the real part otherwise), the `conjugate` every Hermitian formula needs,
/// and real-axis embedding through [`from_real`](Self::from_real) (with the `0`/`1` constants
/// inherited from [`Element`] via the ring chain). Refined by [`Real`], [`Complex`],
/// [`Quaternion`].
pub trait NormedDivisionAlgebra: Arithmetic {
    /// The real field of magnitudes / norms — `Self` for a real scalar, the real
    /// component for a complex one.
    type RealPart: Real;

    /// The complex conjugate; the identity for a real scalar.
    fn conjugate(&self) -> Self;
    /// The squared magnitude (`x²`, or `re² + im²`) — no `sqrt`.
    fn norm_sqr(&self) -> Self::RealPart;
    /// The real part (the value itself for a real scalar; `re` for complex).
    fn real_part(&self) -> Self::RealPart;
    /// Embed a real value (`re + 0i` for complex; the identity for a real scalar).
    fn from_real(re: Self::RealPart) -> Self;
    /// The magnitude `|x|`.
    fn magnitude(&self) -> Self::RealPart;
}

mod boolean;
mod complex;
mod impls;
mod integer;
mod interpolate;
mod natural;
mod real;
#[cfg(test)]
mod tests;

/// Crate-internal reference implementations of the special functions the public [`Real`] trait
/// names (`erf`/`ln_gamma`/`beta_reg`/`bessel_i0e`/…).
pub(crate) mod special;

pub use boolean::{Boolean, Compare};
pub use complex::{Complex, Quaternion};
pub use integer::Integer;
pub use interpolate::Interpolate;
pub use natural::Natural;
pub use real::Real;
