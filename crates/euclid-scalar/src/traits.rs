//! Backend-neutral scalar contracts, grouped by the operations a carrier supports.

use core::ops::{Add, Div, Mul, Neg, Sub};

/// The carrier's canonical constants `0` and `1`.
///
/// Every element kind implements it — booleans (where `0`/`1` are `false`/`true`), naturals,
/// integers, and the real/complex/quaternion division algebras — so one bound names a
/// zero/one for all of them. It sits *below* the operators so booleans, which carry `0`/`1`
/// but not `+`/`*`, can share the identities without being a [`Semiring`].
///
/// Both are required rather than defaulted through a host-`f64` lift, so a carrier defines its
/// own exact representation. Arbitrary real lifting stays on [`Real::scalar`].
pub trait Element: Clone {
    /// The additive identity — `0` for numbers, `false` for booleans.
    fn zero() -> Self;
    /// The multiplicative identity — `1` for numbers, `true` for booleans.
    fn one() -> Self;
}

/// `+` and `*` over an [`Element`]'s identities, with subtraction and negation withheld.
/// Naturals sit at this tier.
pub trait Semiring: Element + Add<Output = Self> + Mul<Output = Self> {}
impl<T> Semiring for T where T: Element + Add<Output = T> + Mul<Output = T> {}

/// A [`Semiring`] plus `-` and negation, with division withheld. Signed integers sit at this
/// tier; unsigned carriers stop at [`Semiring`], having no `Neg`.
pub trait Ring: Semiring + Sub<Output = Self> + Neg<Output = Self> {}
impl<T> Ring for T where T: Semiring + Sub<Output = T> + Neg<Output = T> {}

/// A [`Ring`] plus division — the operator bound behind `+ - * /`. Constant lifting lives on
/// [`Real`].
pub trait Arithmetic: Ring + Div<Output = Self> {}

/// A normed division algebra over ℝ. By Frobenius/Hurwitz the associative ones are exactly
/// ℝ, ℂ, ℍ, so one kernel written against this trait runs over all three.
///
/// It names the real field that magnitudes and norms live in, the `conjugate` every Hermitian
/// formula needs, and real-axis embedding through [`from_real`](Self::from_real).
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
mod control_flow;
mod integer;
mod interpolate;
mod natural;
mod real;

pub use boolean::{Boolean, Compare};
pub use complex::{Complex, Quaternion};
pub use control_flow::ControlFlow;
pub use integer::Integer;
pub use interpolate::Interpolate;
pub use natural::Natural;
pub use real::Real;
