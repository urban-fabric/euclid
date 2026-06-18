//! The number traits, in two tiers.
//!
//! **Lazy tier** ([`Arithmetic`] → [`LazyReal`]/[`LazyComplex`]) — the *elementwise
//! analysis* surface over the four operators, `Clone` not `Copy`: the
//! transcendentals (`exp`/`ln`/`sqrt`, trig, `atan2`/`hypot`), rounding/sign, the
//! comparison-free `min`/`max`/`clamp` selectors, `conj`/magnitude, plus
//! scalar-literal lifting for the constants formulas name. Every operation here is
//! something a lazy backend (a differentiable `Tensor`, a columnar `Expr`) can
//! lower — a tensor compiler maps them to `Sin`/`Cos`/`Max`/`Select`/… — so a pure
//! formula written in analysis + operators + scalar constants runs on every
//! backend. What is *not* here: anything that returns a scalar `bool` (a tensor
//! comparison yields a mask, not a bool) or that needs the concrete number line.
//!
//! **Concrete tier** ([`Real`]/[`Complex`]) — a `Copy` real / complex *number* with
//! the full API, formed by **fusing** the lazy analysis with the algebraic field
//! ([`OrderedField`](crate::algebra::OrderedField) /
//! [`AlgebraicallyClosedField`](crate::algebra::AlgebraicallyClosedField)): one
//! bound gives operators, `ZERO`/`ONE`, `÷`, `conjugate`, ordering, *and*
//! `from_f64` (the concrete-tier spelling for [`LazyReal::scalar`]) plus
//! trig/`hypot`/rounding/constants. `Real: LazyReal` and `Complex: LazyComplex`, so
//! a concrete number also satisfies a lazy formula — but a `Tensor` (no `Copy`, no
//! associated constants) is *only* lazy. Implemented by `f64`/`f32` (and
//! `jacobi::Complex{32,64}`); the geometry / matrix / physics kernels bound on
//! the concrete tier.
//!
//! The word *field* stays reserved for the algebraic
//! [`Field`](crate::algebra::Field).

use core::ops::{Add, Div, Mul, Neg, Sub};

/// The arithmetic substrate: `Clone` plus the four operations and negation — the
/// bound behind `+ - * /` in a generic formula. Pure operators, no scalar or vector
/// construction: implemented by `f64`/`f32`, complex scalars, `Tensor`, and `Expr`
/// alike.
/// Lifting a scalar *constant* into a real lazy substrate lives on [`LazyReal`],
/// because probability/statistics formulas need constants without giving up lazy
/// execution.
pub trait Arithmetic:
    Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
}

/// A **normed division algebra over ℝ** — the number-system abstraction the rank-graded
/// array contracts ([`LazyVector`](crate::array::LazyVector),
/// [`LazyMatrix`](crate::array::LazyMatrix), [`LazyTensor`](crate::array::LazyTensor))
/// are generic over. By Frobenius/Hurwitz the associative ones are exactly **ℝ (1D), ℂ
/// (2D), ℍ (4D)**, so one matrix/vector kernel runs over all three (octonions are
/// non-associative — no matrices).
///
/// It names the **real field** the magnitudes, norms, and SPD quantities
/// (log-determinant, Mahalanobis distance) live in — `Self` for a real number, the real
/// part otherwise — and the [`conjugate`](NormedDivisionAlgebra::conjugate) every
/// Hermitian formula needs. `conjugate` is the **identity** on a real number, so a kernel
/// written with `conjugate`/`magnitude` specializes to the real (symmetric/orthogonal)
/// case for free.
///
/// [`LazyReal`], [`LazyComplex`], and [`LazyQuaternion`] refine it: `f64`/`f32` implement
/// it with `RealPart = Self`; the hypercomplex numbers (ℂ/ℍ) implement it — in the crate
/// that owns them — with `RealPart` the real component. It carries no number line of its
/// own, so a tensor backend lowers it. (This is a *number*, not the relational *scalar*
/// role — that stays the `Scalar` associated type of a vector space / matrix.)
pub trait NormedDivisionAlgebra: Arithmetic {
    /// The real field of magnitudes / norms — `Self` for a real scalar, the real
    /// component for a complex one.
    type RealPart: LazyReal;

    /// The complex conjugate; the **identity** for a real scalar.
    fn conjugate(&self) -> Self;
    /// The squared magnitude (`x²`, or `re² + im²`) — no `sqrt`.
    fn norm_sqr(&self) -> Self::RealPart;
    /// The real part (the value itself for a real scalar; `re` for complex).
    fn real_part(&self) -> Self::RealPart;
    /// Embed a real value (`re + 0i` for complex; the identity for a real scalar).
    fn from_real(re: Self::RealPart) -> Self;
    /// The magnitude `|x|` (`|z| = √(re² + im²)` for complex). Defaults to `√(norm_sqr)`;
    /// a real scalar overrides it with `abs`.
    fn magnitude(&self) -> Self::RealPart {
        self.norm_sqr().sqrt()
    }
}

mod complex;
mod f32_impl;
mod f64_impl;
mod real;
#[cfg(test)]
mod tests;

pub use complex::{Complex, LazyComplex, LazyQuaternion};
pub use real::{LazyReal, Real};
