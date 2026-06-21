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
//! **Ring tier** ([`LazyRing`] → [`LazyInteger`]) — the same lazy surface for the
//! carriers that have `+ - *` but *no division*: the integers. XLA/JAX expose integer
//! dtypes (`s32`/`s64`) and integer ops, so a count / index / shape array is a
//! first-class lazy value here too; [`Arithmetic`] (the field operators) and
//! [`LazyInteger`] (the integer surface) are the two refinements of [`LazyRing`], the
//! `Ring → Field` split of the algebra branch reflected into the lazy tier. The bridge
//! a discrete log-density crosses — integer count to real analysis — is
//! [`LazyInteger::to_real`].
//!
//! **Mask tier** ([`LazyBool`] + [`LazyCompare`]) — the comparison/`select` surface.
//! Booleans are a Boolean algebra, *not* a ring, so they sit beside the tower: a
//! comparison returns the abstract [`LazyCompare::Mask`] (`bool` eagerly, a `pred` tensor
//! on a backend) and `select` consumes it, so a piecewise / truncated / masked formula is
//! write-once. This is how the lazy tier expresses comparisons without the scalar `bool` a
//! tensor backend cannot return.
//!
//! **Concrete tier** (`Real`/`Complex`, in the `napier` crate) — a `Copy` real /
//! complex *number* with the full API, formed by **fusing** this lazy analysis with
//! the algebraic field (`OrderedField` /
//! `AlgebraicallyClosedField`): one
//! bound gives operators, `ZERO`/`ONE`, `÷`, `conjugate`, ordering, *and*
//! `from_f64` (the concrete-tier spelling for [`LazyReal::scalar`]) plus
//! trig/`hypot`/rounding/constants. `napier::Real: LazyReal` and
//! `napier::Complex: LazyComplex`, so a concrete number also satisfies a lazy
//! formula — but a `Tensor` (no `Copy`, no associated constants) is *only* lazy.
//! Implemented by `f64`/`f32` (and `jacobi::Complex{32,64}`); the geometry / matrix
//! / physics kernels bound on the concrete tier.
//!
//! The word *field* stays reserved for the algebraic
//! `Field`.

use core::ops::{Add, Div, Mul, Neg, Sub};

/// The **ring** substrate: `Clone` plus `+ - *` and negation — `÷` deliberately
/// withheld. The lazy peer of the algebraic `Ring`, and the
/// bound behind a generic formula that uses only the three ring operators. Pure
/// operators, no scalar or vector construction.
///
/// Blanket-implemented for every carrier that has the four ring operators, so the
/// existing division-bearing scalars (`f64`/`f32`, complex, `Tensor`, `Expr`) join it
/// for free, *and* so do `i64`/`i128` and an integer-dtype tensor — which a field bound
/// would (correctly) reject. [`Arithmetic`] refines it with `Div`; [`LazyInteger`]
/// refines it with the integer surface. This is exactly the `Ring → Field` split of the
/// algebra branch, reflected into the lazy tier.
pub trait LazyRing:
    Clone + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Neg<Output = Self>
{
}
impl<T> LazyRing for T where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Neg<Output = T>
{
}

/// The arithmetic substrate: a [`LazyRing`] plus division — the bound behind `+ - * /`
/// in a generic formula. Pure operators, no scalar or vector construction: implemented
/// by `f64`/`f32`, complex scalars, `Tensor`, and `Expr` alike. It is the *field*-tier
/// operator bound; integer carriers stop at [`LazyRing`]/[`LazyInteger`] because they
/// have no `÷`.
/// Lifting a scalar *constant* into a real lazy substrate lives on [`LazyReal`],
/// because probability/statistics formulas need constants without giving up lazy
/// execution.
pub trait Arithmetic: LazyRing + Div<Output = Self> {}

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

mod boolean;
mod complex;
mod integer;
mod real;
mod real_impl_f32;
mod real_impl_f64;
#[cfg(test)]
mod tests;

/// The `f64`/`f32` reference implementations of the special functions the lazy
/// analysis surface names (`erf`/`ln_gamma`/`beta_reg`/`bessel_i0e`/…). Lives under
/// `numeric` because the numeric scalars are its primary consumer; also re-exported
/// as `euclid::special` for backward compatibility.
pub mod special;

pub use boolean::{LazyBool, LazyCompare};
pub use complex::{LazyComplex, LazyQuaternion};
pub use integer::LazyInteger;
pub use real::LazyReal;
