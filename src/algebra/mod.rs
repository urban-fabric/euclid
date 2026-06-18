//! The unified algebra: one **semiring/dioid trunk** refined by a **numeric
//! ring/field branch**.
//!
//! - **Trunk** (method-based `⊕`/`⊗`): the property-driven semiring hierarchy —
//!   [`AdditiveMagma`]/[`MultiplicativeMagma`], identities, marker laws, and the
//!   blanket-implemented structures up to [`Dioid`]/[`StarSemiring`]. Operations
//!   are *methods* (`add`/`mul`), not operators, so one carrier can host many
//!   semirings — `min-plus` shortest paths, boolean reachability, counting — none
//!   of which could overload `+`. This is the algebra the graph world (cayley)
//!   binds.
//! - **Numeric branch** (operator-based): [`Ring`] → [`Field`] add the operations
//!   a semiring lacks — additive inverses and division — as `std::ops` (`- /`),
//!   and bind `+ * ` as operators too so numeric/matrix code reads naturally and
//!   the scalar crates (ramanujan) are unchanged. A numeric type bridges the two
//!   layers with a one-liner: its `AdditiveMagma::add` *is* `+`.
//!
//! `f64`/`f32` climb the whole chain `Semiring → Ring → Field`; an idempotent
//! carrier like min-plus stops at `Dioid` (no additive inverse). Analysis
//! (`exp`/`ln`/`conj`) is a separate concern entirely — see [`crate::numeric`].

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ===== Operations: the magma layer (method-based ⊕/⊗) =====================

/// The additive operation `⊕`. By value (`Copy` carriers); the witness/element
/// semirings choose what `add` means (`+`, `min`, `∨`, …).
pub trait AdditiveMagma: Copy + PartialEq {
    fn add(self, rhs: Self) -> Self;
}

/// The multiplicative operation `⊗`.
pub trait MultiplicativeMagma: Copy + PartialEq {
    fn mul(self, rhs: Self) -> Self;
}

/// By-reference `⊕` for a **non-`Copy`** carrier (a bignum, interval, polynomial)
/// that should join the algebra without a clone per op. Blanket-free; a `Copy`
/// carrier delegates (`self.add_ref(rhs) == (*self).add(*rhs)`).
pub trait RefAdditiveMagma: PartialEq {
    #[must_use]
    fn add_ref(&self, rhs: &Self) -> Self;
}

/// By-reference `⊗` for a non-`Copy` carrier.
pub trait RefMultiplicativeMagma: PartialEq {
    #[must_use]
    fn mul_ref(&self, rhs: &Self) -> Self;
}

// ===== Identities =========================================================

/// The additive identity `0` (`+∞` under min-plus, `false` under boolean).
pub trait HasZero: Copy {
    const ZERO: Self;
}

/// The multiplicative identity `1` (`0` under min-plus, `true` under boolean).
pub trait HasOne: Copy {
    const ONE: Self;
}

// ===== Property markers (laws the implementor promises; not machine-checked) =

/// `a ⊕ (b ⊕ c) = (a ⊕ b) ⊕ c`.
pub trait AssociativeAddition {}
/// `a ⊕ b = b ⊕ a`.
pub trait CommutativeAddition {}
/// `a ⊕ a = a` (the additive idempotence that makes a dioid).
pub trait IdempotentAddition {}
/// `a ⊗ (b ⊗ c) = (a ⊗ b) ⊗ c`.
pub trait AssociativeMultiplication {}
/// `a ⊗ b = b ⊗ a`.
pub trait CommutativeMultiplication {}
/// `a ⊗ (b ⊕ c) = a⊗b ⊕ a⊗c` and the right form.
pub trait Distributive {}
/// `0 ⊗ a = a ⊗ 0 = 0`.
pub trait ZeroAnnihilatesMultiplication {}
/// `⊗` preserves the natural order of `⊕` (relaxation stays monotone).
pub trait MonotoneMultiplication {}

// ===== Structures (blanket-derived from operations + identities + laws) =====

pub trait AdditiveSemigroup: AdditiveMagma + AssociativeAddition {}
impl<T: AdditiveMagma + AssociativeAddition> AdditiveSemigroup for T {}

pub trait AdditiveMonoid: AdditiveSemigroup + HasZero {}
impl<T: AdditiveSemigroup + HasZero> AdditiveMonoid for T {}

pub trait CommutativeAdditiveMonoid: AdditiveMonoid + CommutativeAddition {}
impl<T: AdditiveMonoid + CommutativeAddition> CommutativeAdditiveMonoid for T {}

pub trait AdditiveBand: AdditiveSemigroup + IdempotentAddition {}
impl<T: AdditiveSemigroup + IdempotentAddition> AdditiveBand for T {}

pub trait AdditiveSemilattice: AdditiveBand + CommutativeAddition {}
impl<T: AdditiveBand + CommutativeAddition> AdditiveSemilattice for T {}

pub trait MultiplicativeSemigroup: MultiplicativeMagma + AssociativeMultiplication {}
impl<T: MultiplicativeMagma + AssociativeMultiplication> MultiplicativeSemigroup for T {}

pub trait MultiplicativeMonoid: MultiplicativeSemigroup + HasOne {}
impl<T: MultiplicativeSemigroup + HasOne> MultiplicativeMonoid for T {}

pub trait CommutativeMultiplicativeMonoid:
    MultiplicativeMonoid + CommutativeMultiplication
{
}
impl<T: MultiplicativeMonoid + CommutativeMultiplication> CommutativeMultiplicativeMonoid for T {}

/// A **semiring**: a commutative additive monoid and a multiplicative monoid with
/// distributivity and an annihilating zero. The trunk every other structure
/// refines.
pub trait Semiring:
    CommutativeAdditiveMonoid + MultiplicativeMonoid + Distributive + ZeroAnnihilatesMultiplication
{
}
impl<
    T: CommutativeAdditiveMonoid + MultiplicativeMonoid + Distributive + ZeroAnnihilatesMultiplication,
> Semiring for T
{
}

pub trait CommutativeSemiring: Semiring + CommutativeMultiplicativeMonoid {}
impl<T: Semiring + CommutativeMultiplicativeMonoid> CommutativeSemiring for T {}

/// A semiring whose `⊕` is idempotent (`a ⊕ a = a`).
pub trait IdempotentSemiring: Semiring + AdditiveSemilattice {}
impl<T: Semiring + AdditiveSemilattice> IdempotentSemiring for T {}

/// A **dioid** — an idempotent semiring. The home of the algebraic path problem
/// (`min-plus`, boolean, widest-path).
pub trait Dioid: IdempotentSemiring {}
impl<T: IdempotentSemiring> Dioid for T {}

pub trait OrderedSemiring: IdempotentSemiring + MonotoneMultiplication {}
impl<T: IdempotentSemiring + MonotoneMultiplication> OrderedSemiring for T {}

// ===== Star (Kleene closure) and the canonical order ======================

/// A **closed / star semiring**: `star(a) = ONE ⊕ a ⊕ a² ⊕ … = ONE ⊕ a⊗star(a)`,
/// the asteration that solves the algebraic path problem (all-walks closure).
pub trait StarSemiring: Semiring {
    #[must_use]
    fn star(self) -> Self;

    #[inline]
    #[must_use]
    fn plus(self) -> Self {
        self.mul(self.star())
    }
}

/// An **absorptive** dioid (`ONE ⊕ a = ONE`): a cycle never improves on the empty
/// walk, so `star` collapses to `ONE`. Blanket-implements [`StarSemiring`].
pub trait Absorptive: Dioid {}
impl<T: Absorptive> StarSemiring for T {
    #[inline]
    fn star(self) -> Self {
        Self::ONE
    }
}

/// The natural order an idempotent `⊕` induces: `a ≼ b ⟺ a ⊕ b = a` (combining
/// keeps `a`). Under min-plus this is `a ≤ b`; the algebraic justification for
/// relaxation being well-defined on any dioid.
pub trait CanonicalOrder: AdditiveSemilattice {
    #[inline]
    #[must_use]
    fn precedes(self, other: Self) -> bool {
        self.add(other) == self
    }
    #[inline]
    #[must_use]
    fn meet(self, other: Self) -> Self {
        self.add(other)
    }
}
impl<T: AdditiveSemilattice> CanonicalOrder for T {}

// ===== Numeric branch: ring -> field (operator-based) =====================
//
// A semiring has no subtraction or division; the numeric branch adds them as
// `std::ops`, and binds `+`/`*` as operators too so numeric code reads naturally.
// A numeric type bridges the layers with `AdditiveMagma::add(self, rhs) == self + rhs`.

/// A **ring**: a [`Semiring`] whose addition is a group (additive inverses), with
/// the operations exposed as `std::ops`. ℤ is the canonical non-field ring.
pub trait Ring:
    Semiring
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
{
}
impl<T> Ring for T where
    T: Semiring
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Neg<Output = Self>
        + AddAssign
        + SubAssign
        + MulAssign
{
}

pub trait CommutativeRing: Ring + CommutativeMultiplication {}
impl<T: Ring + CommutativeMultiplication> CommutativeRing for T {}

/// Marker: a [`CommutativeRing`] with no zero divisors. ℤ and every field qualify.
pub trait IntegralDomain: CommutativeRing {}
/// Marker: a unique factorization domain.
pub trait Ufd: IntegralDomain {}
/// Marker: a principal ideal domain.
pub trait Pid: Ufd {}

/// A **Euclidean domain**: division with remainder. For a field the remainder is
/// always `ZERO`; for ℤ it is the usual integer division and modulus.
pub trait EuclideanDomain: Pid {
    fn div_rem(self, rhs: Self) -> (Self, Self);
}

/// A **field**: every nonzero element is invertible. The [`conjugate`](Self::conjugate)
/// is the identity for real-like fields — writing adjoint/Hermitian algorithms
/// against it unifies the real and complex cases.
pub trait Field: EuclideanDomain + Div<Output = Self> + DivAssign {
    fn conjugate(self) -> Self;
}

/// Marker: a [`Field`] with a total order compatible with the operations (ℝ, ℚ).
pub trait OrderedField: Field + PartialOrd {}
impl<T: Field + PartialOrd> OrderedField for T {}

/// Marker: a [`Field`] in which every non-constant polynomial has a root (ℂ).
pub trait AlgebraicallyClosedField: Field {}

mod gaussian;
mod impls;
mod integer;
mod modular;
mod rational;
#[cfg(test)]
mod tests;

// The concrete exact number systems — pure algebraic carriers of the tower above.
// Their matrix-kernel impls (`MatrixRing`/`MatrixField`) live in `jacobi`, which owns
// that trait (the orphan rule); here they implement only euclid's own algebra.
pub use gaussian::GaussianRational;
pub use integer::Integer;
pub use modular::{Zn, Zp};
pub use rational::{Rational, Rational128, Rational64};
