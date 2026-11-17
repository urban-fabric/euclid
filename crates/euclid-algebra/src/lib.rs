//! # euclid-algebra — the abstract-algebra tower.
//!
//! One **semiring/dioid trunk** refined by a **numeric ring/field branch**, plus the exact
//! algebraic carriers that inhabit them. This crate is pure algebraic *structure*: what `⊕` and
//! `⊗` must satisfy, and which tier a carrier reaches.
//!
//! - **Trunk** (method-based `⊕`/`⊗`): the property-driven semiring hierarchy —
//!   [`AdditiveMagma`]/[`MultiplicativeMagma`], identities, marker laws, and the
//!   blanket-implemented structures up to [`Dioid`]/[`StarSemiring`]. Operations are *methods*
//!   (`add`/`mul`), not operators, so one carrier can host many semirings — `min-plus` shortest
//!   paths, boolean reachability, counting — none of which could overload `+`. This is the
//!   algebra the graph world (cayley) binds.
//! - **Numeric branch** (operator-based): [`Ring`] → [`Field`] add the operations a semiring
//!   lacks — additive inverses and division — as `std::ops` (`- /`), and bind `+ *` as operators
//!   too so numeric/matrix code reads naturally.
//!
//! `f64`/`f32` climb the whole chain `Semiring → Ring → Field`; an idempotent carrier like
//! min-plus stops at [`Dioid`] (no additive inverse).
//!
//! - **Witnessed semirings**: [`witness`] carries the structure in a zero-sized type parameter
//!   instead of in the type, so one domain hosts several algebras at once — `i64` as min-plus,
//!   max-plus, widest-path and arithmetic simultaneously, which coherence forbids on the element
//!   layer. The standard path algebras live there because they have to be agreed on.
//! - **Semimodules**: [`LeftSemimodule`]/[`RightSemimodule`] name the action of a scalar
//!   semiring on a value. Every semiring is one over itself, by blanket impl. This is the tier
//!   a linear-algebra kernel is written over — a matrix–vector product is the action of a matrix
//!   semiring on a vector — so the structure belongs here even though the kernels do not.
//!
//! ## What is not here
//!
//! **Analysis** — `exp`/`ln`, real and complex special functions — lives in `euclid-scalar`.
//! That crate owns both backend-neutral scalar contracts and eager `HostReal`/`HostComplex`
//! traits, alongside native implementations and numerical kernels. It depends on this
//! algebraic tower; the algebraic tower does not depend on analysis.
//! The optional `conformance` feature adds the independent property engine.

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ===== Operations: the magma layer (method-based ⊕/⊗) =====================

/// The additive operation `⊕`. By value (`Copy` carriers); the witness/element
/// semirings choose what `add` means (`+`, `min`, `∨`, …).
pub trait AdditiveMagma: Copy + PartialEq {
    /// Combine with `⊕`.
    ///
    /// A magma promises *closure* and nothing else — no associativity, no identity, no
    /// commutativity. Those are claimed separately by the property markers below, which is what
    /// lets min-plus and the booleans sit in the same tower as `+` on ℤ.
    fn add(self, rhs: Self) -> Self;
}

/// The multiplicative operation `⊗`.
pub trait MultiplicativeMagma: Copy + PartialEq {
    /// Combine with `⊗`. Closure only; the laws are the markers' job, as for
    /// [`add`](AdditiveMagma::add).
    fn mul(self, rhs: Self) -> Self;
}

/// By-reference `⊕` for a **non-`Copy`** carrier (a bignum, interval, polynomial)
/// that should join the algebra without a clone per op. Blanket-free; a `Copy`
/// carrier delegates (`self.add_ref(rhs) == (*self).add(*rhs)`).
pub trait RefAdditiveMagma: PartialEq {
    /// Combine with `⊕` without consuming either operand.
    ///
    /// Must agree with [`AdditiveMagma::add`] where both exist:
    /// `a.add_ref(&b) == a.add(b)` for a `Copy` carrier.
    #[must_use]
    fn add_ref(&self, rhs: &Self) -> Self;
}

/// By-reference `⊗` for a non-`Copy` carrier.
pub trait RefMultiplicativeMagma: PartialEq {
    /// Combine with `⊗` without consuming either operand, agreeing with
    /// [`MultiplicativeMagma::mul`] where both exist.
    #[must_use]
    fn mul_ref(&self, rhs: &Self) -> Self;
}

// ===== Identities =========================================================

/// The additive identity `0` (`+∞` under min-plus, `false` under boolean).
pub trait HasZero: Copy {
    /// The value `a ⊕ ZERO = a` holds for. **Not necessarily the number zero** — it is `+∞`
    /// under min-plus and `false` under the booleans, which is the whole reason this is an
    /// associated constant rather than a literal.
    const ZERO: Self;
}

/// The multiplicative identity `1` (`0` under min-plus, `true` under boolean).
pub trait HasOne: Copy {
    /// The value `a ⊗ ONE = a` holds for. `0` under min-plus, `true` under the booleans.
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
//
// **Do not implement any of these.** Every one is blanket-implemented for whatever satisfies its
// bounds, so a carrier reaches a tier by implementing the operations and claiming the markers,
// and the tier name is then a *consequence* rather than a second assertion to keep in step. A
// manual impl would conflict with the blanket one, which is the compiler enforcing that.
//
// They exist to be *named in a bound*. `fn relax<T: Dioid>` says exactly what it needs, and the
// carrier side of that contract is the markers.

/// `⊕` is associative. Blanket-derived; see the note above.
pub trait AdditiveSemigroup: AdditiveMagma + AssociativeAddition {}
impl<T: AdditiveMagma + AssociativeAddition> AdditiveSemigroup for T {}

/// An associative `⊕` with an identity. Blanket-derived.
pub trait AdditiveMonoid: AdditiveSemigroup + HasZero {}
impl<T: AdditiveSemigroup + HasZero> AdditiveMonoid for T {}

/// An additive monoid whose `⊕` also commutes — the additive half of a [`Semiring`]. Blanket-derived.
pub trait CommutativeAdditiveMonoid: AdditiveMonoid + CommutativeAddition {}
impl<T: AdditiveMonoid + CommutativeAddition> CommutativeAdditiveMonoid for T {}

/// An associative `⊕` that is also idempotent (`a ⊕ a = a`). Blanket-derived.
pub trait AdditiveBand: AdditiveSemigroup + IdempotentAddition {}
impl<T: AdditiveSemigroup + IdempotentAddition> AdditiveBand for T {}

/// An idempotent, commutative, associative `⊕` — so `⊕` is a meet and
/// [`CanonicalOrder`] applies. Blanket-derived.
pub trait AdditiveSemilattice: AdditiveBand + CommutativeAddition {}
impl<T: AdditiveBand + CommutativeAddition> AdditiveSemilattice for T {}

/// `⊗` is associative. Blanket-derived.
pub trait MultiplicativeSemigroup: MultiplicativeMagma + AssociativeMultiplication {}
impl<T: MultiplicativeMagma + AssociativeMultiplication> MultiplicativeSemigroup for T {}

/// An associative `⊗` with an identity. Blanket-derived.
pub trait MultiplicativeMonoid: MultiplicativeSemigroup + HasOne {}
impl<T: MultiplicativeSemigroup + HasOne> MultiplicativeMonoid for T {}

/// A multiplicative monoid whose `⊗` also commutes. Blanket-derived.
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

/// A [`Semiring`] whose `⊗` commutes. Blanket-derived. ℤ and every field qualify; matrix
/// multiplication does not.
pub trait CommutativeSemiring: Semiring + CommutativeMultiplicativeMonoid {}
impl<T: Semiring + CommutativeMultiplicativeMonoid> CommutativeSemiring for T {}

/// A semiring whose `⊕` is idempotent (`a ⊕ a = a`).
pub trait IdempotentSemiring: Semiring + AdditiveSemilattice {}
impl<T: Semiring + AdditiveSemilattice> IdempotentSemiring for T {}

/// A **dioid** — an idempotent semiring. The home of the algebraic path problem
/// (`min-plus`, boolean, widest-path).
pub trait Dioid: IdempotentSemiring {}
impl<T: IdempotentSemiring> Dioid for T {}

/// A [`Dioid`] whose `⊗` preserves the natural order of `⊕`. Blanket-derived.
///
/// The tier a shortest-path relaxation actually needs: monotone `⊗` is what makes "extending a
/// better prefix cannot give a worse walk" true, and so what makes label-setting correct.
pub trait OrderedSemiring: IdempotentSemiring + MonotoneMultiplication {}
impl<T: IdempotentSemiring + MonotoneMultiplication> OrderedSemiring for T {}

// ===== Star (Kleene closure) and the canonical order ======================

/// A **closed / star semiring**: `star(a) = ONE ⊕ a ⊕ a² ⊕ … = ONE ⊕ a⊗star(a)`,
/// the asteration that solves the algebraic path problem (all-walks closure).
pub trait StarSemiring: Semiring {
    /// `ONE ⊕ a ⊕ a² ⊕ …` — the closure over walks of *any* length, including the empty one.
    ///
    /// Must satisfy the fixed-point identity `star(a) = ONE ⊕ a ⊗ star(a)`, which is what an
    /// iterative solver relies on to terminate. For an [`Absorptive`] dioid this is `ONE`.
    #[must_use]
    fn star(self) -> Self;

    /// `a ⊗ star(a)` — the closure over walks of length **one or more**.
    ///
    /// [`star`](Self::star) without the empty walk. Defaulted, and an implementor should
    /// override only if there is a cheaper direct form.
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
    /// `a ≼ b`, i.e. `a ⊕ b = a` — combining with `b` does not improve on `a`.
    ///
    /// Under min-plus this is `a ≤ b`. Note it is a *partial* order in general: two elements may
    /// be incomparable, so a relaxation must not assume totality.
    #[inline]
    #[must_use]
    fn precedes(self, other: Self) -> bool {
        self.add(other) == self
    }

    /// The greatest lower bound under [`precedes`](Self::precedes), which an idempotent `⊕`
    /// already is — so this is `⊕` under the name that says what it means order-theoretically.
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

/// A [`Ring`] whose `⊗` commutes. Blanket-derived.
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
    /// Quotient and remainder together, satisfying `self == q ⊗ rhs + r`.
    ///
    /// The remainder's sign convention is the carrier's to state — ℤ here follows Rust's `%`,
    /// which truncates towards zero, so `(-7).div_rem(2)` is `(-3, -1)` rather than `(-4, 1)`.
    /// A carrier wanting the least non-negative residue should say so in its own docs.
    ///
    /// # Panics
    /// Implementations may panic when `rhs` is [`ZERO`](HasZero::ZERO); division by zero is the
    /// caller's precondition rather than a case this signature can report.
    fn div_rem(self, rhs: Self) -> (Self, Self);
}

/// A **field**: every nonzero element is invertible. The [`conjugate`](Self::conjugate)
/// is the identity for real-like fields — writing adjoint/Hermitian algorithms
/// against it unifies the real and complex cases.
pub trait Field: EuclideanDomain + Div<Output = Self> + DivAssign {
    /// The field conjugate: **the identity for a real field**, complex conjugation for ℂ.
    ///
    /// Required even where it does nothing, which is the point: an adjoint or Hermitian
    /// algorithm written with `conjugate` is correct over both ℝ and ℂ, where one written
    /// without it is silently real-only.
    fn conjugate(self) -> Self;
}

/// Marker: a [`Field`] with a total order compatible with the operations (ℝ, ℚ).
pub trait OrderedField: Field + PartialOrd {}
impl<T: Field + PartialOrd> OrderedField for T {}

/// Marker: a [`Field`] in which every non-constant polynomial has a root (ℂ).
pub trait AlgebraicallyClosedField: Field {}

#[cfg(any(feature = "conformance", test))]
pub mod conformance;

mod gaussian;
mod impls;
mod integer;
mod modular;
mod rational;
mod semimodule;
#[cfg(test)]
mod tests;
pub mod witness;

// The concrete exact number systems — pure algebraic carriers of the tower above.
// Their matrix-kernel impls (`MatrixRing`/`MatrixField`) live in `jacobi`, which owns
// that trait (the orphan rule); here they implement only euclid's own algebra.
pub use gaussian::GaussianRational;
pub use integer::Integer;
pub use modular::{Zn, Zp};
pub use rational::{Rational, Rational64, Rational128};
pub use semimodule::{LeftSemimodule, RightSemimodule};

// The README's examples, compiled.
//
// `cfg(doctest)` means this item exists only while rustdoc is collecting doctests: it is absent
// from every real build, adds nothing to the public surface, and never appears in the rendered
// docs. What it buys is that `cargo test --doc` compiles the code blocks in README.md itself.
//
// A crate whose deliverable is its trait surface has a README whose example *is* part of that
// surface, and a README example that no longer compiles is worse than none — it documents a
// signature the crate no longer has. Two crates used to check this by hand-copying the example
// into `tests/readme_check.rs`, which compiles the copy and lets the original drift. This checks
// the original.
//
// Illustrative fragments in the README are marked ```rust,ignore, which is the honest marking:
// they reference names that are deliberately undefined.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;
