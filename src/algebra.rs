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

pub trait CommutativeMultiplicativeMonoid: MultiplicativeMonoid + CommutativeMultiplication {}
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

// ===== f64 / f32: numeric carriers climbing Semiring -> Ring -> Field ======
//
// The magma `add`/`mul` ARE `+`/`*` (the bridge between trunk and numeric branch).
// Reals are associative/commutative/distributive but not additively idempotent,
// so they are Semirings/Rings/Fields -- never dioids.

// Ring / CommutativeRing / OrderedField are blanket-derived for both; only the
// magma bridge, identities, law markers, and the two method-bearing traits
// (EuclideanDomain::div_rem, Field::conjugate) are written per type.

impl AdditiveMagma for f64 {
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}
impl MultiplicativeMagma for f64 {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
}
impl RefAdditiveMagma for f64 {
    #[inline]
    fn add_ref(&self, rhs: &Self) -> Self {
        *self + *rhs
    }
}
impl RefMultiplicativeMagma for f64 {
    #[inline]
    fn mul_ref(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
}
impl HasZero for f64 {
    const ZERO: Self = 0.0;
}
impl HasOne for f64 {
    const ONE: Self = 1.0;
}
impl AssociativeAddition for f64 {}
impl CommutativeAddition for f64 {}
impl AssociativeMultiplication for f64 {}
impl CommutativeMultiplication for f64 {}
impl Distributive for f64 {}
impl ZeroAnnihilatesMultiplication for f64 {}
impl IntegralDomain for f64 {}
impl Ufd for f64 {}
impl Pid for f64 {}
impl EuclideanDomain for f64 {
    #[inline]
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self / rhs, 0.0)
    }
}
impl Field for f64 {
    #[inline]
    fn conjugate(self) -> Self {
        self
    }
}

impl AdditiveMagma for f32 {
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}
impl MultiplicativeMagma for f32 {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
}
impl RefAdditiveMagma for f32 {
    #[inline]
    fn add_ref(&self, rhs: &Self) -> Self {
        *self + *rhs
    }
}
impl RefMultiplicativeMagma for f32 {
    #[inline]
    fn mul_ref(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
}
impl HasZero for f32 {
    const ZERO: Self = 0.0;
}
impl HasOne for f32 {
    const ONE: Self = 1.0;
}
impl AssociativeAddition for f32 {}
impl CommutativeAddition for f32 {}
impl AssociativeMultiplication for f32 {}
impl CommutativeMultiplication for f32 {}
impl Distributive for f32 {}
impl ZeroAnnihilatesMultiplication for f32 {}
impl IntegralDomain for f32 {}
impl Ufd for f32 {}
impl Pid for f32 {}
impl EuclideanDomain for f32 {
    #[inline]
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self / rhs, 0.0)
    }
}
impl Field for f32 {
    #[inline]
    fn conjugate(self) -> Self {
        self
    }
}

// ===== i64 / i128: the canonical non-field rings (ℤ) =======================
//
// Integers climb Semiring -> Ring -> CommutativeRing -> ... -> EuclideanDomain
// but stop short of Field: there is no multiplicative inverse. `div_rem` is the
// usual integer division and modulus (the exact-division primitive the
// fraction-free algorithms build on). The magma `add`/`mul` ARE `+`/`*` (the
// trunk<->numeric-branch bridge). They are ordered, but euclid's OrderedField
// marker is field-only, so order lives on the consumer (`Ord`) rather than here.

impl AdditiveMagma for i64 {
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}
impl MultiplicativeMagma for i64 {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
}
impl RefAdditiveMagma for i64 {
    #[inline]
    fn add_ref(&self, rhs: &Self) -> Self {
        *self + *rhs
    }
}
impl RefMultiplicativeMagma for i64 {
    #[inline]
    fn mul_ref(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
}
impl HasZero for i64 {
    const ZERO: Self = 0;
}
impl HasOne for i64 {
    const ONE: Self = 1;
}
impl AssociativeAddition for i64 {}
impl CommutativeAddition for i64 {}
impl AssociativeMultiplication for i64 {}
impl CommutativeMultiplication for i64 {}
impl Distributive for i64 {}
impl ZeroAnnihilatesMultiplication for i64 {}
impl IntegralDomain for i64 {}
impl Ufd for i64 {}
impl Pid for i64 {}
impl EuclideanDomain for i64 {
    #[inline]
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self / rhs, self % rhs)
    }
}

impl AdditiveMagma for i128 {
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}
impl MultiplicativeMagma for i128 {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
}
impl RefAdditiveMagma for i128 {
    #[inline]
    fn add_ref(&self, rhs: &Self) -> Self {
        *self + *rhs
    }
}
impl RefMultiplicativeMagma for i128 {
    #[inline]
    fn mul_ref(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
}
impl HasZero for i128 {
    const ZERO: Self = 0;
}
impl HasOne for i128 {
    const ONE: Self = 1;
}
impl AssociativeAddition for i128 {}
impl CommutativeAddition for i128 {}
impl AssociativeMultiplication for i128 {}
impl CommutativeMultiplication for i128 {}
impl Distributive for i128 {}
impl ZeroAnnihilatesMultiplication for i128 {}
impl IntegralDomain for i128 {}
impl Ufd for i128 {}
impl Pid for i128 {}
impl EuclideanDomain for i128 {
    #[inline]
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self / rhs, self % rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A numeric formula written once over any field, using operators.
    fn affine_ratio<T: Field>(a: T, b: T) -> T {
        (a * b + T::ONE) / a
    }

    #[test]
    fn reals_climb_semiring_ring_field() {
        assert_eq!(affine_ratio(2.0_f64, 3.0), (2.0 * 3.0 + 1.0) / 2.0);
        assert_eq!(affine_ratio(2.0_f32, 3.0), (2.0 * 3.0 + 1.0) / 2.0);
        assert_eq!(2.5_f64.conjugate(), 2.5);
        assert_eq!(7.0_f64.div_rem(2.0), (3.5, 0.0));
        // The trunk's magma `add` bridges to `+`.
        assert_eq!(AdditiveMagma::add(2.0_f64, 3.0), 5.0);
    }

    #[test]
    fn integers_are_euclidean_domains_not_fields() {
        // ℤ climbs Semiring -> Ring -> EuclideanDomain but is not a Field.
        fn assert_euclidean<T: EuclideanDomain>(_: T) {}
        assert_euclidean(7_i64);
        assert_euclidean(7_i128);
        assert_eq!(7_i64.div_rem(3), (2, 1));
        assert_eq!((-7_i128).div_rem(3), (-2, -1));
        assert_eq!(AdditiveMagma::add(2_i64, 3), 5);
        assert_eq!(<i64 as HasZero>::ZERO, 0);
        assert_eq!(<i128 as HasOne>::ONE, 1);
    }

    /// A minimal idempotent dioid (boolean reachability) to exercise the trunk's
    /// blanket Dioid / StarSemiring / CanonicalOrder, independent of any carrier
    /// that lives in the graph crate.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct Reach(bool);
    impl AdditiveMagma for Reach {
        fn add(self, rhs: Self) -> Self {
            Reach(self.0 || rhs.0)
        }
    }
    impl MultiplicativeMagma for Reach {
        fn mul(self, rhs: Self) -> Self {
            Reach(self.0 && rhs.0)
        }
    }
    impl HasZero for Reach {
        const ZERO: Self = Reach(false);
    }
    impl HasOne for Reach {
        const ONE: Self = Reach(true);
    }
    impl AssociativeAddition for Reach {}
    impl CommutativeAddition for Reach {}
    impl IdempotentAddition for Reach {}
    impl AssociativeMultiplication for Reach {}
    impl CommutativeMultiplication for Reach {}
    impl Distributive for Reach {}
    impl ZeroAnnihilatesMultiplication for Reach {}
    impl Absorptive for Reach {}

    #[test]
    fn boolean_dioid_has_blanket_dioid_star_and_order() {
        fn assert_dioid<T: Dioid>(_: T) {}
        assert_dioid(Reach(true));
        // Absorptive => star collapses to ONE.
        assert_eq!(Reach(false).star(), Reach::ONE);
        // Canonical order: reachable (true) precedes unreachable (false).
        assert!(Reach(true).precedes(Reach(false)));
        assert!(!Reach(false).precedes(Reach(true)));
    }
}
