//! Witnessed semirings — the structure as a type parameter rather than as the type.
//!
//! The rest of this crate makes **the type *be* the semiring**: `i64` carries ordinary
//! arithmetic through its [`AdditiveMagma`](crate::AdditiveMagma) impl, and that is the only
//! semiring it can carry, because Rust's coherence rules forbid a second impl of the same trait
//! for the same type. That is the right default — a formula bounded on
//! [`Semiring`](crate::Semiring) reads naturally and the compiler picks the structure — but it
//! means a plain `i64` can be *exactly one* semiring, and there are several it ought to be.
//!
//! A **witness** lifts the structure into a zero-sized type parameter, so one domain carries
//! many semirings at once and an algorithm selects among them at the call site:
//!
//! ```
//! use euclid_algebra::witness::{MaxPlus, MinPlus, SemiringWitness};
//!
//! // The same `i64` values, read as two different algebras.
//! assert_eq!(MinPlus::add(3, 5), 3); // ⊕ is min
//! assert_eq!(MaxPlus::add(3, 5), 5); // ⊕ is max
//!
//! // Each has its own identities: min-plus annihilates at +∞, max-plus at −∞.
//! assert_eq!(MinPlus::mul(MinPlus::ZERO, 5), MinPlus::ZERO);
//! assert_eq!(MinPlus::mul(MinPlus::ONE, 5), 5);
//! ```
//!
//! # Why these live here
//!
//! Because they have to be agreed. Two independently-written path solvers must concur that
//! min-plus is `(min, +)` with `ZERO = +∞` and `ONE = 0`, *and* that `∞ + x = ∞` rather than
//! wrapping — the saturating guard in [`MinPlus::mul`] is a correctness requirement, not an
//! implementation detail. A second definition of min-plus is a second chance to get that wrong,
//! and the symptom is a shortest path that is quietly too long.
//!
//! The *algorithms* over these witnesses are a different matter and belong with a backend: a
//! dense closure, a Gustavson sparse product and a blocked matrix multiply are all valid, and
//! choosing between them is exactly what a backend is for. This module names the algebras and
//! stops.
//!
//! # Witness or element?
//!
//! Prefer the element layer. `T: Semiring` is the wider bound, reads better, and composes with
//! the rest of the tower — the property markers, the blanket impls, `Ring`, `Field`.
//!
//! Reach for a witness only when one domain genuinely needs several structures at once, which
//! in practice means a kernel parameterised by its algebra: the same matrix closure computing
//! shortest paths under [`MinPlus`], critical paths under [`MaxPlus`] and bottlenecks under
//! [`WidestPath`], with no algorithm code per semiring.
//!
//! The witness layer is deliberately lean for that reason: one trait, one marker for the single
//! property that gates an algorithm, and the standard structures. The full property
//! decomposition lives once, on the element layer.

/// A semiring whose structure is carried by the implementing type rather than by its domain.
///
/// `Self` is uninhabited or zero-sized — it exists only to name the algebra. The domain is the
/// associated [`Domain`](Self::Domain), and the operations are associated functions over it.
///
/// Implementors must satisfy the semiring axioms, which
/// [`conformance::witness_semiring_laws`](crate::conformance::witness_semiring_laws) checks.
/// Note that nothing in this signature enforces them: there are no property markers here, by
/// design, since the marker tower belongs to the element layer.
pub trait SemiringWitness {
    /// The values the algebra operates on.
    type Domain: Copy + PartialEq;

    /// The additive identity, which must also annihilate `⊗`.
    const ZERO: Self::Domain;

    /// The multiplicative identity.
    const ONE: Self::Domain;

    /// The additive operation `⊕`.
    fn add(a: Self::Domain, b: Self::Domain) -> Self::Domain;

    /// The multiplicative operation `⊗`.
    fn mul(a: Self::Domain, b: Self::Domain) -> Self::Domain;
}

/// Marker: a [`SemiringWitness`] whose `⊕` is idempotent (`a ⊕ a = a`).
///
/// The one property that gates an algorithm rather than merely describing a structure — a
/// closure or fixed-point iteration terminates for a dioid and need not otherwise. Checked by
/// [`conformance::witness_dioid_laws`](crate::conformance::witness_dioid_laws).
pub trait DioidWitness: SemiringWitness {}

/// **Min-plus** (the tropical semiring): `⊕` is `min`, `⊗` is `+`.
///
/// The algebra of shortest paths. `ZERO` is `+∞` (no path) and `ONE` is `0` (the empty path).
/// Addition saturates and treats `ZERO` as absorbing, so extending a nonexistent path leaves it
/// nonexistent rather than wrapping to a small number — which is the bug this definition exists
/// to make impossible.
pub enum MinPlus {}

impl SemiringWitness for MinPlus {
    type Domain = i64;
    const ZERO: i64 = i64::MAX;
    const ONE: i64 = 0;

    #[inline]
    fn add(a: i64, b: i64) -> i64 {
        a.min(b)
    }

    #[inline]
    fn mul(a: i64, b: i64) -> i64 {
        if a == Self::ZERO || b == Self::ZERO {
            Self::ZERO
        } else {
            a.saturating_add(b)
        }
    }
}
impl DioidWitness for MinPlus {}

/// **Max-plus**: `⊕` is `max`, `⊗` is `+`.
///
/// The algebra of longest and critical paths — scheduling, not routing. `ZERO` is `−∞`, `ONE` is
/// `0`, with the same absorbing guard as [`MinPlus`].
pub enum MaxPlus {}

impl SemiringWitness for MaxPlus {
    type Domain = i64;
    const ZERO: i64 = i64::MIN;
    const ONE: i64 = 0;

    #[inline]
    fn add(a: i64, b: i64) -> i64 {
        a.max(b)
    }

    #[inline]
    fn mul(a: i64, b: i64) -> i64 {
        if a == Self::ZERO || b == Self::ZERO {
            Self::ZERO
        } else {
            a.saturating_add(b)
        }
    }
}
impl DioidWitness for MaxPlus {}

/// **Widest path** (the bottleneck semiring): `⊕` is `max`, `⊗` is `min`.
///
/// The algebra of capacity. A path's value is its narrowest link, and the best path is the
/// widest of those — so `⊗` is `min` rather than `+`, and no saturating guard is needed because
/// neither operation can overflow.
pub enum WidestPath {}

impl SemiringWitness for WidestPath {
    type Domain = i64;
    const ZERO: i64 = i64::MIN;
    const ONE: i64 = i64::MAX;

    #[inline]
    fn add(a: i64, b: i64) -> i64 {
        a.max(b)
    }

    #[inline]
    fn mul(a: i64, b: i64) -> i64 {
        a.min(b)
    }
}
impl DioidWitness for WidestPath {}

/// **Ordinary arithmetic** as a witness: `⊕` is `+`, `⊗` is `×`, both saturating.
///
/// The algebra of walk counting and weighting. Not a [`DioidWitness`] — `a + a ≠ a` — which is
/// exactly why a closure algorithm must not accept it, and the reason that marker exists.
pub enum Arithmetic {}

impl SemiringWitness for Arithmetic {
    type Domain = i64;
    const ZERO: i64 = 0;
    const ONE: i64 = 1;

    #[inline]
    fn add(a: i64, b: i64) -> i64 {
        a.saturating_add(b)
    }

    #[inline]
    fn mul(a: i64, b: i64) -> i64 {
        a.saturating_mul(b)
    }
}

/// **Arithmetic modulo `P`**: `⊕` and `⊗` are addition and multiplication in `ℤ/Pℤ`.
///
/// The witness counterpart of [`Zn`](crate::Zn)/[`Zp`](crate::Zp), and not a duplicate of them:
/// those are newtype *carriers* that wrap a value, where this leaves the domain as a plain `i64`
/// so it can host modular arithmetic alongside other structures. Same mathematics, two
/// mechanisms — pick the carrier when the type should enforce the modulus, the witness when one
/// domain needs several algebras.
///
/// Intermediate products are widened to `i128` before reduction, so a product that would
/// overflow `i64` still reduces correctly. `rem_euclid` keeps the representative non-negative.
pub struct ModP<const P: i64>;

impl<const P: i64> SemiringWitness for ModP<P> {
    type Domain = i64;
    const ZERO: i64 = 0;
    const ONE: i64 = 1;

    #[inline]
    fn add(a: i64, b: i64) -> i64 {
        ((i128::from(a) + i128::from(b)).rem_euclid(i128::from(P))) as i64
    }

    #[inline]
    fn mul(a: i64, b: i64) -> i64 {
        ((i128::from(a) * i128::from(b)).rem_euclid(i128::from(P))) as i64
    }
}
