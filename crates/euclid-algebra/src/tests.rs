//! The crate's tests, split by what they establish.
//!
//! - [`laws`] — the tier axioms themselves, run over a fixed sample of every carrier. A
//!   carrier "is a `Field`" exactly when it satisfies the field laws, so that suite is the
//!   definition rather than a sampling of it.
//! - [`generated_laws`] — the same laws over *generated* carriers, hunting the representation
//!   invariants underneath them: that `Rational` stays reduced through arithmetic, that `Zp`'s
//!   Fermat inverse agrees with division at every residue. Shares the law definitions with
//!   [`laws`] and changes only where the sample comes from.
//! - [`witnesses`] — the same, for the witnessed semirings, where the structure is a type
//!   parameter rather than the type. These matter more than their element-layer counterparts:
//!   a witness carries no property markers, so the laws are the only thing between it and a
//!   wrong algebra.
//! - [`rational`] — exact ordering at the limits of the backing integers and division by zero.
//! - [`vacuity`] — that the law suites **reject** a carrier that breaks them. Every other test
//!   here runs a suite against a correct carrier, which cannot distinguish a law that holds from
//!   a law that is never checked.
//! - This file — that the *tiers* are reachable: a formula written once at a tier compiles and
//!   runs for the carriers that claim it, and the blanket impls fire as intended.

mod generated_laws;
mod laws;
mod rational;
mod vacuity;
mod witnesses;

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
