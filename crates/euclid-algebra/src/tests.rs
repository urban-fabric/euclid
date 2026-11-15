//! The crate's tests, split by what they establish.
//!
//! - [`laws`] — the tier axioms themselves, run over a fixed sample of every carrier. A
//!   carrier "is a `Field`" exactly when it satisfies the field laws, so that suite is the
//!   definition rather than a sampling of it.
//! - This file — that the *tiers* are reachable: a formula written once at a tier compiles and
//!   runs for the carriers that claim it, and the blanket impls fire as intended.

mod laws;

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
