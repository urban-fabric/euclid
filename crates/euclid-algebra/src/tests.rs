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
