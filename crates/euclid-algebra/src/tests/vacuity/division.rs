//! Division contracts and rejection controls.

use super::*;

// ===== The field and Euclidean tiers ======================================

/// A field that delegates to [`Rational`] except for the one operation `WHICH` selects.
///
/// The tiers above the semiring trunk cannot be broken as cheaply as the trunk can, because their
/// laws are not marker traits: `Field` is `EuclideanDomain + Div + DivAssign` and
/// `EuclideanDomain` is `Pid` with a `div_rem`, so a deliberately-wrong field has to carry the
/// whole tower before it can misbehave in one corner of it.
///
/// One generic wrapper rather than a newtype per defect, so the twenty delegating impls are
/// written once. `Rational<i64>` is the inner carrier because its arithmetic is *exact* — a float
/// would fail these suites for reasons of its own, as `f32_satisfies_the_tower_it_claims`
/// records.
///
/// - `WHICH = 0` — nothing broken; the control that proves the suites still accept a sound carrier
/// - `WHICH = 1` — `Div` returns its numerator, so `a / a ≠ 1`
/// - `WHICH = 2` — `div_rem` leaves a remainder of `ONE`, so `a ≠ qb + r`
/// - `WHICH = 3` — `div_rem` returns `(ZERO, self)`: degenerate, but it *does* reconstruct
/// - `WHICH = 4` — multiplication is `ab + 1`, in **both** layers at once
///
/// `pub(super)` so the generated-suite controls in [`super::generated_suites`] can reuse the
/// tower rather than build a second one.
///
/// `WHICH = 4` is broken in `MultiplicativeMagma::mul`, `RefMultiplicativeMagma::mul_ref` and
/// `core::ops::Mul` together, which is deliberate. The crate requires the element layer and the
/// operator layer to agree — `ring_laws` asserts `⊗ is *` directly — so breaking only one of
/// them would be caught as a *bridge* defect and would never reach the axioms underneath.
/// Breaking all three consistently makes it a wrong multiplication rather than an inconsistent
/// one, which is what lets a single carrier be rejected by all five tiers: `a ⊗ 1 = a` at the
/// trunk, `−(ab) = (−a)b` at the ring, `a · a⁻¹ = 1` at the field, and `a = qb + r` at the
/// Euclidean tier each fail on their own.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) struct Faulty<const WHICH: u8>(pub(super) Rational<i64>);

impl<const W: u8> AdditiveMagma for Faulty<W> {
    fn add(self, rhs: Self) -> Self {
        Self(AdditiveMagma::add(self.0, rhs.0))
    }
}
impl<const W: u8> MultiplicativeMagma for Faulty<W> {
    fn mul(self, rhs: Self) -> Self {
        let product = MultiplicativeMagma::mul(self.0, rhs.0);
        if W == 4 {
            Self(AdditiveMagma::add(product, Rational::ONE))
        } else {
            Self(product)
        }
    }
}
impl<const W: u8> RefAdditiveMagma for Faulty<W> {
    fn add_ref(&self, rhs: &Self) -> Self {
        Self(self.0.add_ref(&rhs.0))
    }
}
impl<const W: u8> RefMultiplicativeMagma for Faulty<W> {
    fn mul_ref(&self, rhs: &Self) -> Self {
        MultiplicativeMagma::mul(*self, *rhs)
    }
}
impl<const W: u8> HasZero for Faulty<W> {
    const ZERO: Self = Self(Rational::ZERO);
}
impl<const W: u8> HasOne for Faulty<W> {
    const ONE: Self = Self(Rational::ONE);
}
impl<const W: u8> AssociativeAddition for Faulty<W> {}
impl<const W: u8> CommutativeAddition for Faulty<W> {}
impl<const W: u8> AssociativeMultiplication for Faulty<W> {}
impl<const W: u8> CommutativeMultiplication for Faulty<W> {}
impl<const W: u8> Distributive for Faulty<W> {}
impl<const W: u8> ZeroAnnihilatesMultiplication for Faulty<W> {}
impl<const W: u8> IntegralDomain for Faulty<W> {}
impl<const W: u8> Ufd for Faulty<W> {}
impl<const W: u8> Pid for Faulty<W> {}

impl<const W: u8> core::ops::Add for Faulty<W> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}
impl<const W: u8> core::ops::Sub for Faulty<W> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}
impl<const W: u8> core::ops::Mul for Faulty<W> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        if W == 4 {
            Self(self.0 * rhs.0 + Rational::ONE) // the defect, matching the element layer
        } else {
            Self(self.0 * rhs.0)
        }
    }
}
impl<const W: u8> core::ops::Neg for Faulty<W> {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}
impl<const W: u8> core::ops::Div for Faulty<W> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        if W == 1 {
            self // the defect: division is not inversion
        } else {
            Self(self.0 / rhs.0)
        }
    }
}
impl<const W: u8> core::ops::AddAssign for Faulty<W> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl<const W: u8> core::ops::SubAssign for Faulty<W> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl<const W: u8> core::ops::MulAssign for Faulty<W> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl<const W: u8> core::ops::DivAssign for Faulty<W> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
impl<const W: u8> EuclideanDomain for Faulty<W> {
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        if W == 2 {
            // The defect: the right quotient with a remainder that does not belong to it.
            (self / rhs, Self::ONE)
        } else if W == 3 {
            // Degenerate but reconstructing: a = 0·b + a. See the test for why it is accepted.
            (Self::ZERO, self)
        } else {
            (self / rhs, Self::ZERO)
        }
    }
}
impl<const W: u8> Field for Faulty<W> {
    fn conjugate(self) -> Self {
        self
    }
}
impl<const W: u8> ConformanceSample for Faulty<W> {
    fn sample() -> Vec<Self> {
        Rational::<i64>::sample().into_iter().map(Self).collect()
    }
}

/// The control: a sound carrier passes every tier it claims.
///
/// Without this the three tests below could be satisfied by suites that reject *everything*,
/// which is the other way a law suite can be useless.
#[test]
fn the_control_carrier_passes_every_tier() {
    semiring_laws_for::<Faulty<0>>("Faulty<0>");
    commutative_semiring_laws_for::<Faulty<0>>("Faulty<0>");
    ring_laws_for::<Faulty<0>>("Faulty<0>");
    conformance::field_laws_for::<Faulty<0>>("Faulty<0>");
    conformance::euclidean_laws_for::<Faulty<0>>("Faulty<0>");
    conformance::ref_magma_laws_for::<Faulty<0>>("Faulty<0>");
}

/// `field_laws` must reject a carrier whose division is not inversion.
#[test]
fn the_field_laws_reject_a_division_that_is_not_inversion() {
    let message = rejects(|| conformance::field_laws_for::<Faulty<1>>("Faulty<1>"))
        .expect("field_laws must reject a carrier where a / a ≠ 1");
    assert!(
        message.contains("a⁻¹") || message.contains("a / a"),
        "the failure must name the field axiom: {message}"
    );
}

/// `euclidean_laws` must reject a `div_rem` that does not reconstruct its input.
///
/// The division-with-remainder identity `a = qb + r` is the whole content of the tier, and
/// `div_rem` is the easiest place in the crate to be lazily wrong — for a field the remainder is
/// always zero, so an implementation that returns a zero remainder *looks* right whatever it does
/// with the quotient.
/// # What this law cannot reject, which is worth knowing
///
/// The first attempt at this test used `div_rem(a, b) = (ZERO, a)` as the defect and **passed** —
/// because `a = 0·b + a` reconstructs perfectly well. "Quotient zero, remainder everything"
/// satisfies the only law the tier states, while being a `div_rem` no caller could use.
///
/// That is not a hole in the test; it is the honest reach of the law. A Euclidean domain also
/// requires the remainder to be *smaller* than the divisor under some norm, and
/// [`EuclideanDomain`] exposes no norm to state that against — its docstring deliberately settles
/// for reconstruction because that "holds regardless of how a carrier rounds the quotient". So
/// passing `euclidean_laws` does not establish that a remainder shrinks, only that the quotient
/// and remainder agree with the division. The `Faulty<3>` assertion below pins that reach so it
/// is a known property rather than a surprise.
#[test]
fn the_euclidean_laws_reject_a_div_rem_that_does_not_reconstruct() {
    let message = rejects(|| conformance::euclidean_laws_for::<Faulty<2>>("Faulty<2>"))
        .expect("euclidean_laws must reject a remainder that does not belong to its quotient");
    assert!(
        message.contains("a mod b"),
        "the failure must name the reconstruction law: {message}"
    );

    // The degenerate case the law *admits*, recorded as behaviour rather than left to surprise
    // someone: quotient zero and remainder everything reconstructs, so it is accepted.
    assert!(
        rejects(|| conformance::euclidean_laws_for::<Faulty<3>>("Faulty<3>")).is_none(),
        "a (ZERO, self) div_rem satisfies reconstruction, so this law accepts it"
    );
}
