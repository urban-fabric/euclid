//! That the conformance suites **reject** a carrier that breaks their laws.
//!
//! # Why this file exists
//!
//! Every other test of `conformance` runs a law suite against a *correct* carrier and checks
//! that it does not complain. That is necessary and it is not sufficient, and mutation testing
//! showed exactly how insufficient: **eighteen of this crate's law suites could be replaced with
//! an empty body** and the entire workspace suite stayed green. A law that asserts nothing
//! passes every correct implementation too.
//!
//! For a crate whose deliverable *is* its conformance suites — a consumer implements the trait
//! tower, runs `semiring_laws_for::<Theirs>()`, and is told whether their implementation is
//! sound — a vacuous law is worse than a missing one. It reports success.
//!
//! So each test below builds a carrier that is deliberately wrong in one specific way, claims
//! the marker trait it violates, and requires the matching suite to **panic**. A suite that
//! asserts nothing fails these tests; a suite that asserts the wrong thing fails them too,
//! because each carrier is wrong in exactly one law and correct in all the others.
//!
//! # The markers are what make this cheap
//!
//! The tower derives its structures from marker traits — `AssociativeMultiplication`,
//! `Distributive`, and so on — which a carrier *claims* rather than proves. A broken carrier is
//! therefore a handful of one-line impls that claim a marker they do not satisfy, which is also
//! a fair description of the mistake a real consumer makes.

// A test asserts by panicking, and these tests assert on a panic *message*, which `expect` is
// the clearest way to extract. Same exemption the other suites in the fleet take.
#![allow(clippy::expect_used)]

use std::panic::{self, AssertUnwindSafe};

use crate::conformance::{
    self, ConformanceSample, commutative_semiring_laws_for, ring_laws_for, semiring_laws_for,
};
use crate::*;

/// Run `body` and report whether it panicked, with the panic message.
///
/// The panic hook is silenced for the duration, because a law suite reports by panicking and
/// these tests expect it — the default hook would print a backtrace per case.
fn rejects(body: impl FnOnce()) -> Option<String> {
    let previous = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let outcome = panic::catch_unwind(AssertUnwindSafe(body));
    panic::set_hook(previous);

    match outcome {
        Ok(()) => None,
        Err(payload) => Some(
            payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "non-string panic".to_string()),
        ),
    }
}

// ===== A carrier whose multiplicative identity is not an identity ==========

/// `a ⊗ b = ab + 1`, which has no identity and is not associative, while claiming both.
#[derive(Clone, Copy, PartialEq, Debug)]
struct NoMulIdentity(i64);

impl AdditiveMagma for NoMulIdentity {
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}
impl MultiplicativeMagma for NoMulIdentity {
    fn mul(self, rhs: Self) -> Self {
        Self(self.0.wrapping_mul(rhs.0).wrapping_add(1))
    }
}
impl HasZero for NoMulIdentity {
    const ZERO: Self = Self(0);
}
impl HasOne for NoMulIdentity {
    const ONE: Self = Self(1);
}
impl AssociativeAddition for NoMulIdentity {}
impl CommutativeAddition for NoMulIdentity {}
impl AssociativeMultiplication for NoMulIdentity {}
impl CommutativeMultiplication for NoMulIdentity {}
impl Distributive for NoMulIdentity {}
impl ZeroAnnihilatesMultiplication for NoMulIdentity {}
impl ConformanceSample for NoMulIdentity {
    fn sample() -> Vec<Self> {
        vec![Self(0), Self(1), Self(2), Self(-3)]
    }
}

/// The eight semiring axioms must reject a carrier whose `⊗` has no identity.
#[test]
fn the_semiring_laws_reject_a_broken_multiplicative_identity() {
    let message = rejects(|| semiring_laws_for::<NoMulIdentity>("NoMulIdentity"))
        .expect("semiring_laws must reject a carrier where a ⊗ 1 ≠ a");
    assert!(
        message.contains("⊗ 1 = a") || message.contains("1 ⊗ a"),
        "the failure must name the axiom that broke: {message}"
    );
}

/// And it must still *accept* a correct carrier, or the suite is simply always-fail.
#[test]
fn the_semiring_laws_accept_a_correct_carrier() {
    assert!(
        rejects(|| semiring_laws_for::<Rational<i64>>("Rational")).is_none(),
        "semiring_laws must accept a correct carrier"
    );
}

// ===== A correct semiring that does not commute ============================

/// 2x2 matrices over `i64`, in row-major order — a genuine non-commutative semiring.
///
/// This one is not *broken*: every semiring axiom holds. It claims
/// [`CommutativeMultiplication`], which it does not have, so it separates
/// `commutative_semiring_laws` from the `semiring_laws` it delegates to. Without a carrier like
/// this, a `commutative_semiring_laws` that forgot to check commutativity would be
/// indistinguishable from one that checks it.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Matrix2([i64; 4]);

impl AdditiveMagma for Matrix2 {
    fn add(self, rhs: Self) -> Self {
        let mut out = self.0;
        for (slot, r) in out.iter_mut().zip(rhs.0) {
            *slot = slot.wrapping_add(r);
        }
        Self(out)
    }
}
impl MultiplicativeMagma for Matrix2 {
    fn mul(self, rhs: Self) -> Self {
        let [a, b, c, d] = self.0;
        let [e, f, g, h] = rhs.0;
        Self([a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h])
    }
}
impl HasZero for Matrix2 {
    const ZERO: Self = Self([0, 0, 0, 0]);
}
impl HasOne for Matrix2 {
    const ONE: Self = Self([1, 0, 0, 1]);
}
impl AssociativeAddition for Matrix2 {}
impl CommutativeAddition for Matrix2 {}
impl AssociativeMultiplication for Matrix2 {}
impl Distributive for Matrix2 {}
impl ZeroAnnihilatesMultiplication for Matrix2 {}
// The false claim.
impl CommutativeMultiplication for Matrix2 {}
impl ConformanceSample for Matrix2 {
    fn sample() -> Vec<Self> {
        vec![
            Self([0, 0, 0, 0]),
            Self([1, 0, 0, 1]),
            Self([1, 1, 0, 1]),
            Self([0, 1, 1, 0]),
        ]
    }
}

/// `commutative_semiring_laws` must check the thing its name adds.
#[test]
fn the_commutative_semiring_laws_reject_a_non_commutative_product() {
    // The semiring axioms themselves hold for 2x2 matrices, so the rejection below can only
    // come from the commutativity check.
    assert!(
        rejects(|| conformance::semiring_laws(&Matrix2::sample(), "Matrix2")).is_none(),
        "precondition: 2x2 matrices satisfy every semiring axiom"
    );

    let message = rejects(|| commutative_semiring_laws_for::<Matrix2>("Matrix2"))
        .expect("commutative_semiring_laws must reject a non-commutative product");
    assert!(
        message.contains("b ⊗ a"),
        "the failure must name multiplicative commutativity: {message}"
    );
}

// ===== A ring whose operator bridge is broken ==============================

/// A ring in which `AdditiveMagma::add` and `+` disagree.
///
/// The crate's own words for why this matters: "a formula written against the trunk and one
/// written with operators compute different things on the same type, which is precisely the
/// failure the library exists to prevent." It is also the easiest mistake to make, because the
/// two are written in different impl blocks.
#[derive(Clone, Copy, PartialEq, Debug)]
struct BrokenBridge(i64);

impl AdditiveMagma for BrokenBridge {
    /// Disagrees with `Add` below by one.
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0).wrapping_add(1))
    }
}
impl MultiplicativeMagma for BrokenBridge {
    fn mul(self, rhs: Self) -> Self {
        Self(self.0.wrapping_mul(rhs.0))
    }
}
impl HasZero for BrokenBridge {
    const ZERO: Self = Self(0);
}
impl HasOne for BrokenBridge {
    const ONE: Self = Self(1);
}
impl AssociativeAddition for BrokenBridge {}
impl CommutativeAddition for BrokenBridge {}
impl AssociativeMultiplication for BrokenBridge {}
impl CommutativeMultiplication for BrokenBridge {}
impl Distributive for BrokenBridge {}
impl ZeroAnnihilatesMultiplication for BrokenBridge {}

impl core::ops::Add for BrokenBridge {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}
impl core::ops::Sub for BrokenBridge {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
}
impl core::ops::Mul for BrokenBridge {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self(self.0.wrapping_mul(rhs.0))
    }
}
impl core::ops::Neg for BrokenBridge {
    type Output = Self;
    fn neg(self) -> Self {
        Self(self.0.wrapping_neg())
    }
}
impl core::ops::AddAssign for BrokenBridge {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl core::ops::SubAssign for BrokenBridge {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl core::ops::MulAssign for BrokenBridge {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl ConformanceSample for BrokenBridge {
    fn sample() -> Vec<Self> {
        vec![Self(0), Self(1), Self(2), Self(-3)]
    }
}

/// `ring_laws` must catch the trunk and the operators disagreeing.
#[test]
fn the_ring_laws_reject_a_broken_operator_bridge() {
    let message = rejects(|| ring_laws_for::<BrokenBridge>("BrokenBridge"))
        .expect("ring_laws must reject a carrier whose ⊕ and + disagree");
    assert!(
        message.contains("⊕ is +"),
        "the failure must name the bridge law: {message}"
    );
}

// ===== What is not covered here yet =======================================
//
// `field_laws`, `euclidean_laws`, `dioid_laws`, `star_laws`, `canonical_order_laws`,
// `ordered_semiring_laws`, `left_semimodule_laws` and the two `witness_*` suites are still
// vacuity-untested, and each is still deletable under `cargo mutants` for the reason given in
// this module's header.
//
// The pattern above is the template for them, and the cost is not uniform: the three suites
// covered here sit on the semiring trunk, whose laws are *marker traits*, so a broken carrier is
// a dozen one-line impls. The tiers above need more. `Field` is `EuclideanDomain + Div + DivAssign`
// and `EuclideanDomain` is `Pid` with a `div_rem`, so a deliberately-wrong field has to delegate
// the whole tower before it can break one operation in it — which is mechanical, but it is a
// newtype with twenty impls rather than ten lines.
//
// Worth doing, and worth doing as its own change rather than wedged into this one.

// ===== Surfaces nothing was exercising ====================================

/// `f32`'s place in the tower, which no suite was instantiating.
///
/// Mutation testing found `<f32 as EuclideanDomain>::div_rem` and `<f32 as Field>::conjugate`
/// both replaceable with `Default::default()` — returning `(0.0, 0.0)` and `0.0` — with the whole
/// workspace green. The reason is simple and worth stating: the carrier list in
/// [`laws`](super::laws) checks `i64`, `Rational64`, `Zp`, `Zn` and `GaussianRational`, and the
/// float impls ride along only through `f64`. `f32` has its own impl block and nothing reached
/// it.
///
/// For a field the remainder is always zero, which is what makes `div_rem` on a float look
/// trivial enough to skip.
///
/// # The sample is all powers of two, and it has to be
///
/// Writing this test turned up something worth recording: **no float carrier is run through
/// `field_laws` anywhere in the fleet**, and a first attempt with the sample
/// `[0.0, 1.0, -1.0, 2.5, -0.75]` failed on `b / a = b · a⁻¹` — `-3.3333333` against
/// `-3.3333335`. That is not a defect in `f32`'s impl. The law is stated with exact equality,
/// and float division is not exactly multiplication by the reciprocal, nor is float addition
/// associative.
///
/// So a float carrier can only satisfy these suites over a sample closed under *exact*
/// arithmetic. Every value here is a small power of two, which makes each sum, product,
/// quotient and reciprocal in play exactly representable. A consumer bringing a float carrier to
/// `semiring_laws` or `field_laws` needs to know that, and now there is a test that says so.
#[test]
fn f32_satisfies_the_tower_it_claims() {
    // Powers of two only — see the note above on exact equality.
    let sample = [0.0_f32, 1.0, -1.0, 2.0, -0.5];

    conformance::semiring_laws(&sample, "f32");
    conformance::ring_laws(&sample, "f32");
    conformance::field_laws(&sample, "f32");
    conformance::euclidean_laws(&sample, "f32");
    conformance::ref_magma_laws(&sample, "f32");

    // And the two operations spelled out, since a field's remainder being zero is the thing a
    // `Default::default()` body accidentally gets right.
    for &a in &sample {
        assert_eq!(Field::conjugate(a), a, "a real field conjugates to itself");
        for &b in &sample {
            if b != 0.0 {
                let (q, r) = EuclideanDomain::div_rem(a, b);
                assert_eq!(q, a / b, "f32: div_rem's quotient is a / b");
                assert_eq!(r, 0.0, "f32: a field leaves no remainder");
                assert_eq!(q * b + r, a, "f32: a = qb + r");
            }
        }
    }
}

/// `Integer::abs` and `Integer::signum`, which `gcd` depends on and no test pinned.
///
/// `gcd` is implemented as `gcd_abs(self.abs(), other.abs(), 0)`, so a broken `abs` is a broken
/// `gcd` for every negative input — and mutation testing found `abs` replaceable with
/// `Default::default()`, returning zero. The existing `gcd` tests evidently use non-negative
/// arguments, which is exactly the shape of input that cannot tell the two apart.
#[test]
fn the_integer_helpers_handle_sign() {
    for (x, want) in [(0_i64, 0), (7, 7), (-7, 7), (i64::MAX, i64::MAX)] {
        assert_eq!(Integer::abs(x), want, "abs({x})");
    }
    for (x, want) in [(0_i64, 0), (7, 1), (-7, -1)] {
        assert_eq!(Integer::signum(x), want, "signum({x})");
    }
    // `from_i64` is the way every other carrier is built from a literal, and it could be made to
    // return zero for every input.
    for x in [0_i64, 1, -1, 7, -7, i64::MAX, i64::MIN] {
        assert_eq!(<i64 as Integer>::from_i64(x), x, "from_i64({x})");
    }

    // And `gcd` therefore works through the sign, in both operands and both orders.
    assert_eq!(Integer::gcd(12_i64, 18), 6);
    assert_eq!(Integer::gcd(-12_i64, 18), 6, "gcd must ignore sign");
    assert_eq!(Integer::gcd(12_i64, -18), 6);
    assert_eq!(Integer::gcd(-12_i64, -18), 6);
    assert_eq!(Integer::gcd(0_i64, -5), 5, "gcd(0, n) = |n|");
}

/// `Rational`'s ordering, including the comparison `PartialOrd` delegates.
///
/// `partial_cmp` could be made to return `None` for every pair — "these are incomparable" — and
/// nothing noticed, because `Rational` is only ever compared through `==` in the suites. A
/// rational that cannot be ordered silently breaks any sort, `min`/`max`, or binary search over
/// it.
#[test]
fn rationals_are_totally_ordered_through_partial_cmp() {
    use core::cmp::Ordering;

    let third = Rational::<i64>::new(1, 3);
    let half = Rational::<i64>::new(1, 2);
    let neg_half = Rational::<i64>::new(-1, 2);

    assert_eq!(third.partial_cmp(&half), Some(Ordering::Less));
    assert_eq!(half.partial_cmp(&third), Some(Ordering::Greater));
    assert_eq!(half.partial_cmp(&half), Some(Ordering::Equal));
    assert_eq!(neg_half.partial_cmp(&third), Some(Ordering::Less));

    // Never `None`: a rational is comparable to every other rational.
    for a in Rational::<i64>::sample() {
        for b in Rational::<i64>::sample() {
            assert!(
                a.partial_cmp(&b).is_some(),
                "{a:?} and {b:?} must be comparable"
            );
        }
    }

    // And the operators agree with it, which is what a consumer actually writes.
    assert!(third < half);
    assert!(neg_half < third);
    assert!(half >= half);
}

/// The `Display` impls, which are part of the published surface and were unexercised.
///
/// All three could be replaced with a body that writes nothing and returns `Ok(())`. A carrier
/// whose `Display` is empty is a diagnostic that says nothing — and these are the types that
/// appear in a conformance failure message, so the loss lands exactly where it hurts.
#[test]
fn the_carriers_display_their_values() {
    assert_eq!(Zp::<7>::new(3).to_string(), "3 (mod 7)");
    assert_eq!(Zn::<6>::new(4).to_string(), "4 (mod 6)");
    assert_eq!(
        GaussianRational::new(Rational::new(1, 2), Rational::new(-3, 4)).to_string(),
        "1/2 + -3/4i"
    );

    // Non-empty for every sampled value, so a future carrier cannot quietly render to nothing.
    for z in GaussianRational::sample() {
        assert!(!z.to_string().is_empty(), "{z:?} rendered to nothing");
    }
}

// ===== The ref-magma suite, checked against itself ========================

/// A carrier whose `add_ref` disagrees with its `add`.
///
/// [`conformance::ref_magma_laws`] was added in response to a mutation finding, so it owes the
/// same debt every other suite here owes: a suite nobody has seen fail is a suite nobody has
/// tested. Without this, `ref_magma_laws` could itself be replaced with an empty body — which
/// would restore exactly the hole it was written to close.
#[derive(Clone, Copy, PartialEq, Debug)]
struct DivergentRef(i64);

impl AdditiveMagma for DivergentRef {
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}
impl MultiplicativeMagma for DivergentRef {
    fn mul(self, rhs: Self) -> Self {
        Self(self.0.wrapping_mul(rhs.0))
    }
}
impl RefAdditiveMagma for DivergentRef {
    /// Disagrees with `add` by one — the shape of a copy-paste slip between two impl blocks.
    fn add_ref(&self, rhs: &Self) -> Self {
        Self(self.0.wrapping_add(rhs.0).wrapping_add(1))
    }
}
impl RefMultiplicativeMagma for DivergentRef {
    fn mul_ref(&self, rhs: &Self) -> Self {
        Self(self.0.wrapping_mul(rhs.0))
    }
}
impl ConformanceSample for DivergentRef {
    fn sample() -> Vec<Self> {
        vec![Self(0), Self(1), Self(2), Self(-3)]
    }
}

#[test]
fn the_ref_magma_laws_reject_a_divergent_by_reference_operation() {
    // Through the `_for` wrapper, so the sample-reading delegation is pinned as well as the
    // laws it delegates to — the wrappers were dead code until this suite started using them.
    let message = rejects(|| conformance::ref_magma_laws_for::<DivergentRef>("DivergentRef"))
        .expect("ref_magma_laws must reject an add_ref that disagrees with add");
    assert!(
        message.contains("add_ref"),
        "the failure must name the operation that diverged: {message}"
    );
}

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
#[derive(Clone, Copy, PartialEq, Debug)]
struct Faulty<const WHICH: u8>(Rational<i64>);

impl<const W: u8> AdditiveMagma for Faulty<W> {
    fn add(self, rhs: Self) -> Self {
        Self(AdditiveMagma::add(self.0, rhs.0))
    }
}
impl<const W: u8> MultiplicativeMagma for Faulty<W> {
    fn mul(self, rhs: Self) -> Self {
        Self(MultiplicativeMagma::mul(self.0, rhs.0))
    }
}
impl<const W: u8> RefAdditiveMagma for Faulty<W> {
    fn add_ref(&self, rhs: &Self) -> Self {
        Self(self.0.add_ref(&rhs.0))
    }
}
impl<const W: u8> RefMultiplicativeMagma for Faulty<W> {
    fn mul_ref(&self, rhs: &Self) -> Self {
        Self(self.0.mul_ref(&rhs.0))
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
        Self(self.0 * rhs.0)
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

// ===== The dioid tier =====================================================

/// A carrier claiming idempotent addition while adding normally.
///
/// `Dioid` is `IdempotentSemiring`, which is `Semiring + AdditiveSemilattice`, and the whole
/// content of the tier is `a ⊕ a = a`. The suite's own docstring names this as the axiom "a
/// carrier is most likely to claim by accident — the impl is a bare marker, so nothing stops a
/// non-idempotent carrier declaring it", and then nothing checked that the suite would catch
/// one. This is that carrier: ordinary integer addition under an idempotence claim, so
/// `2 ⊕ 2 = 4`.
#[derive(Clone, Copy, PartialEq, Debug)]
struct NotIdempotent(i64);

impl AdditiveMagma for NotIdempotent {
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}
impl MultiplicativeMagma for NotIdempotent {
    fn mul(self, rhs: Self) -> Self {
        Self(self.0.wrapping_mul(rhs.0))
    }
}
impl HasZero for NotIdempotent {
    const ZERO: Self = Self(0);
}
impl HasOne for NotIdempotent {
    const ONE: Self = Self(1);
}
impl AssociativeAddition for NotIdempotent {}
impl CommutativeAddition for NotIdempotent {}
impl AssociativeMultiplication for NotIdempotent {}
impl Distributive for NotIdempotent {}
impl ZeroAnnihilatesMultiplication for NotIdempotent {}
// The false claim: `⊕` is not idempotent for any nonzero value.
impl IdempotentAddition for NotIdempotent {}
impl ConformanceSample for NotIdempotent {
    fn sample() -> Vec<Self> {
        vec![Self(0), Self(1), Self(2), Self(5)]
    }
}

#[test]
fn the_dioid_laws_reject_a_non_idempotent_addition() {
    // The semiring axioms hold, so the rejection can only come from the idempotence check.
    assert!(
        rejects(|| conformance::semiring_laws(&NotIdempotent::sample(), "NotIdempotent")).is_none(),
        "precondition: this carrier is a sound semiring"
    );

    let message = rejects(|| conformance::dioid_laws_for::<NotIdempotent>("NotIdempotent"))
        .expect("dioid_laws must reject a carrier whose ⊕ is not idempotent");
    assert!(
        message.contains("idempotent"),
        "the failure must name the dioid axiom: {message}"
    );
}
