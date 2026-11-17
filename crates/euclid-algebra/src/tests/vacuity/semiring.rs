//! Semiring contracts and rejection controls.

use super::*;

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
