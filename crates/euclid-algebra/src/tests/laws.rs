//! Every carrier this crate ships, checked against the tier it claims.
//!
//! The laws themselves live in [`crate::conformance`], which is public API — a backend or a
//! consumer-defined carrier calls the same functions. This file is the proof that the suites
//! are satisfiable and that *our* carriers satisfy them, so it deliberately adds nothing the
//! laws do not already say.
//!
//! Each sample is chosen to stress the implementation rather than to be large: `ZERO`, `ONE`, a
//! negative, and for the modular carriers the residues adjacent to the modulus, where a missed
//! reduction shows. Laws needing three operands run over every triple, so six values give 216
//! cases per law.
//!
//! Beyond the tiers, three carrier-specific invariants are checked here because they are
//! properties of these types rather than of any tier: that `Rational` stays reduced after
//! arithmetic, that `Zp`'s Fermat inverse agrees with `ONE / a`, and that `Z/6` has the
//! zero-divisor pair which is why `Zn` must not reach `IntegralDomain`.

use super::*;
use crate::conformance::{self as laws, ConformanceSample};

// Each carrier declares the values that stress *it*, once, and every suite draws from here.
// Adding a suite to `conformance` then covers all of these without touching this file.

impl ConformanceSample for i64 {
    /// `0`, `1`, a negative, and values that are not round numbers — enough to catch a dropped
    /// sign or a mis-ordered operand.
    fn sample() -> Vec<Self> {
        vec![0, 1, -1, 7, -3, 12]
    }
}

impl ConformanceSample for Rational64 {
    fn sample() -> Vec<Self> {
        vec![
            Self::integer(0),
            Self::integer(1),
            Self::new(-1, 2),
            Self::new(3, 4),
            Self::new(-7, 3),
            Self::new(5, 1),
        ]
    }
}

impl ConformanceSample for Zp<7> {
    /// Residues around the modulus: `0`, `1`, the largest, a middling one, and two that need
    /// reducing on the way in — from both directions.
    fn sample() -> Vec<Self> {
        vec![
            Self::new(0),
            Self::new(1),
            Self::new(3),
            Self::new(6),
            Self::from_i64(-1),
            Self::new(9),
        ]
    }
}

impl ConformanceSample for Zn<6> {
    fn sample() -> Vec<Self> {
        vec![
            Self::new(0),
            Self::new(1),
            Self::new(2),
            Self::new(3),
            Self::new(5),
            Self::from_i64(-1),
        ]
    }
}

impl ConformanceSample for GaussianRational {
    fn sample() -> Vec<Self> {
        let q = Rational128::new;
        vec![
            Self::integer(0),
            Self::integer(1),
            Self::new(q(1, 2), q(1, 2)),
            Self::new(q(-3, 4), q(5, 4)),
            Self::new(q(0, 1), q(1, 1)), // i
            Self::new(q(2, 1), q(-1, 1)),
        ]
    }
}

impl ConformanceSample for f64 {
    /// Dyadic rationals only: exactly representable, so the field laws hold without a
    /// tolerance. `f64`'s inexactness is euclid-numeric's subject, not the tower's.
    fn sample() -> Vec<Self> {
        vec![0.0, 1.0, -1.0, 0.5, -2.5, 4.0]
    }
}

impl ConformanceSample for f32 {
    fn sample() -> Vec<Self> {
        vec![0.0, 1.0, -1.0, 0.5, -2.5, 4.0]
    }
}

#[test]
fn integers_satisfy_semiring_ring_and_euclidean_laws() {
    laws::commutative_semiring_laws_for::<i64>("i64");
    laws::ring_laws_for::<i64>("i64");
    laws::euclidean_laws_for::<i64>("i64");
}

#[test]
fn rationals_satisfy_the_field_laws() {
    laws::commutative_semiring_laws_for::<Rational64>("Rational64");
    laws::ring_laws_for::<Rational64>("Rational64");
    laws::field_laws_for::<Rational64>("Rational64");
    laws::euclidean_laws_for::<Rational64>("Rational64");
}

#[test]
fn rationals_are_always_in_lowest_terms() {
    // The invariant the type promises: reduced, with the sign on the numerator. It has to hold
    // after *arithmetic*, not just after `new`, which is where a missing `reduce` would show.
    fn gcd(mut a: i64, mut b: i64) -> i64 {
        a = a.abs();
        b = b.abs();
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }
    let xs = [
        Rational64::new(6, 4),
        Rational64::new(-10, 5),
        Rational64::new(7, -3),
        Rational64::integer(0),
    ];
    for &a in &xs {
        for &b in &xs {
            let mut results = vec![a + b, a - b, a * b];
            if b != Rational64::integer(0) {
                results.push(a / b);
            }
            for c in results {
                assert!(c.denominator() > 0, "denominator stays positive: {c:?}");
                let g = gcd(c.numerator(), c.denominator());
                assert_eq!(g, 1, "in lowest terms: {c:?} has gcd {g}");
            }
        }
    }
    // Zero normalizes to exactly 0/1 rather than 0/n.
    assert_eq!(Rational64::new(0, 5).denominator(), 1);
}

#[test]
fn zp_is_a_field_for_a_prime_modulus() {
    // Residues around the modulus: 0, 1, the largest residue, and a middling one. 7 is prime,
    // so every nonzero residue must invert.
    laws::commutative_semiring_laws_for::<Zp<7>>("Zp<7>");
    laws::ring_laws_for::<Zp<7>>("Zp<7>");
    laws::field_laws_for::<Zp<7>>("Zp<7>");

    // Reduction happens on construction, from both directions.
    assert_eq!(Zp::<7>::new(9).value(), 2);
    assert_eq!(Zp::<7>::from_i64(-1).value(), 6);

    // The dedicated Fermat `inverse` must agree with the general `ONE / a`.
    for a in <Zp<7> as ConformanceSample>::sample() {
        if a != Zp::<7>::ZERO {
            assert_eq!(a.inverse(), Zp::<7>::ONE / a, "inverse agrees with 1/a");
            assert_eq!(a * a.inverse(), Zp::<7>::ONE);
        }
    }
}

#[test]
fn zn_satisfies_the_ring_laws_for_a_composite_modulus() {
    laws::commutative_semiring_laws_for::<Zn<6>>("Zn<6>");
    laws::ring_laws_for::<Zn<6>>("Zn<6>");
    assert_eq!(Zn::<6>::from_i64(-1).value(), 5);
}

#[test]
fn zn_composite_has_zero_divisors() {
    // 2 · 3 = 0 in Z/6 with neither factor zero, so Z/6 is not an integral domain and must not
    // reach `IntegralDomain`/`Field`. The *absence* of those impls is what enforces it; this is
    // the witness that the absence is correct rather than an oversight.
    let two = Zn::<6>::new(2);
    let three = Zn::<6>::new(3);
    assert_ne!(two, Zn::<6>::ZERO);
    assert_ne!(three, Zn::<6>::ZERO);
    assert_eq!(two * three, Zn::<6>::ZERO, "2·3 = 0 in Z/6");

    // The same product is nonzero where the modulus is prime, which is the distinction the
    // tier split exists to record.
    assert_ne!(Zp::<7>::new(2) * Zp::<7>::new(3), Zp::<7>::ZERO);
}

#[test]
fn gaussian_rationals_satisfy_the_field_laws() {
    laws::commutative_semiring_laws_for::<GaussianRational>("GaussianRational");
    laws::ring_laws_for::<GaussianRational>("GaussianRational");
    // The Euclidean tier too: `div_rem` is the one operation a field's zero remainder makes
    // look trivial, and mutation testing found its quotient unpinned.
    laws::euclidean_laws_for::<GaussianRational>("GaussianRational");
    laws::field_laws_for::<GaussianRational>("GaussianRational");
    laws::field_laws_for::<GaussianRational>("GaussianRational");
}

#[test]
fn gaussian_i_squared_is_minus_one() {
    // The one fact that makes it ℚ(i) rather than ℚ×ℚ, and the one a wrong `mul` would break.
    let i = GaussianRational::new(Rational128::integer(0), Rational128::integer(1));
    assert_eq!(i * i, -GaussianRational::integer(1));

    // The norm is multiplicative — `|zw| = |z||w|` — which is what makes it a Euclidean domain.
    let z = GaussianRational::new(Rational128::new(1, 2), Rational128::new(3, 2));
    let w = GaussianRational::new(Rational128::integer(2), Rational128::integer(-1));
    assert_eq!(
        (z * w).norm(),
        z.norm() * w.norm(),
        "norm is multiplicative"
    );
}

#[test]
fn floats_satisfy_the_field_laws_on_exact_binary_values() {
    // Dyadic rationals only: these are represented exactly, so the field laws hold without a
    // tolerance. The inexact behaviour of `f64` is euclid-numeric's subject, not this crate's —
    // here the question is only whether the tower's axioms are wired up correctly.
    laws::commutative_semiring_laws_for::<f64>("f64");
    laws::ring_laws_for::<f64>("f64");
    laws::field_laws_for::<f64>("f64");

    laws::commutative_semiring_laws_for::<f32>("f32");
    laws::ring_laws_for::<f32>("f32");
    laws::field_laws_for::<f32>("f32");
}

/// A minimal idempotent dioid (boolean reachability), exercising the dioid, star and canonical
/// order suites against a carrier whose `⊕` is **not** `+` — which is the case the trunk's
/// method-based design exists to admit.
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
// `∧` is monotone for the canonical order: `a ⪯ b` means `b → a`, and `b ∧ c → a ∧ c`
// follows, so conjunction cannot reorder two ordered values. That makes Reach an
// `OrderedSemiring` as well as a `Dioid`.
impl MonotoneMultiplication for Reach {}
impl Absorptive for Reach {}

#[test]
fn boolean_dioid_satisfies_the_dioid_star_and_order_laws() {
    let xs = [Reach(false), Reach(true)];
    laws::dioid_laws(&xs, "Reach");
    laws::star_laws(&xs, "Reach");
    laws::canonical_order_laws(&xs, "Reach");

    // Absorptive => star collapses to ONE, which the star law above confirms is consistent.
    assert_eq!(Reach(false).star(), Reach::ONE);
    // Canonical order: reachable (true) precedes unreachable (false).
    assert!(Reach(true).precedes(Reach(false)));
    assert!(!Reach(false).precedes(Reach(true)));
}

#[test]
fn every_semiring_is_a_semimodule_over_itself() {
    // The blanket impl, checked on three carriers with different `⊕`: ordinary addition, the
    // boolean `∨`, and modular addition. `scale` is `⊗` in each case.
    laws::left_semimodule_laws(
        &<i64 as ConformanceSample>::sample(),
        &<i64 as ConformanceSample>::sample(),
        "i64 over itself",
    );

    let bs = [Reach(false), Reach(true)];
    laws::left_semimodule_laws(&bs, &bs, "Reach over itself");

    let zs = [Zn::<6>::new(0), Zn::<6>::new(1), Zn::<6>::new(5)];
    laws::left_semimodule_laws(&zs, &zs, "Zn<6> over itself");
}

#[test]
fn the_boolean_dioid_is_an_ordered_semiring() {
    // Monotonicity is what makes a relaxation step sound: extending two ordered partial results
    // by the same factor must not reorder them.
    laws::ordered_semiring_laws(&[Reach(false), Reach(true)], "Reach");
}

/// The by-reference magma operations, for every carrier that offers them.
///
/// A separate test rather than a line inside each carrier's block above, because the `Ref`
/// traits are not part of the tier tower — a carrier can host a semiring without them — so the
/// check is keyed on the operations rather than on the tier.
///
/// Mutation testing is what prompted it: `add_ref` and `mul_ref` were replaceable with
/// `Default::default()` for every carrier in the crate, returning zero for all input, with the
/// whole workspace suite green. See [`conformance::ref_magma_laws`](crate::conformance::ref_magma_laws)
/// for why the duplicate implementation is the one nothing exercises.
#[test]
fn by_reference_operations_agree_with_by_value() {
    laws::ref_magma_laws_for::<i64>("i64");
    // `i128` has the operations but declares no `ConformanceSample`, so it brings its own.
    laws::ref_magma_laws(&[0_i128, 1, -1, 7, -12], "i128");
    laws::ref_magma_laws_for::<f32>("f32");
    laws::ref_magma_laws_for::<f64>("f64");
    laws::ref_magma_laws_for::<Rational64>("Rational64");
    laws::ref_magma_laws_for::<GaussianRational>("GaussianRational");
    laws::ref_magma_laws_for::<Zp<7>>("Zp<7>");
    laws::ref_magma_laws_for::<Zn<6>>("Zn<6>");
}
