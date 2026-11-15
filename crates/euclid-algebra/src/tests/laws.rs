//! The tier laws, checked as laws.
//!
//! Every structure in this crate is *defined* by the identities its operations satisfy — a
//! [`Semiring`] is not a type with `add` and `mul`, it is one where `⊗` distributes over `⊕` and
//! `ZERO` annihilates. So the test for a carrier reaching a tier is the tier's own axioms, run
//! over a fixed sample of that carrier, rather than a handful of transcribed products.
//!
//! That makes these tests **metamorphic**: each asserts a relation between results, never a
//! reference value, so none of them can be mistyped into passing. It also makes them total in a
//! useful way — [`semiring_laws`] applied to a new carrier checks the same eight axioms the
//! existing ones satisfy, so adding a carrier is a one-line call rather than a new suite.
//!
//! # Samples, not random inputs
//!
//! The samples are hand-picked and fixed, matching the house rule in `euclid-numeric`'s test
//! tree: a failure must be reproducible from the test name alone, with no seed to recover. Each
//! set deliberately includes the values that break naive implementations — `ZERO`, `ONE`, a
//! negative, and for the modular carriers the residues adjacent to the modulus, where a missed
//! reduction shows up.
//!
//! Laws are checked over every pair (and every triple, where the law needs three) drawn from the
//! sample, so a set of six values exercises 36 pairs and 216 triples per law.
//!
//! # What is deliberately *not* here
//!
//! `Zn<N>` for composite `N` has zero divisors, so it must **not** reach [`IntegralDomain`]. That
//! is a compile-time property — the impl simply does not exist — and the closest a running test
//! can come is [`zn_composite_has_zero_divisors`], which demonstrates the witness pair. The
//! negative half is enforced by the type system, not by this file.

use super::*;

// --- the law harnesses ----------------------------------------------------------------------

/// The eight semiring axioms, over the method-based `⊕`/`⊗` trunk.
///
/// Two identities each (left and right) for the additive and multiplicative identity, zero
/// annihilation and distributivity; one each for additive and multiplicative commutativity and
/// associativity.
///
/// Stated against [`AdditiveMagma`]/[`MultiplicativeMagma`] rather than the operators, so a
/// carrier that hosts a semiring *without* overloading `+` (min-plus, boolean reachability) is
/// checked by the same code as `f64`.
fn semiring_laws<T>(xs: &[T], what: &str)
where
    T: Semiring + CommutativeMultiplication + core::fmt::Debug + Copy,
{
    for &a in xs {
        // additive and multiplicative identity
        assert_eq!(a.add(T::ZERO), a, "{what}: a ⊕ 0 = a");
        assert_eq!(T::ZERO.add(a), a, "{what}: 0 ⊕ a = a");
        assert_eq!(a.mul(T::ONE), a, "{what}: a ⊗ 1 = a");
        assert_eq!(T::ONE.mul(a), a, "{what}: 1 ⊗ a = a");
        // zero annihilates
        assert_eq!(a.mul(T::ZERO), T::ZERO, "{what}: a ⊗ 0 = 0");
        assert_eq!(T::ZERO.mul(a), T::ZERO, "{what}: 0 ⊗ a = 0");

        for &b in xs {
            // commutativity
            assert_eq!(a.add(b), b.add(a), "{what}: a ⊕ b = b ⊕ a");
            assert_eq!(a.mul(b), b.mul(a), "{what}: a ⊗ b = b ⊗ a");

            for &c in xs {
                // associativity
                assert_eq!(
                    a.add(b).add(c),
                    a.add(b.add(c)),
                    "{what}: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)"
                );
                assert_eq!(
                    a.mul(b).mul(c),
                    a.mul(b.mul(c)),
                    "{what}: (a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)"
                );
                // distributivity, both sides
                assert_eq!(
                    a.mul(b.add(c)),
                    a.mul(b).add(a.mul(c)),
                    "{what}: a ⊗ (b ⊕ c) = a⊗b ⊕ a⊗c"
                );
                assert_eq!(
                    b.add(c).mul(a),
                    b.mul(a).add(c.mul(a)),
                    "{what}: (b ⊕ c) ⊗ a = b⊗a ⊕ c⊗a"
                );
            }
        }
    }
}

/// The ring axioms the semiring lacks — additive inverses — plus the **operator bridge**.
///
/// The bridge is the one law unique to this crate's design: a numeric carrier joins the
/// method-based trunk by making `AdditiveMagma::add` *be* `+`. If the two ever disagree, a
/// formula written against the trunk and one written with operators compute different things on
/// the same type, which is precisely the failure the library exists to prevent.
#[allow(
    clippy::eq_op,
    reason = "`a - a` and `a / a` are the laws under test, not accidental self-comparisons"
)]
fn ring_laws<T>(xs: &[T], what: &str)
where
    T: Ring + CommutativeMultiplication + core::fmt::Debug + Copy,
{
    for &a in xs {
        assert_eq!(a + (-a), T::ZERO, "{what}: a + (−a) = 0");
        assert_eq!(-(-a), a, "{what}: −(−a) = a");
        assert_eq!(a - a, T::ZERO, "{what}: a − a = 0");

        for &b in xs {
            // the operator bridge
            assert_eq!(AdditiveMagma::add(a, b), a + b, "{what}: ⊕ is +");
            assert_eq!(MultiplicativeMagma::mul(a, b), a * b, "{what}: ⊗ is *");
            // subtraction is addition of the inverse
            assert_eq!(a - b, a + (-b), "{what}: a − b = a + (−b)");
            // negation distributes through multiplication
            assert_eq!(-(a * b), (-a) * b, "{what}: −(ab) = (−a)b");
        }
    }
}

/// The field axiom: every nonzero element is invertible, and division is multiplication by the
/// inverse.
///
/// `inverse` is not a trait method — [`Field`] exposes invertibility through `Div` — so the
/// inverse is obtained as `ONE / a`, which is also what makes the second law non-trivial.
#[allow(
    clippy::eq_op,
    reason = "`a / a` is the law under test, not an accidental self-comparison"
)]
fn field_laws<T>(xs: &[T], what: &str)
where
    T: Field + CommutativeMultiplication + core::fmt::Debug + Copy,
{
    for &a in xs {
        if a == T::ZERO {
            continue; // not invertible, and division by it is undefined
        }
        let inv = T::ONE / a;
        assert_eq!(a * inv, T::ONE, "{what}: a · a⁻¹ = 1");
        assert_eq!(a / a, T::ONE, "{what}: a / a = 1");

        for &b in xs {
            assert_eq!(b / a, b * inv, "{what}: b / a = b · a⁻¹");
        }
    }
    // A field is an integral domain: no zero divisors among the sample.
    for &a in xs {
        for &b in xs {
            if a != T::ZERO && b != T::ZERO {
                assert_ne!(a * b, T::ZERO, "{what}: no zero divisors");
            }
        }
    }
}

/// `div_rem` must reconstruct its input: `a = (a/b)·b + (a mod b)`.
///
/// This is the defining property of a Euclidean domain and the one a caller actually relies on,
/// and it holds for every carrier here regardless of how each rounds the quotient — which is why
/// it is worth more than asserting a particular quotient.
fn euclidean_laws<T>(xs: &[T], what: &str)
where
    T: EuclideanDomain + CommutativeMultiplication + core::fmt::Debug + Copy,
{
    for &a in xs {
        for &b in xs {
            if b == T::ZERO {
                continue;
            }
            let (q, r) = a.div_rem(b);
            assert_eq!(q * b + r, a, "{what}: a = (a/b)·b + (a mod b)");
        }
    }
}

// --- the carriers ---------------------------------------------------------------------------

/// `0`, `1`, a negative, and two values whose product overflows nothing but is not a round
/// number — enough to catch a dropped sign or a mis-ordered operand.
const I64S: [i64; 6] = [0, 1, -1, 7, -3, 12];

#[test]
fn integers_satisfy_semiring_ring_and_euclidean_laws() {
    semiring_laws(&I64S, "i64");
    ring_laws(&I64S, "i64");
    euclidean_laws(&I64S, "i64");
}

#[test]
fn rationals_satisfy_the_field_laws() {
    let xs = [
        Rational64::integer(0),
        Rational64::integer(1),
        Rational64::new(-1, 2),
        Rational64::new(3, 4),
        Rational64::new(-7, 3),
        Rational64::new(5, 1),
    ];
    semiring_laws(&xs, "Rational64");
    ring_laws(&xs, "Rational64");
    field_laws(&xs, "Rational64");
    euclidean_laws(&xs, "Rational64");
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
            for c in [a + b, a - b, a * b] {
                assert!(c.denominator() > 0, "denominator stays positive: {c:?}");
                let g = gcd(c.numerator(), c.denominator());
                assert_eq!(g, 1, "in lowest terms: {c:?} has gcd {g}");
            }
            if b != Rational64::integer(0) {
                let c = a / b;
                assert!(c.denominator() > 0, "denominator stays positive: {c:?}");
                assert_eq!(
                    gcd(c.numerator(), c.denominator()),
                    1,
                    "lowest terms: {c:?}"
                );
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
    let xs = [
        Zp::<7>::new(0),
        Zp::<7>::new(1),
        Zp::<7>::new(3),
        Zp::<7>::new(6),
        Zp::<7>::from_i64(-1),
        Zp::<7>::new(9),
    ];
    semiring_laws(&xs, "Zp<7>");
    ring_laws(&xs, "Zp<7>");
    field_laws(&xs, "Zp<7>");

    // Reduction happens on construction, from both directions.
    assert_eq!(Zp::<7>::new(9).value(), 2);
    assert_eq!(Zp::<7>::from_i64(-1).value(), 6);

    // And the dedicated `inverse` agrees with `ONE / a`.
    for &a in &xs {
        if a != Zp::<7>::ZERO {
            assert_eq!(a.inverse(), Zp::<7>::ONE / a, "inverse agrees with 1/a");
            assert_eq!(a * a.inverse(), Zp::<7>::ONE);
        }
    }
}

#[test]
fn zn_satisfies_the_ring_laws_for_a_composite_modulus() {
    let xs = [
        Zn::<6>::new(0),
        Zn::<6>::new(1),
        Zn::<6>::new(2),
        Zn::<6>::new(3),
        Zn::<6>::new(5),
        Zn::<6>::from_i64(-1),
    ];
    semiring_laws(&xs, "Zn<6>");
    ring_laws(&xs, "Zn<6>");
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
    let q = Rational128::new;
    let xs = [
        GaussianRational::integer(0),
        GaussianRational::integer(1),
        GaussianRational::new(q(1, 2), q(1, 2)),
        GaussianRational::new(q(-3, 4), q(5, 4)),
        GaussianRational::new(q(0, 1), q(1, 1)), // i
        GaussianRational::new(q(2, 1), q(-1, 1)),
    ];
    semiring_laws(&xs, "GaussianRational");
    ring_laws(&xs, "GaussianRational");
    field_laws(&xs, "GaussianRational");
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
    let xs = [0.0_f64, 1.0, -1.0, 0.5, -2.5, 4.0];
    semiring_laws(&xs, "f64");
    ring_laws(&xs, "f64");
    field_laws(&xs, "f64");

    let xs32 = [0.0_f32, 1.0, -1.0, 0.5, -2.5, 4.0];
    semiring_laws(&xs32, "f32");
    ring_laws(&xs32, "f32");
    field_laws(&xs32, "f32");
}
