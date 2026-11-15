//! Every carrier this crate ships, checked against its tier over **generated**
//! elements.
//!
//! The companion to `src/tests/laws.rs`, which runs the same laws over
//! `ConformanceSample`'s six fixed values. Nothing is restated here: the laws
//! are [`euclid_algebra::conformance`]'s public functions, taking a slice, and
//! this file changes only where the slice comes from. A law added there is
//! immediately checked over generated input too, and a law cannot drift between
//! the two suites because there is only one copy of it.
//!
//! ## Why generate at all, when a tier either holds or it does not
//!
//! `conformance`'s own documentation makes the case for fixed samples, and it is
//! a good one: an algebraic law over a small carrier "either holds structurally
//! or fails on the first awkward triple", so six well-chosen values find what a
//! thousand random ones would. That argument is about the *law*, and it is
//! right.
//!
//! What it does not cover is the carrier's **representation**. `Rational` has to
//! stay reduced, with the sign on the numerator, after arithmetic rather than
//! only after `new`. `Zp`'s Fermat inverse has to agree with `ONE / a` at every
//! residue, not just at the four the sample lists. `GaussianRational` carries
//! two rationals that each have to stay reduced through a complex multiply that
//! mixes them. Those are properties of a large state space — the exception
//! `euclid`'s own docs name — and a fixed list of six cannot cover the
//! interaction between a reduction step and an overflow boundary.
//!
//! So the division of labour is: the fixed suite proves the tiers are
//! *satisfiable* and that our carriers satisfy them; this one hunts the
//! representation invariants underneath, and runs the tier laws again for free
//! while it is there.
//!
//! ## Everything here is exact
//!
//! No tolerances anywhere, which is unusual for a generated suite and worth
//! saying. These carriers are exact arithmetic — integers, reduced fractions,
//! residues — so every comparison is `assert_eq!` inside the conformance
//! functions and every failure is a real disagreement rather than a rounding
//! question. The whole category of "is this tolerance honest" that dominates the
//! geometry suites simply does not arise.
//!
//! ## The ranges are bounded by overflow, not by taste
//!
//! `Rational64` multiplies numerators and denominators, and the conformance
//! laws run over *triples* — distributivity alone forms `(a + b) · c`, which
//! takes a product of three denominators and a sum over three numerators. In a
//! debug build an `i64` overflow panics, and that panic is indistinguishable
//! from a law failure at the call site.
//!
//! The bounds below keep a triple's worst case orders of magnitude inside
//! `i64`, so a panic from this suite means a broken law rather than an
//! arithmetic edge nobody intended to test. Overflow behaviour is a separate
//! question and belongs in a suite that asserts something about it; silently
//! half-covering it here would make every other failure ambiguous.

use super::*;
use crate::conformance::{self as laws};
use euclid_stability::{Config, Gen, check, classify, generate, property};

/// Enough cases that the shrinker has something to work with, and few enough
/// that each one running every tier law over a triple stays quick.
fn settings(suite: &str) -> Config {
    Config::default()
        .cases(300)
        .regressions(format!("src/tests/regressions/{suite}.txt"))
}

/// A small integer, weighted toward the values that break things.
///
/// `0` and `±1` are the identities and the annihilator, so they are the ones a
/// law is most likely to get wrong, and a uniform draw over a wide range would
/// almost never produce them.
fn small_int() -> Gen<i64> {
    generate::weighted(vec![
        (3, generate::just(0)),
        (3, generate::just(1)),
        (3, generate::just(-1)),
        (1, generate::just(i64::from(u8::MAX))),
        (10, generate::int(-12, 12)),
    ])
}

/// A nonzero small integer, for a denominator.
fn nonzero_small_int() -> Gen<i64> {
    small_int().filter("nonzero", |v: &i64| *v != 0)
}

fn rational64() -> Gen<Rational64> {
    generate::pair(small_int(), nonzero_small_int()).map(|(n, d)| Rational64::new(n, d))
}

fn rational128() -> Gen<Rational128> {
    generate::pair(small_int(), nonzero_small_int())
        .map(|(n, d)| Rational128::new(i128::from(n), i128::from(d)))
}

fn gaussian() -> Gen<GaussianRational> {
    generate::pair(rational128(), rational128()).map(|(re, im)| GaussianRational::new(re, im))
}

/// A residue, drawn across the whole ring including the ones adjacent to the
/// modulus where a missed reduction shows.
fn residue<const N: u64>() -> Gen<u64> {
    generate::int(0, (N as i64) - 1).map(|v| v as u64)
}

/// Sample sizes for the tier laws.
///
/// Three is the minimum that exercises a three-operand law at all; six matches
/// what the fixed suite uses, and the laws are `O(n³)`, so going wider buys
/// repetition rather than coverage.
const MIN_SAMPLE: usize = 3;
const MAX_SAMPLE: usize = 6;

/// The reduced-form invariant `Rational` promises, as a predicate.
fn is_reduced(r: Rational64) -> bool {
    fn gcd(mut a: i64, mut b: i64) -> i64 {
        a = a.abs();
        b = b.abs();
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }
    // The denominator is positive (the sign lives on the numerator), and the
    // two share no factor. Zero is the one case with a free choice, and the
    // type's answer is 0/1.
    r.denominator() > 0 && gcd(r.numerator(), r.denominator()) == 1
}

// ===== The integers =======================================================

property! {
    #![config(settings("integers"))]

    /// `i64` against every tier it claims.
    fn integers_satisfy_their_tiers(xs in generate::vec(small_int(), MIN_SAMPLE, MAX_SAMPLE)) {
        classify!(if xs.contains(&0) { "contains zero" } else { "no zero" });
        laws::commutative_semiring_laws(&xs, "i64");
        laws::ring_laws(&xs, "i64");
        laws::euclidean_laws(&xs, "i64");
    }
}

// ===== Rational ===========================================================

property! {
    #![config(settings("rational"))]

    /// `Rational64` against every tier it claims, up to `Field`.
    fn rationals_satisfy_their_tiers(
        xs in generate::vec(rational64(), MIN_SAMPLE, MAX_SAMPLE),
    ) {
        classify!(if xs.contains(&Rational64::ZERO) {
            "contains zero"
        } else {
            "no zero"
        });
        laws::commutative_semiring_laws(&xs, "Rational64");
        laws::ring_laws(&xs, "Rational64");
        laws::field_laws(&xs, "Rational64");
        laws::euclidean_laws(&xs, "Rational64");
    }

    /// The invariant the type promises, after **arithmetic** rather than after
    /// `new` — which is where a missing `reduce` would show.
    ///
    /// This is the property the fixed suite can only spot-check: it holds the
    /// claim over four hand-written values, while a reduction bug needs a
    /// specific common factor to appear in a specific operand position. The
    /// generator produces those.
    fn rationals_stay_reduced_through_arithmetic(a in rational64(), b in rational64()) {
        classify!(if a == b { "equal operands" } else { "distinct" });
        check!(is_reduced(a), "a generated rational is not reduced: {a:?}");

        for (op, got) in [
            ("a + b", a + b),
            ("a - b", a - b),
            ("a * b", a * b),
        ] {
            check!(
                is_reduced(got),
                "{op} left {got:?} unreduced (from {a:?} and {b:?})"
            );
        }

        if b != Rational64::ZERO {
            let q = a / b;
            check!(is_reduced(q), "a / b left {q:?} unreduced");
        }
    }

    /// A rational built from a scaled pair is the same number as the unscaled
    /// one — the metamorphic statement of reduction, which needs no reference
    /// value.
    fn scaling_a_fraction_does_not_change_it(
        a in rational64(),
        k in generate::int(2, 11),
    ) {
        classify!(if a.numerator() == 0 { "zero" } else { "nonzero" });
        let scaled = Rational64::new(a.numerator() * k, a.denominator() * k);
        check!(
            scaled == a,
            "{}/{} scaled by {k} became {scaled:?}, not {a:?}",
            a.numerator(),
            a.denominator()
        );
    }
}

// ===== Gaussian rationals =================================================

property! {
    #![config(settings("gaussian"))]

    /// `GaussianRational` against every tier it claims.
    fn gaussian_rationals_satisfy_their_tiers(
        xs in generate::vec(gaussian(), MIN_SAMPLE, MAX_SAMPLE),
    ) {
        classify!(if xs.iter().any(|g| g.im() == Rational128::integer(0)) {
            "contains a real"
        } else {
            "all properly complex"
        });
        laws::commutative_semiring_laws(&xs, "GaussianRational");
        laws::ring_laws(&xs, "GaussianRational");
        laws::field_laws(&xs, "GaussianRational");
    }

    /// The norm is multiplicative: `N(ab) = N(a)N(b)`.
    ///
    /// The identity that makes ℚ(i) a normed division algebra rather than two
    /// rationals in a trenchcoat, and the one a wrong `mul` breaks while still
    /// passing every additive law.
    fn the_gaussian_norm_is_multiplicative(a in gaussian(), b in gaussian()) {
        classify!(if a.im() == Rational128::integer(0) || b.im() == Rational128::integer(0) {
            "a real operand"
        } else {
            "both complex"
        });
        check!(
            (a * b).norm() == a.norm() * b.norm(),
            "N(ab) = {:?} but N(a)N(b) = {:?}",
            (a * b).norm(),
            a.norm() * b.norm()
        );
    }

    /// Conjugation is an involution, and `a · conj(a)` is the norm — so the
    /// norm is real.
    fn the_norm_is_the_product_with_the_conjugate(a in gaussian()) {
        classify!(if a.im() == Rational128::integer(0) { "real" } else { "complex" });
        let conj = GaussianRational::new(a.re(), -a.im());
        let product = a * conj;
        check!(
            product.im() == Rational128::integer(0),
            "a · conj(a) should be real, got imaginary part {:?}",
            product.im()
        );
        check!(
            product.re() == a.norm(),
            "a · conj(a) = {:?} but the norm is {:?}",
            product.re(),
            a.norm()
        );
    }
}

// ===== Modular arithmetic =================================================

property! {
    #![config(settings("modular"))]

    /// `Zp` is a field for a prime modulus, at every residue.
    fn zp_is_a_field(xs in generate::vec(residue::<7>(), MIN_SAMPLE, MAX_SAMPLE)) {
        let zs: Vec<Zp<7>> = xs.iter().map(|&v| Zp::<7>::new(v)).collect();
        classify!(if xs.contains(&0) { "contains zero" } else { "no zero" });
        laws::commutative_semiring_laws(&zs, "Zp<7>");
        laws::ring_laws(&zs, "Zp<7>");
        laws::field_laws(&zs, "Zp<7>");
    }

    /// The Fermat inverse agrees with division by `ONE`, at every nonzero
    /// residue rather than the handful a fixed sample lists.
    fn the_fermat_inverse_agrees_with_division(x in residue::<7>()) {
        let a = Zp::<7>::new(x);
        classify!(if x == 0 { "zero" } else { "unit" });
        if a == Zp::<7>::ZERO {
            return Ok(());
        }
        let one = Zp::<7>::ONE;
        check!(
            a.inverse() == one / a,
            "inverse({x}) = {:?} but ONE / {x} = {:?}",
            a.inverse(),
            one / a
        );
        check!(
            a * a.inverse() == one,
            "{x} · inverse({x}) is not ONE"
        );
    }

    /// `Zn` reaches the ring laws for a composite modulus, and no further.
    fn zn_satisfies_the_ring_laws(xs in generate::vec(residue::<6>(), MIN_SAMPLE, MAX_SAMPLE)) {
        let zs: Vec<Zn<6>> = xs.iter().map(|&v| Zn::<6>::new(v)).collect();
        classify!(if xs.iter().any(|v| *v == 2 || *v == 3) {
            "contains a zero divisor"
        } else {
            "no zero divisor"
        });
        laws::commutative_semiring_laws(&zs, "Zn<6>");
        laws::ring_laws(&zs, "Zn<6>");
    }

    /// Reduction is idempotent: a residue re-reduced is itself.
    ///
    /// Trivial-looking and the reason `Zn::new` takes `x % N` rather than
    /// trusting its caller — a generated value well above the modulus is what
    /// distinguishes the two.
    fn reducing_a_residue_twice_changes_nothing(x in generate::int(0, 10_000)) {
        let once = Zn::<6>::new(x as u64);
        let twice = Zn::<6>::new(once.value());
        classify!(if x < 6 { "already reduced" } else { "needs reduction" });
        check!(once == twice, "reducing {x} twice gave {twice:?}, not {once:?}");
        check!(once.value() < 6, "residue {} is not in [0, 6)", once.value());
    }
}
