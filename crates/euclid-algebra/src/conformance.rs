//! Conformance suites — the tier axioms as functions you can check a carrier against.
//!
//! This crate says which algebraic tier a carrier reaches, but a tier is a *claim*: writing
//! `impl Semiring for MyThing {}` asserts that `⊗` distributes over `⊕` and that `ZERO`
//! annihilates, and nothing in the type system verifies either. These functions do.
//!
//! ```
//! # use euclid_algebra::conformance;
//! // A carrier author checks the claim rather than asserting it.
//! conformance::semiring_laws(&[0_i64, 1, -1, 7], "i64");
//! conformance::ring_laws(&[0_i64, 1, -1, 7], "i64");
//! ```
//!
//! # Why this ships with the contract
//!
//! The failure this guards against is two implementors reading the same signature differently.
//! A carrier that gets `⊕` subtly wrong still compiles, still type-checks at every call site,
//! and silently computes different answers from every other carrier — which defeats the point
//! of having a shared vocabulary at all. The contract is not really complete until an
//! implementor can ask whether they honored it.
//!
//! # Availability
//!
//! Gated behind the `conformance` feature, so a consumer that only writes formulas pays
//! nothing for it, and always compiled under `cfg(test)` so this crate's own suites exercise
//! the same code a carrier author calls.
//!
//! Note that these docs are the module's own (`//!`), with no outer `///` on the `mod`
//! declaration. The two forms are concatenated, and an outer comment resolves its intra-doc
//! links in the *parent's* scope — so a link to a function in here would silently fail to
//! resolve from there. Keep the documentation inside the module.
//!
//! # How to use them
//!
//! Each function takes a **sample** of your carrier and a label for failure messages. The laws
//! needing three operands run over every triple, so a sample of six values yields 216 cases per
//! law — pick values that stress your implementation rather than many values:
//!
//! - `ZERO` and `ONE`, which is where a missing identity case shows
//! - a negative, or whatever your carrier's awkward sign case is
//! - for a modular or saturating carrier, the values adjacent to the boundary
//!
//! Samples are yours to choose and fix. These functions generate nothing, so a failure is
//! reproducible from the test that called them with no seed to recover.
//!
//! # What the laws cannot check
//!
//! A tier a carrier must *not* reach is enforced by the absent impl, not by these functions.
//! `Zn<6>` has zero divisors so it must not be an [`IntegralDomain`](crate::IntegralDomain);
//! the type system is what guarantees that, and the closest a running check comes is
//! demonstrating the witness pair yourself.
//!
//! Panics are the reporting channel, as in any assertion helper — a failure names the law and
//! the label, so the message identifies which axiom broke and on which carrier.

use crate::{
    AdditiveMagma, CanonicalOrder, CommutativeSemiring, Dioid, EuclideanDomain, Field,
    MultiplicativeMagma, Ring, Semiring, StarSemiring,
};

/// The eight semiring axioms, over the method-based `⊕`/`⊗` trunk.
///
/// Two identities each (left and right) for the additive and multiplicative identity, zero
/// annihilation and distributivity; one each for additive and multiplicative associativity.
///
/// Commutativity is **not** checked here — a matrix semiring is not commutative, and requiring
/// it would exclude the carriers this tier exists to admit. Use
/// [`commutative_semiring_laws`] when the carrier claims it.
///
/// Stated against [`AdditiveMagma`]/[`MultiplicativeMagma`] rather than the operators, so a
/// carrier that hosts a semiring *without* overloading `+` — min-plus, boolean reachability,
/// counting — is checked by the same code as `f64`.
pub fn semiring_laws<T>(xs: &[T], what: &str)
where
    T: Semiring + core::fmt::Debug + Copy,
{
    for &a in xs {
        assert_eq!(a.add(T::ZERO), a, "{what}: a ⊕ 0 = a");
        assert_eq!(T::ZERO.add(a), a, "{what}: 0 ⊕ a = a");
        assert_eq!(a.mul(T::ONE), a, "{what}: a ⊗ 1 = a");
        assert_eq!(T::ONE.mul(a), a, "{what}: 1 ⊗ a = a");
        assert_eq!(a.mul(T::ZERO), T::ZERO, "{what}: a ⊗ 0 = 0");
        assert_eq!(T::ZERO.mul(a), T::ZERO, "{what}: 0 ⊗ a = 0");

        for &b in xs {
            for &c in xs {
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

/// [`semiring_laws`] plus both commutativity axioms.
pub fn commutative_semiring_laws<T>(xs: &[T], what: &str)
where
    T: CommutativeSemiring + core::fmt::Debug + Copy,
{
    semiring_laws(xs, what);
    for &a in xs {
        for &b in xs {
            assert_eq!(a.add(b), b.add(a), "{what}: a ⊕ b = b ⊕ a");
            assert_eq!(a.mul(b), b.mul(a), "{what}: a ⊗ b = b ⊗ a");
        }
    }
}

/// The ring axioms a semiring lacks — additive inverses — plus the **operator bridge**.
///
/// The bridge is the law specific to this crate's design: a numeric carrier joins the
/// method-based trunk by making [`AdditiveMagma::add`] *be* `+`. If the two ever disagree, a
/// formula written against the trunk and one written with operators compute different things on
/// the same type, which is precisely the failure the library exists to prevent.
#[allow(
    clippy::eq_op,
    reason = "`a - a` is the law under test, not an accidental self-comparison"
)]
pub fn ring_laws<T>(xs: &[T], what: &str)
where
    T: Ring + core::fmt::Debug + Copy,
{
    for &a in xs {
        assert_eq!(a + (-a), T::ZERO, "{what}: a + (−a) = 0");
        assert_eq!(-(-a), a, "{what}: −(−a) = a");
        assert_eq!(a - a, T::ZERO, "{what}: a − a = 0");

        for &b in xs {
            assert_eq!(AdditiveMagma::add(a, b), a + b, "{what}: ⊕ is +");
            assert_eq!(MultiplicativeMagma::mul(a, b), a * b, "{what}: ⊗ is *");
            assert_eq!(a - b, a + (-b), "{what}: a − b = a + (−b)");
            assert_eq!(-(a * b), (-a) * b, "{what}: −(ab) = (−a)b");
        }
    }
}

/// The field axiom: every nonzero element is invertible, and division is multiplication by the
/// inverse.
///
/// [`Field`] exposes invertibility through `Div` rather than an `inverse` method, so the inverse
/// is obtained as `ONE / a` — which is also what makes the division law non-trivial. A field is
/// an integral domain, so the absence of zero divisors in the sample is checked too.
#[allow(
    clippy::eq_op,
    reason = "`a / a` is the law under test, not an accidental self-comparison"
)]
pub fn field_laws<T>(xs: &[T], what: &str)
where
    T: Field + core::fmt::Debug + Copy,
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
    for &a in xs {
        for &b in xs {
            if a != T::ZERO && b != T::ZERO {
                assert_ne!(a * b, T::ZERO, "{what}: no zero divisors");
            }
        }
    }
}

/// `div_rem` must reconstruct its input: `a = (a / b) · b + (a mod b)`.
///
/// This is the defining property of a Euclidean domain and the one a caller actually relies on.
/// It holds regardless of how a carrier rounds the quotient, which is why it is worth more than
/// asserting any particular quotient — two carriers may round differently and both be correct.
pub fn euclidean_laws<T>(xs: &[T], what: &str)
where
    T: EuclideanDomain + core::fmt::Debug + Copy,
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

/// The dioid axiom: `⊕` is idempotent, so `a ⊕ a = a`.
///
/// This is what separates the algebraic-path tier (min-plus, boolean reachability,
/// widest-path) from the numeric one, and it is the axiom a carrier is most likely to claim by
/// accident — the impl is a bare marker, so nothing stops a non-idempotent carrier declaring it.
pub fn dioid_laws<T>(xs: &[T], what: &str)
where
    T: Dioid + core::fmt::Debug + Copy,
{
    semiring_laws(xs, what);
    for &a in xs {
        assert_eq!(a.add(a), a, "{what}: a ⊕ a = a (idempotent)");
        for &b in xs {
            // Idempotent addition is commutative at this tier, by `AdditiveSemilattice`.
            assert_eq!(a.add(b), b.add(a), "{what}: a ⊕ b = b ⊕ a");
        }
    }
}

/// The star axiom: `a* = 1 ⊕ a ⊗ a*`.
///
/// The fixed-point identity that makes `star` a closure operator, and the one a path-algebra
/// kernel depends on when it replaces an unbounded sum with a single call.
pub fn star_laws<T>(xs: &[T], what: &str)
where
    T: StarSemiring + core::fmt::Debug + Copy,
{
    for &a in xs {
        assert_eq!(
            a.star(),
            T::ONE.add(a.mul(a.star())),
            "{what}: a* = 1 ⊕ a ⊗ a*"
        );
    }
}

/// The canonical order agrees with `⊕`: `a ⪯ b` exactly when `a ⊕ b = a`.
///
/// [`CanonicalOrder::precedes`] is a provided method defined that way, so this checks the
/// properties the definition implies — reflexivity, antisymmetry and transitivity — which a
/// carrier can still break by overriding it.
pub fn canonical_order_laws<T>(xs: &[T], what: &str)
where
    T: CanonicalOrder + core::fmt::Debug + Copy,
{
    for &a in xs {
        assert!(a.precedes(a), "{what}: a ⪯ a (reflexive)");
        for &b in xs {
            assert_eq!(a.precedes(b), a.add(b) == a, "{what}: a ⪯ b iff a ⊕ b = a");
            if a.precedes(b) && b.precedes(a) {
                assert_eq!(a, b, "{what}: a ⪯ b and b ⪯ a imply a = b (antisymmetric)");
            }
            for &c in xs {
                if a.precedes(b) && b.precedes(c) {
                    assert!(a.precedes(c), "{what}: ⪯ is transitive");
                }
            }
        }
    }
}
