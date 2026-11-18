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
//! The `generated` submodule sits behind a **second** feature, `conformance-generated`, which
//! enables this one and additionally pulls the property engine. The division: these functions
//! are the contract, so `conformance` resolves no engine at all and is what a consumer binding
//! the laws should take; the generated search is tooling for locating where a carrier fails
//! them, and costs a dependency.
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

use crate::witness::{DioidWitness, SemiringWitness};
use crate::{
    AdditiveMagma, CanonicalOrder, CommutativeSemiring, Dioid, EuclideanDomain, Field,
    LeftSemimodule, MultiplicativeMagma, OrderedSemiring, RefAdditiveMagma, RefMultiplicativeMagma,
    Ring, Semiring, StarSemiring,
};

/// A carrier's own curated sample, declared once.
///
/// This is euclid's answer to proptest's `Arbitrary`, with the generator replaced by a fixed
/// list. A carrier says which values stress *it* — `ZERO` and `ONE`, a negative, the residues
/// either side of a modulus — and every suite in this crate then draws from the same place,
/// rather than each call site inventing a sample and getting it slightly wrong.
///
/// ```ignore
/// impl ConformanceSample for MyCarrier {
///     fn sample() -> Vec<Self> {
///         vec![Self::ZERO, Self::ONE, Self::from(-3), Self::from(7)]
///     }
/// }
///
/// // every suite now needs no sample argument
/// ring_laws_for::<MyCarrier>("MyCarrier");
/// ```
///
/// # Why fixed rather than generated
///
/// A generator buys input breadth at the cost of reproducibility: a failure becomes a seed to
/// recover rather than a value in the source. The laws here are algebraic, so breadth adds
/// little — `(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)` either holds structurally or fails on the first
/// awkward triple — while reproducibility is what makes a failure diagnosable.
///
/// The trade would be different for a property over a large state space, which is where
/// proptest earns its shrinking machinery. It is not where these laws live.
///
/// # Keep it small
///
/// The three-operand laws run over every triple, so a sample of `n` values costs `n³` cases per
/// law. Six is comfortable; twenty is 8,000 triples per law and buys nothing a well-chosen six
/// did not already catch.
pub trait ConformanceSample: Sized {
    /// The values this carrier should be checked over.
    fn sample() -> Vec<Self>;
}

/// [`semiring_laws`] over a carrier's own [`ConformanceSample`].
pub fn semiring_laws_for<T>(what: &str)
where
    T: Semiring + ConformanceSample + core::fmt::Debug + Copy,
{
    semiring_laws(&T::sample(), what);
}

/// [`commutative_semiring_laws`] over a carrier's own [`ConformanceSample`].
pub fn commutative_semiring_laws_for<T>(what: &str)
where
    T: CommutativeSemiring + ConformanceSample + core::fmt::Debug + Copy,
{
    commutative_semiring_laws(&T::sample(), what);
}

/// [`ring_laws`] over a carrier's own [`ConformanceSample`].
pub fn ring_laws_for<T>(what: &str)
where
    T: Ring + ConformanceSample + core::fmt::Debug + Copy,
{
    ring_laws(&T::sample(), what);
}

/// [`field_laws`] over a carrier's own [`ConformanceSample`].
pub fn field_laws_for<T>(what: &str)
where
    T: Field + ConformanceSample + core::fmt::Debug + Copy,
{
    field_laws(&T::sample(), what);
}

/// [`euclidean_laws`] over a carrier's own [`ConformanceSample`].
pub fn euclidean_laws_for<T>(what: &str)
where
    T: EuclideanDomain + ConformanceSample + core::fmt::Debug + Copy,
{
    euclidean_laws(&T::sample(), what);
}

/// [`dioid_laws`] over a carrier's own [`ConformanceSample`].
pub fn dioid_laws_for<T>(what: &str)
where
    T: Dioid + ConformanceSample + core::fmt::Debug + Copy,
{
    dioid_laws(&T::sample(), what);
}

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

            // The **compound-assignment bridge**, which is the operator bridge's other half and
            // was missing until mutation testing noticed. `Ring` requires `AddAssign`,
            // `SubAssign` and `MulAssign`, and nothing checked that any of them agrees with the
            // binary operator it abbreviates — so `a += b` could be written to do nothing at all
            // and every carrier in the fleet still conformed.
            //
            // It is a separate impl block from `Add`, which is exactly why it drifts: the usual
            // spelling is `*self = *self + rhs`, and the usual mistake is a typo in the operator
            // or an operand order that only matters for a non-commutative carrier.
            let mut assigned = a;
            assigned += b;
            assert_eq!(assigned, a + b, "{what}: a += b agrees with a + b");

            let mut assigned = a;
            assigned -= b;
            assert_eq!(assigned, a - b, "{what}: a −= b agrees with a − b");

            let mut assigned = a;
            assigned *= b;
            assert_eq!(assigned, a * b, "{what}: a ×= b agrees with a × b");
        }
    }
}

/// The by-reference magma operations agree with their by-value counterparts.
///
/// [`RefAdditiveMagma::add_ref`] and [`RefMultiplicativeMagma::mul_ref`] exist for carriers that
/// are not [`Copy`], and their documentation states the law directly: "Must agree with
/// [`AdditiveMagma::add`] where both exist: `a.add_ref(&b) == a.add(b)` for a `Copy` carrier."
///
/// Nothing checked it. Mutation testing found every `add_ref` and `mul_ref` in the crate
/// replaceable with `Default::default()` — returning zero for every input — and the whole
/// workspace suite stayed green, across `i64`, `i128`, `f64`, `f32` and `GaussianRational`.
///
/// The reason the gap is structural rather than careless: a by-reference operation is written so
/// that a generic algorithm can avoid a clone, so it is reached only from code paths that are
/// generic over the *ref* traits — and the fleet's own suites all instantiate at `Copy` carriers,
/// which take the by-value path. The duplicate implementation is never the one that runs here,
/// and an implementation nothing runs is an implementation nothing checks.
pub fn ref_magma_laws<T>(xs: &[T], what: &str)
where
    T: AdditiveMagma
        + MultiplicativeMagma
        + RefAdditiveMagma
        + RefMultiplicativeMagma
        + core::fmt::Debug
        + Copy,
{
    for &a in xs {
        for &b in xs {
            assert_eq!(
                a.add_ref(&b),
                AdditiveMagma::add(a, b),
                "{what}: a.add_ref(&b) must agree with a ⊕ b"
            );
            assert_eq!(
                a.mul_ref(&b),
                MultiplicativeMagma::mul(a, b),
                "{what}: a.mul_ref(&b) must agree with a ⊗ b"
            );
        }
    }
}

/// [`ref_magma_laws`] over a carrier's own [`ConformanceSample`].
pub fn ref_magma_laws_for<T>(what: &str)
where
    T: AdditiveMagma
        + MultiplicativeMagma
        + RefAdditiveMagma
        + RefMultiplicativeMagma
        + ConformanceSample
        + core::fmt::Debug
        + Copy,
{
    ref_magma_laws(&T::sample(), what);
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

            // The division half of the compound-assignment bridge; see `ring_laws` for why this
            // whole family went unchecked. `Field` requires `DivAssign`, and `/=` could be
            // written to do nothing.
            let mut assigned = b;
            assigned /= a;
            assert_eq!(assigned, b / a, "{what}: b /= a agrees with b / a");
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

/// The monotonicity axiom: `⊗` preserves the canonical order.
///
/// `a ⪯ b` must imply both `a ⊗ c ⪯ b ⊗ c` and `c ⊗ a ⪯ c ⊗ b`. This is what makes a
/// shortest-path relaxation sound — extending two partial paths by the same edge cannot reorder
/// them — so a carrier that claims [`OrderedSemiring`] and gets it wrong produces an algorithm
/// that silently returns non-optimal answers rather than failing.
///
/// Both sides are checked because they differ for a non-commutative `⊗`.
pub fn ordered_semiring_laws<T>(xs: &[T], what: &str)
where
    T: OrderedSemiring + core::fmt::Debug + Copy,
{
    dioid_laws(xs, what);
    for &a in xs {
        for &b in xs {
            if !a.precedes(b) {
                continue;
            }
            for &c in xs {
                assert!(
                    a.mul(c).precedes(b.mul(c)),
                    "{what}: a ⪯ b must imply a⊗c ⪯ b⊗c (right-monotone)"
                );
                assert!(
                    c.mul(a).precedes(c.mul(b)),
                    "{what}: a ⪯ b must imply c⊗a ⪯ c⊗b (left-monotone)"
                );
            }
        }
    }
}

/// The six semimodule axioms: a scalar semiring acting on a value.
///
/// ```text
///   ONE · m = m                  (s ⊗ t) · m = s · (t · m)
///   ZERO · m = 0_M               s · 0_M     = 0_M
///   (s ⊕ t) · m = s·m ⊞ t·m      s · (m ⊞ n) = s·m ⊞ s·n
/// ```
///
/// Pass the same slice twice to check a semiring as a semimodule over itself, which the blanket
/// impl in this crate provides. Pass different ones to check a value type acted on by a
/// different scalar algebra.
pub fn left_semimodule_laws<S, M>(scalars: &[S], modules: &[M], what: &str)
where
    S: Semiring + core::fmt::Debug + Copy,
    M: LeftSemimodule<S> + core::fmt::Debug + Copy,
{
    for &m in modules {
        assert_eq!(m.scale(S::ONE), m, "{what}: 1 · m = m");
        assert_eq!(m.scale(S::ZERO), M::ZERO, "{what}: 0 · m = 0_M");
    }
    for &s in scalars {
        assert_eq!(M::ZERO.scale(s), M::ZERO, "{what}: s · 0_M = 0_M");
        for &m in modules {
            for &n in modules {
                assert_eq!(
                    m.add(n).scale(s),
                    m.scale(s).add(n.scale(s)),
                    "{what}: s · (m ⊞ n) = s·m ⊞ s·n"
                );
            }
            for &t in scalars {
                assert_eq!(
                    m.scale(s.add(t)),
                    m.scale(s).add(m.scale(t)),
                    "{what}: (s ⊕ t) · m = s·m ⊞ t·m"
                );
                assert_eq!(
                    m.scale(s.mul(t)),
                    m.scale(t).scale(s),
                    "{what}: (s ⊗ t) · m = s · (t · m)"
                );
            }
        }
    }
}

/// The semiring axioms for a [`SemiringWitness`], where the structure is a type parameter
/// rather than the type.
///
/// The same eight axioms [`semiring_laws`] checks, restated over associated functions. A witness
/// carries no property markers — the marker tower belongs to the element layer — so these laws
/// are the *only* thing standing between a witness and a wrong algebra, which makes running them
/// on a new witness obligatory rather than advisable.
///
/// Commutativity is not checked, for the same reason as on the element layer: a witness may name
/// a non-commutative algebra.
///
/// # The sample must be in normal form
///
/// If the witness **canonicalizes** — as [`ModP`](crate::witness::ModP) does, reducing into
/// `[0, P)` — then every value you pass must already be canonical. `ModP::<7>::add(-1, 0)` is
/// `6`, so the identity law `a ⊕ 0 = a` fails for `-1` even though `-1` and `6` are the same
/// residue: the domain is `i64` and equality is `i64`'s, which cannot see that. Pass `6`, not
/// `-1`.
///
/// This is a precondition on your sample rather than a weakness in the law. A witness is free
/// to accept non-canonical input, and the laws are free to require canonical input; what would
/// be wrong is a witness that *produced* non-canonical output.
pub fn witness_semiring_laws<W>(xs: &[W::Domain], what: &str)
where
    W: SemiringWitness,
    W::Domain: core::fmt::Debug,
{
    for &a in xs {
        assert_eq!(W::add(a, W::ZERO), a, "{what}: a ⊕ 0 = a");
        assert_eq!(W::add(W::ZERO, a), a, "{what}: 0 ⊕ a = a");
        assert_eq!(W::mul(a, W::ONE), a, "{what}: a ⊗ 1 = a");
        assert_eq!(W::mul(W::ONE, a), a, "{what}: 1 ⊗ a = a");
        assert_eq!(W::mul(a, W::ZERO), W::ZERO, "{what}: a ⊗ 0 = 0");
        assert_eq!(W::mul(W::ZERO, a), W::ZERO, "{what}: 0 ⊗ a = 0");

        for &b in xs {
            for &c in xs {
                assert_eq!(
                    W::add(W::add(a, b), c),
                    W::add(a, W::add(b, c)),
                    "{what}: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)"
                );
                assert_eq!(
                    W::mul(W::mul(a, b), c),
                    W::mul(a, W::mul(b, c)),
                    "{what}: (a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)"
                );
                assert_eq!(
                    W::mul(a, W::add(b, c)),
                    W::add(W::mul(a, b), W::mul(a, c)),
                    "{what}: a ⊗ (b ⊕ c) = a⊗b ⊕ a⊗c"
                );
                assert_eq!(
                    W::mul(W::add(b, c), a),
                    W::add(W::mul(b, a), W::mul(c, a)),
                    "{what}: (b ⊕ c) ⊗ a = b⊗a ⊕ c⊗a"
                );
            }
        }
    }
}

/// [`witness_semiring_laws`] plus idempotent `⊕`, for a [`DioidWitness`].
///
/// Idempotence is what lets a closure or fixed-point iteration terminate, so a witness that
/// claims the marker and fails this law yields an algorithm that does not converge — or worse,
/// converges to the wrong answer.
pub fn witness_dioid_laws<W>(xs: &[W::Domain], what: &str)
where
    W: DioidWitness,
    W::Domain: core::fmt::Debug,
{
    witness_semiring_laws::<W>(xs, what);
    for &a in xs {
        assert_eq!(W::add(a, a), a, "{what}: a ⊕ a = a (idempotent)");
    }
}

/// The tier laws over **generated** carriers, for a carrier author who wants the search.
///
/// # Why this one needs a trait and the others did not
///
/// `euclid-tensor` can generate a carrier through `Tensor::from_host` and `euclid-units`
/// through `Measure::from_raw`, because both contracts already contain a
/// constructor. The algebra tower contains none and should not: a [`Semiring`] is a set with two
/// operations, and "build me one from a number" is not part of being one — `Zp<7>` and a
/// polynomial ring have nothing in common to build *from*.
///
/// So the carrier supplies its own generator through `generated::GeneratedSample`. That is
/// the same
/// division of labour [`ConformanceSample`] already uses, with a generator in place of a list.
///
/// # What generation reaches that a fixed sample does not
///
/// `conformance`'s own docs argue for fixed samples and the argument is right — *about the law*.
/// An algebraic law over a small carrier either holds structurally or fails on the first awkward
/// triple, so six well-chosen values find what a thousand random ones would.
///
/// What it does not cover is the carrier's **representation**. A reduced fraction has to *stay*
/// reduced after arithmetic rather than only after construction; a Fermat inverse has to agree
/// with division at every residue, not just the four a sample lists; a Gaussian rational carries
/// two rationals that each have to stay reduced through a multiply that mixes them. Those are
/// properties of a large state space, and a fixed list of six cannot cover the interaction
/// between a reduction step and an overflow boundary.
///
/// The division of labour, then: the fixed suite proves the tiers are satisfiable and that your
/// carrier satisfies them; this one hunts the representation invariants underneath, and re-runs
/// the tier laws for free while it is there.
#[cfg(any(feature = "conformance-generated", test))]
pub mod generated {
    use euclid_stability::{Config, Gen, check, classify};

    use super::{commutative_semiring_laws, euclidean_laws, field_laws, ring_laws, semiring_laws};
    use crate::{CommutativeSemiring, EuclideanDomain, Field, Ring, Semiring};

    /// A carrier's own generator, declared once — the generated peer of
    /// [`ConformanceSample`](super::ConformanceSample).
    ///
    /// Weight it toward the values that break things rather than uniformly: `0` and `±1` are the
    /// identities and the annihilator, so they are what a law is most likely to get wrong, and a
    /// uniform draw over a wide range would almost never produce them.
    ///
    /// Bound the magnitudes by what your arithmetic can hold. The laws run over *triples* —
    /// distributivity alone forms `(a ⊕ b) ⊗ c` — so a carrier backed by a fixed-width integer
    /// should keep a triple's worst case well inside it. A debug-build overflow panics, and that
    /// panic is indistinguishable from a law failure at the call site.
    pub trait GeneratedSample: Sized {
        /// Values to draw from.
        fn generator() -> Gen<Self>;

        /// How many cases to run. Defaulted at 300: enough that the shrinker has something to
        /// work with, few enough that every tier law over a triple stays quick.
        fn cases() -> usize {
            300
        }
    }

    /// Three is the minimum that exercises a three-operand law at all; six matches what a fixed
    /// sample uses, and the laws are `O(n³)`.
    fn sample<T: GeneratedSample + 'static>() -> Gen<Vec<T>> {
        euclid_stability::generate::vec(T::generator(), 3, 6)
    }

    /// `pub(crate)` only so the vacuity suite can assert the contract below directly.
    ///
    /// The thing worth pinning is that a carrier's declared [`GeneratedSample::cases`] is what
    /// the search actually runs. Replacing this body with `Config::default()` silently drops
    /// that — `Config`'s own default is 256 cases — and no behavioural test separates the two
    /// without counting draws, which would be a statistical assertion about a generator rather
    /// than a statement about this function. One `assert_eq!` on the field says it exactly.
    pub(crate) fn settings<T: GeneratedSample>() -> Config {
        Config::default().cases(T::cases())
    }

    /// [`semiring_laws`] over generated elements — the trunk every other tier refines.
    pub fn semiring_laws_generated<T>(what: &str)
    where
        T: Semiring + GeneratedSample + core::fmt::Debug + Copy + 'static,
    {
        check(what, &settings::<T>(), &sample::<T>(), |xs| {
            classify!("contains zero", xs.contains(&T::ZERO));
            semiring_laws(&xs, what);
            Ok(())
        });
    }

    /// [`commutative_semiring_laws`] over generated elements.
    ///
    /// Separate from [`semiring_laws_generated`] rather than folded into it, because commutative
    /// `⊗` is a strictly narrower claim — matrix multiplication is a semiring and is not
    /// commutative, so a single entry point would exclude it.
    pub fn commutative_semiring_laws_generated<T>(what: &str)
    where
        T: CommutativeSemiring + GeneratedSample + core::fmt::Debug + Copy + 'static,
    {
        check(what, &settings::<T>(), &sample::<T>(), |xs| {
            classify!("contains zero", xs.contains(&T::ZERO));
            commutative_semiring_laws(&xs, what);
            Ok(())
        });
    }

    /// [`ring_laws`] over generated elements — the additive-inverse tier.
    pub fn ring_laws_generated<T>(what: &str)
    where
        T: Ring + GeneratedSample + core::fmt::Debug + Copy + 'static,
    {
        check(what, &settings::<T>(), &sample::<T>(), |xs| {
            classify!("contains zero", xs.contains(&T::ZERO));
            ring_laws(&xs, what);
            Ok(())
        });
    }

    /// [`field_laws`] over generated elements.
    ///
    /// The division laws skip `ZERO` internally, so a generator that produces it
    /// is not a problem — and should, since a field law that silently never divided by anything
    /// awkward is the one most worth running.
    pub fn field_laws_generated<T>(what: &str)
    where
        T: Field + GeneratedSample + core::fmt::Debug + Copy + 'static,
    {
        check(what, &settings::<T>(), &sample::<T>(), |xs| {
            classify!("contains zero", xs.contains(&T::ZERO));
            field_laws(&xs, what);
            Ok(())
        });
    }

    /// [`euclidean_laws`] over generated elements — division with remainder.
    pub fn euclidean_laws_generated<T>(what: &str)
    where
        T: EuclideanDomain + GeneratedSample + core::fmt::Debug + Copy + 'static,
    {
        check(what, &settings::<T>(), &sample::<T>(), |xs| {
            classify!("contains zero", xs.contains(&T::ZERO));
            euclidean_laws(&xs, what);
            Ok(())
        });
    }
}
