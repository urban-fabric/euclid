//! The [`Boolean`] and [`Compare`] contracts — a carrier's lattice and its masks.
//!
//! Together because a mask *is* the boolean surface of a carrier: these laws check that the
//! lattice operations agree with each other and that selection honours them. Kept away from the
//! `Real` laws because nothing here needs an arithmetic at all — a carrier can get every mask
//! law right and every number wrong, and the division says so.

use super::*;

/// The boolean algebra laws for a [`Boolean`] carrier.
///
/// Exact, since nothing here rounds. De Morgan is the pair worth having: a carrier that
/// implements `or` as `not(and(not, not))` and gets a negation wrong still satisfies
/// idempotence and identity, and fails only here.
pub fn boolean_laws<B>(xs: &[B], what: &str)
where
    B: Boolean + PartialEq + core::fmt::Debug,
{
    for a in xs {
        // identity and annihilation
        assert_eq!(a.and(&B::one()), *a, "{what}: a ∧ 1 = a");
        assert_eq!(a.or(&B::zero()), *a, "{what}: a ∨ 0 = a");
        assert_eq!(a.and(&B::zero()), B::zero(), "{what}: a ∧ 0 = 0");
        assert_eq!(a.or(&B::one()), B::one(), "{what}: a ∨ 1 = 1");
        // idempotence and involution
        assert_eq!(a.and(a), *a, "{what}: a ∧ a = a");
        assert_eq!(a.or(a), *a, "{what}: a ∨ a = a");
        assert_eq!(a.not().not(), *a, "{what}: ¬¬a = a");
        // complement
        assert_eq!(a.and(&a.not()), B::zero(), "{what}: a ∧ ¬a = 0");
        assert_eq!(a.or(&a.not()), B::one(), "{what}: a ∨ ¬a = 1");
        // xor against its definition
        assert_eq!(a.xor(a), B::zero(), "{what}: a ⊕ a = 0");
        assert_eq!(a.xor(&B::zero()), *a, "{what}: a ⊕ 0 = a");

        for b in xs {
            // commutativity
            assert_eq!(a.and(b), b.and(a), "{what}: ∧ commutes ({a:?}, {b:?})");
            assert_eq!(a.or(b), b.or(a), "{what}: ∨ commutes ({a:?}, {b:?})");
            // De Morgan, both directions
            assert_eq!(
                a.and(b).not(),
                a.not().or(&b.not()),
                "{what}: ¬(a ∧ b) = ¬a ∨ ¬b ({a:?}, {b:?})"
            );
            assert_eq!(
                a.or(b).not(),
                a.not().and(&b.not()),
                "{what}: ¬(a ∨ b) = ¬a ∧ ¬b ({a:?}, {b:?})"
            );
            // xor in terms of and/or/not
            assert_eq!(
                a.xor(b),
                a.or(b).and(&a.and(b).not()),
                "{what}: a ⊕ b = (a ∨ b) ∧ ¬(a ∧ b) ({a:?}, {b:?})"
            );
        }
    }
}

/// [`boolean_laws`] over a carrier's own [`ConformanceSample`].
pub fn boolean_laws_for<B>(what: &str)
where
    B: Boolean + ConformanceSample + PartialEq + core::fmt::Debug,
{
    boolean_laws(&B::sample(), what);
}

/// The comparison and selection laws for a [`Compare`] carrier.
///
/// Exact. Every law relates one comparison to another or to `select`, so none of them rounds:
///
/// - `lt`/`ge` and `gt`/`le` and `eq`/`ne` are complementary
/// - `eq` is reflexive, `lt` and `gt` are not
/// - `le` is `lt ∨ eq`
/// - `select` with an all-true mask takes the true branch, all-false the false branch
///
/// That last pair is the one that catches a swapped `select`, which is otherwise invisible —
/// a carrier with its branches reversed type-checks everywhere and silently inverts every
/// conditional written against the contract.
pub fn mask_laws<T>(xs: &[T], what: &str)
where
    T: Compare + PartialEq + core::fmt::Debug,
    T::Mask: PartialEq + core::fmt::Debug,
{
    let t = <T::Mask as Element>::one();
    let f = <T::Mask as Element>::zero();

    for a in xs {
        assert_eq!(Compare::eq(a, a), t, "{what}: a = a ({a:?})");
        assert_eq!(Compare::ne(a, a), f, "{what}: ¬(a ≠ a) ({a:?})");
        assert_eq!(a.lt(a), f, "{what}: ¬(a < a) ({a:?})");
        assert_eq!(a.gt(a), f, "{what}: ¬(a > a) ({a:?})");
        assert_eq!(a.le(a), t, "{what}: a ≤ a ({a:?})");
        assert_eq!(a.ge(a), t, "{what}: a ≥ a ({a:?})");

        for b in xs {
            // complementary pairs
            assert_eq!(
                a.lt(b),
                a.ge(b).not(),
                "{what}: (a < b) = ¬(a ≥ b) ({a:?}, {b:?})"
            );
            assert_eq!(
                a.gt(b),
                a.le(b).not(),
                "{what}: (a > b) = ¬(a ≤ b) ({a:?}, {b:?})"
            );
            assert_eq!(
                Compare::eq(a, b),
                Compare::ne(a, b).not(),
                "{what}: (a = b) = ¬(a ≠ b) ({a:?}, {b:?})"
            );
            // le decomposes
            assert_eq!(
                a.le(b),
                a.lt(b).or(&Compare::eq(a, b)),
                "{what}: (a ≤ b) = (a < b) ∨ (a = b) ({a:?}, {b:?})"
            );
            // antisymmetry of the strict order
            assert_eq!(
                a.lt(b).and(&b.lt(a)),
                f,
                "{what}: a < b and b < a cannot both hold ({a:?}, {b:?})"
            );

            // select chooses, and chooses the right way round
            assert_eq!(
                &T::select(&t, a.clone(), b.clone()),
                a,
                "{what}: select(true, a, b) must be a ({a:?}, {b:?})"
            );
            assert_eq!(
                &T::select(&f, a.clone(), b.clone()),
                b,
                "{what}: select(false, a, b) must be b ({a:?}, {b:?})"
            );
            // the alias must agree
            assert_eq!(
                T::if_then_else(&t, a.clone(), b.clone()),
                T::select(&t, a.clone(), b.clone()),
                "{what}: if_then_else must agree with select"
            );
        }
    }
}

/// [`mask_laws`] over a carrier's own [`ConformanceSample`].
pub fn mask_laws_for<T>(what: &str)
where
    T: Compare + ConformanceSample + PartialEq + core::fmt::Debug,
    T::Mask: PartialEq + core::fmt::Debug,
{
    mask_laws(&T::sample(), what);
}
