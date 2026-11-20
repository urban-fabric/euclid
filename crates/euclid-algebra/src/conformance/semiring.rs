//! The semiring tier: two monoids, and the distributivity that ties them together.
//!
//! The base of the hierarchy, so these laws are the ones every other suite in this module
//! re-checks by inclusion. Commutativity of `⊗` is separate because it is exactly what divides
//! the semirings that behave like numbers from the ones that behave like matrices.

use super::*;

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

/// [`semiring_laws`] over a carrier's own [`ConformanceSample`].
pub fn semiring_laws_for<T>(what: &str)
where
    T: Semiring + ConformanceSample + core::fmt::Debug + Copy,
{
    semiring_laws(&T::sample(), what);
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

/// [`commutative_semiring_laws`] over a carrier's own [`ConformanceSample`].
pub fn commutative_semiring_laws_for<T>(what: &str)
where
    T: CommutativeSemiring + ConformanceSample + core::fmt::Debug + Copy,
{
    commutative_semiring_laws(&T::sample(), what);
}
