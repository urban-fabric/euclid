//! The idempotent and ordered tiers: dioid, the star operation, and the canonical order.
//!
//! A dioid's `⊕` is idempotent, which is what makes its natural order a lattice rather than
//! merely a relation — so the order laws belong here rather than beside the semiring. `star` is
//! the closure those orders make meaningful: the least fixed point of `x ↦ 1 ⊕ ax`.

use super::*;

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

/// [`dioid_laws`] over a carrier's own [`ConformanceSample`].
pub fn dioid_laws_for<T>(what: &str)
where
    T: Dioid + ConformanceSample + core::fmt::Debug + Copy,
{
    dioid_laws(&T::sample(), what);
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
