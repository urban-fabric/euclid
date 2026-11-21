//! The witness suites: laws checked against a carrier's own declared structure.
//!
//! A [`SemiringWitness`] is a carrier's claim about which identities and inverses it has. These
//! suites take the claim and hold the carrier to it, which is a different question from the tier
//! laws: those ask whether the algebra is sound, these ask whether the carrier described itself
//! correctly.

use super::*;

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
