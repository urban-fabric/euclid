//! Star, canonical-order, ordered-semiring and semimodule rejection controls.
//!
//! Four suites that mutation testing found replaceable with an empty body. Each one below gets a
//! carrier broken in exactly one law, and the suite is required to panic.
//!
//! # Two of these laws cannot be broken from outside, and that is worth knowing
//!
//! Writing this file turned up a structural limit on what a vacuity carrier can reach, and it
//! explains why these four survived longer than the semiring suites did.
//!
//! **`CanonicalOrder::precedes` is not overridable.** The trait is blanket-implemented —
//! `impl<T: AdditiveSemilattice> CanonicalOrder for T {}` — so every carrier gets the default
//! body `self.add(other) == self`. [`conformance::canonical_order_laws`]'s central assertion,
//! `a.precedes(b) == (a.add(b) == a)`, is therefore comparing the default with its own
//! definition: a *tautology*, true for every carrier that can exist. It is not dead code — the
//! doc comment says a carrier "can still break it by overriding", which was true of an earlier
//! design — but today the falsifiable content of that suite is reflexivity, antisymmetry and
//! transitivity, which are properties of `⊕` and are reachable.
//!
//! **Monotone `⊗` is implied by distributivity.** `ordered_semiring_laws` asserts that `a ⪯ b`
//! gives `a⊗c ⪯ b⊗c`, and that follows from the right-distributivity
//! [`conformance::semiring_laws`] already checks on the same sample:
//!
//! ```text
//! a ⪯ b  means  a ⊕ b = a
//! a⊗c = (a ⊕ b)⊗c = a⊗c ⊕ b⊗c          (right distributivity)
//! so     a⊗c ⊕ b⊗c = a⊗c,  i.e.  a⊗c ⪯ b⊗c
//! ```
//!
//! and symmetrically on the left. So no carrier can pass `dioid_laws` and fail the monotonicity
//! assertions — they are redundant given the tier beneath them. What *is* killable is the whole
//! body, because an empty body also skips the `dioid_laws` call at the top, and that is what the
//! test here exploits: a carrier broken in a dioid law must still be rejected.
//!
//! Neither observation is a reason to delete anything. Both suites are public API that a
//! consumer runs, and a redundant assertion that names the axiom it checks is better
//! documentation than a comment. But a reader chasing a survivor in either function should know
//! the ceiling before trying to write a carrier that cannot exist.

use super::*;
use crate::conformance::{
    canonical_order_laws, dioid_laws, left_semimodule_laws, ordered_semiring_laws, star_laws,
};

// ===== A correct min-plus dioid, as the control ============================

/// Min-plus (tropical) over `i64`: `⊕` is `min`, `⊗` is `+`, and `ZERO` is `+∞`.
///
/// A genuine dioid, and the reference every broken carrier below is a one-law edit of. `ZERO` is
/// `i64::MAX` standing for infinity, which `mul` has to special-case — `i64::MAX + (-3)` is not
/// `i64::MAX`, so without the guard the annihilation law `a ⊗ 0 = 0` would fail for negative
/// values and the carrier would be broken in a second place by accident.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Tropical(i64);

impl AdditiveMagma for Tropical {
    fn add(self, rhs: Self) -> Self {
        Self(self.0.min(rhs.0))
    }
}
impl MultiplicativeMagma for Tropical {
    fn mul(self, rhs: Self) -> Self {
        if self == Self::ZERO || rhs == Self::ZERO {
            Self::ZERO
        } else {
            Self(self.0 + rhs.0)
        }
    }
}
impl HasZero for Tropical {
    const ZERO: Self = Self(i64::MAX);
}
impl HasOne for Tropical {
    const ONE: Self = Self(0);
}
impl AssociativeAddition for Tropical {}
impl CommutativeAddition for Tropical {}
impl IdempotentAddition for Tropical {}
impl AssociativeMultiplication for Tropical {}
impl CommutativeMultiplication for Tropical {}
impl Distributive for Tropical {}
impl ZeroAnnihilatesMultiplication for Tropical {}
impl MonotoneMultiplication for Tropical {}
impl ConformanceSample for Tropical {
    fn sample() -> Vec<Self> {
        vec![Self::ZERO, Self::ONE, Self(1), Self(4)]
    }
}

#[test]
fn the_order_and_star_suites_accept_a_correct_dioid() {
    // Without this, every rejection test below is satisfiable by a suite that always panics.
    let sample = Tropical::sample();
    assert!(
        rejects(|| dioid_laws(&sample, "Tropical")).is_none(),
        "dioid_laws must accept min-plus"
    );
    assert!(
        rejects(|| canonical_order_laws(&sample, "Tropical")).is_none(),
        "canonical_order_laws must accept min-plus"
    );
    assert!(
        rejects(|| ordered_semiring_laws(&sample, "Tropical")).is_none(),
        "ordered_semiring_laws must accept min-plus"
    );
}

// ===== A star that is not a fixed point ====================================

/// Min-plus with `star(a) = a`, where the fixed-point identity requires `ONE`.
///
/// For a min-plus carrier with non-negative weights no cycle improves on the empty walk, so
/// `a* = ONE ⊕ a ⊕ a² ⊕ … = 0`. Returning `a` instead satisfies the identity at `a = ONE` and
/// nowhere else, which is exactly the shape of a real mistake: it looks right on the one value
/// anybody checks by hand.
#[derive(Clone, Copy, PartialEq, Debug)]
struct SelfStar(i64);

impl AdditiveMagma for SelfStar {
    fn add(self, rhs: Self) -> Self {
        Self(self.0.min(rhs.0))
    }
}
impl MultiplicativeMagma for SelfStar {
    fn mul(self, rhs: Self) -> Self {
        if self == Self::ZERO || rhs == Self::ZERO {
            Self::ZERO
        } else {
            Self(self.0 + rhs.0)
        }
    }
}
impl HasZero for SelfStar {
    const ZERO: Self = Self(i64::MAX);
}
impl HasOne for SelfStar {
    const ONE: Self = Self(0);
}
impl AssociativeAddition for SelfStar {}
impl CommutativeAddition for SelfStar {}
impl IdempotentAddition for SelfStar {}
impl AssociativeMultiplication for SelfStar {}
impl CommutativeMultiplication for SelfStar {}
impl Distributive for SelfStar {}
impl ZeroAnnihilatesMultiplication for SelfStar {}
// Deliberately *not* `Absorptive`: that marker blanket-implements `StarSemiring` with the
// correct `ONE`, and claiming it would make this carrier right by construction.
impl StarSemiring for SelfStar {
    fn star(self) -> Self {
        self
    }
}

#[test]
fn the_star_laws_reject_a_star_that_is_not_a_fixed_point() {
    let sample = [SelfStar::ONE, SelfStar(1), SelfStar(4)];
    let message = rejects(|| star_laws(&sample, "SelfStar"))
        .expect("star_laws must reject a carrier where a* != 1 + a*a*");
    assert!(
        message.contains("a* = 1 ⊕ a ⊗ a*"),
        "the failure must name the axiom that broke: {message}"
    );
}

// ===== Addition that is not idempotent, while claiming to be ===============

/// Plain integer addition claiming [`IdempotentAddition`], so `a ⊕ a = 2a ≠ a`.
///
/// This reaches three suites at once, and through three different assertions:
///
/// - `dioid_laws` on its own idempotence check;
/// - `canonical_order_laws` on **reflexivity**, since `a ⪯ a` unfolds to `a ⊕ a = a`;
/// - `ordered_semiring_laws` through the `dioid_laws` call it opens with.
///
/// It is a plausible mistake rather than a contrived one: `Arithmetic` is a perfectly good
/// semiring, and the only thing wrong here is the claim that its `⊕` is a meet.
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
impl IdempotentAddition for NotIdempotent {}
impl AssociativeMultiplication for NotIdempotent {}
impl CommutativeMultiplication for NotIdempotent {}
impl Distributive for NotIdempotent {}
impl ZeroAnnihilatesMultiplication for NotIdempotent {}
impl MonotoneMultiplication for NotIdempotent {}

#[test]
fn the_canonical_order_laws_reject_a_non_idempotent_addition() {
    let sample = [NotIdempotent(0), NotIdempotent(1), NotIdempotent(3)];
    let message = rejects(|| canonical_order_laws(&sample, "NotIdempotent"))
        .expect("canonical_order_laws must reject a carrier whose order is not reflexive");
    assert!(
        message.contains("a ⪯ a") || message.contains("a ⊕ b = a"),
        "the failure must name the order property that broke: {message}"
    );
}

#[test]
fn the_ordered_semiring_laws_reject_what_their_dioid_tier_rejects() {
    // The monotonicity assertions are unreachable (see the module header), so this pins the
    // delegation instead — which is what the empty-body mutant actually removed.
    let sample = [NotIdempotent(0), NotIdempotent(1), NotIdempotent(3)];
    let message = rejects(|| ordered_semiring_laws(&sample, "NotIdempotent"))
        .expect("ordered_semiring_laws must reject a carrier that is not a dioid");
    assert!(
        message.contains("idempotent"),
        "the failure must come from the dioid tier it delegates to: {message}"
    );
}

// ===== A scalar action that is not additive in the scalar ==================

/// A pair under component-wise addition, acted on by `i64`.
///
/// The correct action, as the control for [`SkewPair`].
#[derive(Clone, Copy, PartialEq, Debug)]
struct Pair(i64, i64);

impl AdditiveMagma for Pair {
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}
impl HasZero for Pair {
    const ZERO: Self = Self(0, 0);
}
impl AssociativeAddition for Pair {}
impl CommutativeAddition for Pair {}
impl LeftSemimodule<i64> for Pair {
    fn scale(self, scalar: i64) -> Self {
        Self(self.0 * scalar, self.1 * scalar)
    }
}

/// The same pair, but the second component is scaled by `s²`.
///
/// Chosen because it breaks **exactly one** of the six axioms. `1 · m = m`, `0 · m = 0_M` and
/// `s · 0_M = 0_M` all still hold because `1² = 1` and `0² = 0`; additivity in the *module*
/// argument holds because each component is still linear in `m`; and `(s ⊗ t) · m = s · (t · m)`
/// holds because `(st)² = s²t²`. Only additivity in the **scalar** fails, since
/// `(s + t)² ≠ s² + t²` — so a suite that checked five of six axioms would still pass this
/// carrier, which is the discrimination a vacuity control is for.
#[derive(Clone, Copy, PartialEq, Debug)]
struct SkewPair(i64, i64);

impl AdditiveMagma for SkewPair {
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}
impl HasZero for SkewPair {
    const ZERO: Self = Self(0, 0);
}
impl AssociativeAddition for SkewPair {}
impl CommutativeAddition for SkewPair {}
impl LeftSemimodule<i64> for SkewPair {
    fn scale(self, scalar: i64) -> Self {
        Self(self.0 * scalar, self.1 * scalar * scalar)
    }
}

#[test]
fn the_semimodule_laws_accept_a_correct_action() {
    let scalars = [0i64, 1, 2, 3];
    let modules = [Pair(0, 0), Pair(1, 0), Pair(0, 1), Pair(2, 3)];
    assert!(
        rejects(|| left_semimodule_laws(&scalars, &modules, "Pair")).is_none(),
        "left_semimodule_laws must accept a component-wise scalar action"
    );
}

#[test]
fn the_semimodule_laws_reject_an_action_that_is_not_additive_in_the_scalar() {
    let scalars = [0i64, 1, 2, 3];
    let modules = [
        SkewPair(0, 0),
        SkewPair(1, 0),
        SkewPair(0, 1),
        SkewPair(2, 3),
    ];
    let message = rejects(|| left_semimodule_laws(&scalars, &modules, "SkewPair"))
        .expect("left_semimodule_laws must reject an action where (s + t)*m != s*m + t*m");
    assert!(
        message.contains("(s ⊕ t) · m"),
        "the failure must name the axiom that broke: {message}"
    );
}
