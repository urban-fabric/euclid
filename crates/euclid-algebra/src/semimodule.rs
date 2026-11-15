//! Semimodules — the action of a scalar semiring on a value.
//!
//! A **left semimodule** over a semiring `S` is a commutative additive monoid `(M, ⊞, 0_M)`
//! together with a scalar action `S × M → M` satisfying:
//!
//! ```text
//!   ONE · m = m                  (s ⊗ t) · m = s · (t · m)
//!   ZERO · m = 0_M               s · 0_M     = 0_M
//!   (s ⊕ t) · m = s·m ⊞ t·m      s · (m ⊞ n) = s·m ⊞ s·n
//! ```
//!
//! This is the tier that sits between a semiring and a linear-algebra kernel. A matrix–vector
//! product is the action of a matrix semiring on a vector, and a relaxation step in a
//! shortest-path algorithm is one application of it — so the *law* belongs here with the rest of
//! the tower, even though the kernels that exploit it belong with a backend.
//!
//! [`LeftSemimodule`] and [`RightSemimodule`] differ only in which side `⊗` multiplies on, and
//! coincide for a commutative semiring. Every semiring is a semimodule over itself, which the
//! blanket impls below provide: `scale` is just `⊗`.

use super::{CommutativeAdditiveMonoid, Semiring};

/// A commutative additive monoid carrying a left scalar action by `S`.
///
/// Implemented automatically for every [`Semiring`] over itself. Implement it by hand for a
/// value type that is acted on by a *different* scalar algebra — a vector over a min-plus
/// semiring, say.
pub trait LeftSemimodule<S: Semiring>: CommutativeAdditiveMonoid {
    /// `scalar · self`.
    #[must_use]
    fn scale(self, scalar: S) -> Self;
}

/// A commutative additive monoid carrying a right scalar action by `S`.
///
/// The mirror of [`LeftSemimodule`]. The two coincide when `S` is commutative; they differ for a
/// matrix or lexicographic semiring, where side matters.
pub trait RightSemimodule<S: Semiring>: CommutativeAdditiveMonoid {
    /// `self · scalar`.
    #[must_use]
    fn scale(self, scalar: S) -> Self;
}

impl<S: Semiring> LeftSemimodule<S> for S {
    #[inline]
    fn scale(self, scalar: S) -> Self {
        scalar.mul(self)
    }
}

impl<S: Semiring> RightSemimodule<S> for S {
    #[inline]
    fn scale(self, scalar: S) -> Self {
        self.mul(scalar)
    }
}
