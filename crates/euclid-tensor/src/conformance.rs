//! Conformance suites — the structural half of the [`Tensor`] contract, as functions you can
//! check a backend against.
//!
//! euclid-tensor ships no tensor, so every implementation of this trait lives somewhere else.
//! A backend whose `transpose_axes` is not an involution, whose `reshape` to its own shape is
//! not the identity, or whose `slice_in_dim` over the full extent drops an element compiles and
//! type-checks exactly like a correct one — and then produces wrong answers in a kernel written
//! once against the contract.
//!
//! # What these suites assert, and what they deliberately do not
//!
//! The trait docs are explicit that *"axis naming, mask representation, broadcast/gather/scatter
//! details, and precision policy are outside this contract"*. Conventions the contract leaves
//! open are not laws, so nothing here asserts one. In particular:
//!
//! - **Reductions are not checked.** Whether `sum(&[])` reduces over all axes or none, and
//!   whether a reduced axis is removed or kept at length 1, is a convention each backend picks.
//!   A suite that asserted either would fail a correct backend that chose the other.
//! - **Broadcast, gather and scatter semantics are not checked**, for the same reason.
//! - **Numeric results are not checked.** That is `euclid-scalar`'s subject; here the question
//!   is whether the *structure* holds.
//!
//! What is left is the part the contract does fix: shape algebra, and the operations that must
//! be identities or involutions whatever the backend. Those are enough to catch the errors that
//! actually happen — a transposed axis order, an off-by-one extent, a lost rank.
//!
//! # Availability
//!
//! Gated behind the `conformance` feature, so a consumer pays nothing unless it asks, and always
//! compiled under `cfg(test)` so this crate's own suites exercise the same code a backend calls.
//!
//! The `generated` submodule sits behind a **second** feature, `conformance-generated`, which
//! enables this one and additionally pulls the property engine. The division: these functions
//! are the contract, so `conformance` resolves no engine at all and is what a consumer binding
//! the laws should take; the generated search is tooling for locating where a carrier fails
//! them, and costs a dependency.
//!
//! These docs are the module's own (`//!`), with no outer `///` on the `mod` declaration: the
//! two forms are concatenated, and an outer comment resolves intra-doc links in the *parent's*
//! scope, so a link to a function in here would silently fail from there.
//!
//! # Samples
//!
//! Supply tensors of **rank 1 or higher**, with at least one of rank 2 or more so the transpose
//! laws are exercised. A rank-0 sample is accepted and simply skips the laws that need an axis.

//! # The split
//!
//! This module passed the crate's 400-line budget, so each surface's laws now sit beside the trait
//! they check and this file is the index: the carrier-supplied sample, and the re-exports that keep
//! every public path exactly where it was.

mod shape;
mod structural;

/// The same laws over **generated shapes**, for a backend that wants the search rather than its
/// own fixed sample.
///
/// # What this ships that a fixed sample cannot
///
/// A [`ConformanceSample`] proves the laws are *satisfiable* and reaches every rank they need.
/// What it cannot do is vary the **extents**, and the extents are where a structural bug hides:
///
/// - **Repeated extents.** At `[2, 3, 4]` a permutation applied in the wrong direction produces
///   the wrong *shape* and `shape()` alone catches it. At `[3, 3, 3]` the wrong direction
///   produces the **right shape and the wrong data**, so only the values catch it. That is the
///   one case where a backend can be wrong and still look right to every shape assertion.
/// - **Unit axes.** An extent of `1` makes transpose and reverse no-ops along that axis, so a
///   backend that drops such an axis — or computes its stride as `0` — passes an involution law
///   while corrupting the rank.
/// - **Rank 4 and up.** The laws are written for arbitrary rank, and a sample that stops at 3
///   leaves the generic path type-checked but never executed.
///
/// That last one is not hypothetical. [`structural_laws`] built a malformed 3-cycle at rank ≥ 4
/// for as long as euclid's own reference sample stopped at rank 3, and the defect surfaced the
/// first time a generated shape reached rank 4. A backend whose own sample stops where euclid's
/// did inherits exactly that blind spot, which is the reason to ship the search rather than
/// describe it.
///
/// # No new trait method was needed
///
/// The carrier is built through [`Tensor::from_host`], which already takes row-major host data
/// and a shape — so euclid generates the shape and the data and the backend supplies nothing. The
/// `Real` bound comes from `from_host` itself and is why this covers real tensors and not boolean
/// ones; a boolean carrier should run the fixed suite.
#[cfg(any(feature = "conformance-generated", test))]
pub mod generated;

// Re-exported so every public path this module has ever had still resolves: the
// split is an internal rearrangement and no carrier should have to notice it.
pub use shape::shape_laws;
pub use structural::structural_laws;

use crate::Tensor;

/// A carrier's own curated sample, declared once.
///
/// euclid's answer to proptest's `Arbitrary`, with the generator replaced by a fixed list. A
/// carrier says which values stress *it*, and every suite in this module then draws from the
/// same place rather than each call site inventing a sample and getting it slightly wrong.
///
/// Fixed rather than generated on purpose: a generator buys input breadth at the cost of
/// reproducibility, turning a failure into a seed to recover rather than a value in the source.
/// Keep samples small — the multi-operand laws run over every pair or triple, so `n` values
/// cost `n²` or `n³` cases per law.
pub trait ConformanceSample: Sized {
    /// Tensors to check. Include one of **rank 2 or more** so the transpose laws are
    /// exercised, and one of rank 3 or more so the 3-cycle law is — a swap cannot distinguish
    /// a permutation from its inverse.
    fn sample() -> Vec<Self>;
}

/// [`tensor_laws`] over a carrier's own [`ConformanceSample`].
pub fn tensor_laws_for<T>(what: &str)
where
    T: Tensor + ConformanceSample + PartialEq + core::fmt::Debug,
{
    tensor_laws(&T::sample(), what);
}

/// Both structural suites, for a carrier that implements the whole core contract.
pub fn tensor_laws<T>(xs: &[T], what: &str)
where
    T: Tensor + PartialEq + core::fmt::Debug,
{
    shape_laws(xs, what);
    structural_laws(xs, what);
}
