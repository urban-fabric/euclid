//! Conformance suites — the quantity contracts as functions you can check a carrier against.
//!
//! This crate ships no concrete unit, so every one of these traits is implemented by someone
//! else. A carrier that wraps the wrong field, loses the sign in `neg`, or whose `clamp`
//! disagrees with `min`/`max` compiles and type-checks exactly like a correct one, and then
//! quietly computes different answers from every other carrier in the fleet.
//!
//! # Mask-valued, not `bool`-valued
//!
//! [`ScalarQuantity`]'s predicates return a [`Mask`], not a host `bool`, so these suites compare
//! masks against [`Element::one`]/[`Element::zero`] rather than branching on them. That is what
//! lets the same laws check an expression or tensor carrier as well as an `f64` one — nothing
//! here needs a host boolean, and a deferred carrier is not excluded.
//!
//! # Laws are relations, not reference values
//!
//! Nothing below asserts a computed number. `clamp(x, lo, hi) == max(lo, min(x, hi))` is the
//! *definition* of clamping, and `a.add(b).sub(b) == a` is what makes a quantity linear — both
//! hold for any correct carrier regardless of representation, and neither can be mistyped into
//! passing. Where a law would need an ordering (`floor(x) <= x`) it is restated through
//! `min`/`max`, which the contract already provides, so no extra bound is needed.
//!
//! # Availability
//!
//! Gated behind the `conformance` feature, so a consumer that only writes formulas pays nothing
//! for it, and always compiled under `cfg(test)` so this crate's own suites exercise the same
//! code a carrier author calls.
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
//! You supply them, and they should be **finite** — the predicate laws assert finiteness
//! holds, which is a property of your sample rather than of the contract. Four or five values
//! including a zero and a negative is plenty; the three-operand laws run over every triple.

//! # The split
//!
//! This module passed the crate's 400-line budget, so each contract's laws now sit beside the trait
//! they check and this file is the index: the carrier-supplied sample, and the re-exports that keep
//! every public path exactly where it was.

mod affine;
mod discrete;
mod measure;
mod scalar;

/// The same laws over **generated** carriers, for a carrier author who wants the search rather
/// than a fixed five-value sample.
///
/// # What generation reaches that a fixed sample does not
///
/// The tier laws themselves are the kind a fixed sample serves well — they either hold
/// structurally or fail on the first awkward triple. The reason to generate anyway is the
/// **default method bodies**.
///
/// [`ScalarQuantity`] and [`LinearQuantity`] supply `min`, `max`, `clamp`, `floor`, `ceil`,
/// `round`, `trunc`, `approx_eq`, `add`, `sub`, `neg` and `scale` as defaults, so one body serves
/// every carrier in every consuming crate — each one a single point of failure. Those bodies
/// branch on *orderings* and *signs*, which is a combinatorial space rather than a structural
/// one:
///
/// - `clamp(x, lo, hi)` has six orderings of its three arguments, and five fixed values in one
///   fixed order reach a handful of them;
/// - `floor`, `ceil` and `trunc` agree on positives and differ on negatives, so a mostly-positive
///   sample would pass a `trunc` written as `floor`;
/// - `round` has to choose a direction at exactly `±0.5`.
///
/// The generator draws signs and half-integers deliberately for those three reasons, and the
/// coverage floors require that mixed-sign samples actually appeared rather than hoping they did.
///
/// # No new trait method was needed
///
/// The carrier is built through [`Measure::from_raw`], which the contract already requires. The
/// `Repr = f64` bound is the one restriction: `Repr` is any [`euclid_scalar::Element`],
/// and euclid cannot generate an arbitrary one — an integer-backed carrier should run the fixed
/// suite, or pass its own values to the slice-taking laws above.
#[cfg(any(feature = "conformance-generated", test))]
pub mod generated;

// Re-exported so every public path this module has ever had still resolves: the
// split is an internal rearrangement and no carrier should have to notice it.
pub use affine::temperature_laws;
pub use discrete::{discrete_laws, discrete_laws_for, quantity_laws, quantity_laws_for};
pub use measure::{
    measure_laws, measure_laws_for, operator_bridge_laws, operator_bridge_laws_for, scaling_laws,
    scaling_laws_for,
};
pub use scalar::{linear_laws, scalar_laws, scalar_laws_for};

use crate::{DiscreteQuantity, LinearQuantity, Mask, Measure, Quantity, ScalarQuantity};
use euclid_scalar::{Element, Real};

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
    /// The values this carrier should be checked over. Must be **finite** — the predicate laws
    /// assert finiteness holds.
    fn sample() -> Vec<Self>;

    /// The tolerance for the two inexact linear laws, as a quantity in this unit.
    ///
    /// The add/sub inverse and associativity are not exact when `Repr` is not — subtracting
    /// `100.0` from `-9.8 + 100.0` gives `-9.799999999999997`, not `-9.8`. An exact carrier can
    /// return its zero to recover the strict reading.
    fn tolerance() -> Self;
}
