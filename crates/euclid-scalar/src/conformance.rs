//! Conformance suites — the scalar contracts as functions you can check a carrier against.
//!
//! This is the crate every other one is written over, so a carrier that honours it slightly
//! differently is the most expensive kind of disagreement available: a formula bounded on
//! [`Real`] runs on an `f64`, a graph node and a tensor element, and if one of those three
//! implements `select` with its branches swapped, the same source produces three answers and
//! two of them are wrong.
//!
//! ```
//! # use euclid_scalar::conformance;
//! conformance::boolean_laws(&[false, true], "bool");
//! conformance::mask_laws(&[0.0_f64, 1.0, -2.5], "f64");
//!
//! // Strictly positive and inside (0, π/2) — see the sample requirements below. The first
//! // draft of this example used 2.0 and the suite rejected it, which is the precondition
//! // doing its job: `atan(tan(2.0))` is `2.0 − π`, not `2.0`.
//! conformance::real_laws(&[0.25_f64, 0.5, 1.0, 1.25], 1e-14, "f64");
//! ```
//!
//! # Tolerance
//!
//! Most laws here are **inexact**, and that is a property of the carrier rather than a weakness
//! in the law: `exp(ln(x)) = x` does not hold bit-for-bit in binary floating point, and a
//! carrier that made it hold would be doing something stranger than the one that does not. Those
//! laws take a relative tolerance and compare through it.
//!
//! The laws that *are* exact say so, and are checked exactly:
//!
//! - the identity and annihilation laws, which involve no rounding
//! - the mask and boolean laws, which are discrete
//! - `select` with a constant mask, which chooses rather than computes
//!
//! A carrier with exact arithmetic can pass `0.0` and get the strict reading of everything.
//!
//! # Availability
//!
//! Gated behind the `conformance` feature, so a consumer pays nothing unless it asks, and always
//! compiled under `cfg(test)` so this crate's own suites exercise the same code a carrier calls.
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
//! Supply **finite** values, and for [`real_laws`] **strictly positive** ones where the law
//! needs a logarithm — the suite documents which. Include `1` and a value either side of it;
//! that is where a series crossover lives, and where an implementation that is right in the
//! bulk goes wrong.

//! # The split
//!
//! This module passed the crate's 400-line budget, so each surface's laws now sit beside the trait
//! they check and this file is the index: the carrier-supplied sample, and the re-exports that keep
//! every public path exactly where it was.

mod masks;
mod real;

/// The scalar laws over **generated** carriers, for a carrier author who wants the search.
///
/// # Why this needs a trait
///
/// `euclid-tensor` can build a carrier through `Tensor::from_host` and `euclid-units` through
/// `Measure::from_raw`, because both contracts already contain a constructor. [`Real`] contains
/// one too — [`Real::scalar`] lifts an `f64` — but reaching for it here would be a mistake: a
/// carrier that is deliberately *deferred*, an expression handle or a graph node, can lift a
/// literal and cannot be compared, so generating values that way would produce a sample the laws
/// cannot evaluate.
///
/// Implement `generated::GeneratedSample` on your carrier to supply values for the
/// generated laws. The primitive carriers' test fixtures are crate-local. To check a
/// primitive from another crate, supply your samples to the slice-taking laws.
///
/// # What generation reaches that a fixed sample does not
///
/// [`real_laws`] is a web of identities — `exp`/`ln` round trips, `sin²+cos²`, the hyperbolic
/// definitions — and each is checked against a tolerance. A fixed sample picks a handful of
/// arguments; a generated one hunts the argument where a kernel's error is *worst*, which for a
/// series implementation is a thin region near a crossover rather than anywhere a human would
/// think to look.
///
/// That is the whole reason this crate ships `euclid-stability`'s `target` and `budget`: pair a
/// generated run with a reported score and the search climbs toward the worst case instead of
/// sampling uniformly. See [`mod@euclid_stability::target`] and [`euclid_stability::budget`].
#[cfg(any(feature = "conformance-generated", test))]
pub mod generated;

// Re-exported so every public path this module has ever had still resolves: the
// split is an internal rearrangement and no carrier should have to notice it.
pub use masks::{boolean_laws, boolean_laws_for, mask_laws, mask_laws_for};
pub use real::{real_laws, real_laws_for};

use crate::{Boolean, Compare, Element, Real};

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
    /// The values this carrier should be checked over.
    fn sample() -> Vec<Self>;

    /// The relative tolerance the inexact laws should use for this carrier.
    ///
    /// Defaulted to zero, which is the right answer for an exact carrier and wrong for every
    /// float — `f64` wants about `1e-14` and `f32` about `1e-6`, and sharing one bound between
    /// them would either fail at `f32` or stop catching anything at `f64`. Override it.
    fn tolerance() -> f64 {
        0.0
    }
}
