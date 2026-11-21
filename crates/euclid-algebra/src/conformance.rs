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

//! # The split
//!
//! This module passed the crate's 400-line budget, so each tier's laws now sit beside the tier they
//! check and this file is the index: the carrier-supplied sample, and the re-exports that keep every
//! public path exactly where it was.

mod dioid;
mod module;
mod ring;
mod semiring;
mod witness;

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
pub mod generated;

// Re-exported so every public path this module has ever had still resolves: the
// split is an internal rearrangement and no carrier should have to notice it.
pub use dioid::{
    canonical_order_laws, dioid_laws, dioid_laws_for, ordered_semiring_laws, star_laws,
};
pub use module::{left_semimodule_laws, ref_magma_laws, ref_magma_laws_for};
pub use ring::{
    euclidean_laws, euclidean_laws_for, field_laws, field_laws_for, ring_laws, ring_laws_for,
};
pub use semiring::{
    commutative_semiring_laws, commutative_semiring_laws_for, semiring_laws, semiring_laws_for,
};
pub use witness::{witness_dioid_laws, witness_semiring_laws};

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
