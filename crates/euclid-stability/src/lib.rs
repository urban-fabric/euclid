//! # euclid-stability
//!
//! **Numerical stability and property testing**, over a replayable choice tape.
//!
//! ```
//! use euclid_stability::{check_eq, generate, property};
//!
//! property! {
//!     fn reversing_twice_is_the_identity(xs in generate::vec(generate::int(0, 9), 0, 8)) {
//!         let mut round_trip = xs.clone();
//!         round_trip.reverse();
//!         round_trip.reverse();
//!         check_eq!(round_trip, xs);
//!     }
//! }
//! ```
//!
//! # The name, and why it is not `euclid-property`
//!
//! It was, until 0.5. The rename is a correction rather than a preference, because "property
//! testing" names the smaller half of what this does and points at the wrong tradition.
//!
//! In the Haskell and Rust lineage — QuickCheck, proptest — property testing is **type-driven**:
//! an `Arbitrary` instance per type, generation derived from structure, and the interesting
//! question is how to produce a well-formed value. That is a real and useful thing, and it is
//! not where this crate's weight sits. There is no `Arbitrary` here and deliberately so (see
//! [`generate`]); a generator is declared, in the same spirit as euclid's own
//! `ConformanceSample`.
//!
//! What the fleet actually needs from it is **numerical**: how much error a kernel has at its
//! worst input, whether that error is explained by the problem's conditioning or by the
//! implementation, whether an algorithm crossover leaves a discontinuity, whether accuracy got
//! worse since last week. Those questions are not about types at all, and most of the surface
//! here exists to answer them:
//!
//! - [`numeric`] — condition numbers, forward and two-stage error bounds, seam hunting, ulp
//!   monotonicity scans. None of it is property testing; all of it is numerical analysis.
//! - [`tolerance`] — ulp, relative and absolute comparison, and the condition number of a sum,
//!   because `==` on a computed float is the single most common way to write a numerical test
//!   that cannot pass.
//! - [`target()`] — hill-climbing towards the *worst* case rather than sampling uniformly, which
//!   matters because a kernel's worst input is a thin region that uniform draws miss.
//! - [`budget`] — recording that worst case so losing accuracy fails the suite, even while every
//!   assertion still holds.
//! - [`relation`] — the metamorphic shapes, each taking a comparison, including the two that are
//!   purely about stability: `forward_stable` and `precision_agrees`.
//! - [`machine`] — sequential properties, for the bugs that live in an *interleaving* rather than
//!   in a value: a system checked against an obviously-correct model over a generated sequence of
//!   operations.
//!
//! The property-testing machinery — [`source`], [`generate`], [`shrink`], [`runner`],
//! [`persist`] — is the **foundation those sit on** rather than the point. A search needs cases
//! to search over and a way to minimise what it finds; that is what a tape buys, and it is why
//! this is one crate instead of two.
//!
//! # Why this exists, given euclid's own membership rule
//!
//! euclid's rule is that something belongs here when two independently-written parts of the
//! fleet must *agree* on it. A test harness is not that. Two projects using different property
//! runners do not produce two different numbers in production; they produce tests of differing
//! quality, which is a real cost but a different and much smaller one. By the letter of the
//! rule this crate is not a euclid contract, and it does not pretend to be one.
//!
//! It is here on the precedent [`euclid-utils`] sets: a `publish = false`, dev-only member that
//! the workspace and its consumers take under `[dev-dependencies]` and that never reaches a
//! published artifact. The alternative was a separate repository for test tooling, which is a
//! lot of ceremony for eight hundred lines that only ever runs under `cargo test`.
//!
//! What it replaces is `proptest`, and the reason is not supply chain or licensing — proptest
//! is MIT/Apache-2.0 and perfectly reputable. It is that the fleet's generators are
//! overwhelmingly **dependent**: a node count bounds the edge endpoints after it, a matrix
//! dimension bounds the indices into it. That is proptest's known weak spot, documented in its
//! own source — nested `prop_flat_map` combinators "risk exponential run time" during
//! shrinking, so they are capped by a `max_flat_map_regens` counter and simply stop improving.
//! The tape design has no such limit, because the shrinker edits the tape and the generator
//! re-derives every dependent bound for free. See [`source`] for the mechanism.
//!
//! # How it works
//!
//! | module | role |
//! |---|---|
//! | [`source`] | the choice tape; a value is a pure function of it |
//! | [`generate`] | generators and combinators, reading from a tape |
//! | [`shrink`] | one generic minimiser over the tape, with no per-type rules |
//! | [`runner`] | search, shrink, persist, report |
//! | [`persist`] | minimised failures as committable hex |
//! | [`rng`] | SplitMix64, the only randomness in the crate |
//! | [`tolerance`] | ULP and relative-error comparison, and a condition number |
//! | [`numeric`] | hunting instability: seam detection, wiggle scanning, derived tolerances |
//! | [`relation`] | the metamorphic relations, named — involution, round trip, two-impls-agree |
//! | [`stats`] | case classification, with coverage you can *require* rather than merely read |
//! | [`mod@target`] | a score for the search to maximise, so it hunts the worst case |
//! | [`budget`] | that worst score, committed, so losing accuracy fails the suite |
//! | [`machine`] | a system against a model, over a generated sequence of operations |
//!
//! [`generate::float`] is the numerics half: float generation built to find numerical bugs
//! rather than to be uniform, which is a different and much more useful distribution.
//!
//! The consequence worth internalising: **there are no shrink implementations to write.** A new
//! generator composes out of [`generate`] and is immediately as shrinkable as everything else,
//! because shrinking never sees the value's type. Adding a generator cannot introduce a
//! shrinking bug, which is the structural advantage over per-type shrink trees.
//!
//! # Determinism
//!
//! The seed is fixed in [`Config`] by default. A suite that passes today passes tomorrow, and a
//! property that only fails on one seed in ten is a property that fails — surfacing that as a
//! flake rather than a failure wastes the finding. Vary [`Config::seed`] deliberately for a
//! longer soak; do not vary it per run.
//!
//! # What it deliberately does not do
//!
//! No `Arbitrary` derive — declare the generator, in the same spirit as
//! [`ConformanceSample`](https://github.com/urban-fabric/euclid) over a blanket impl. No regex
//! string generation, no process forking for aborts, no `no_std`. Each is absent because
//! nothing in the fleet needed it, and adding one is cheaper than maintaining it unused.
//!
//! [`euclid-utils`]: https://github.com/urban-fabric/euclid

// A test asserts by panicking, which the workspace otherwise denies. Same exemption every
// other crate in the fleet takes for its own suites.
#![cfg_attr(test, allow(clippy::expect_used, clippy::panic, clippy::unwrap_used))]

pub mod budget;
pub mod generate;
pub mod machine;
pub mod numeric;
pub mod persist;
pub mod relation;
pub mod rng;
pub mod runner;
pub mod shrink;
pub mod source;
pub mod stats;
pub mod target;
pub mod tolerance;

mod macros;

pub use generate::{Gen, Rejected};
pub use runner::{Config, Failure, Outcome, check};
pub use source::Source;
pub use target::target;

#[cfg(test)]
mod tests;

// The README's examples, compiled.
//
// `cfg(doctest)` means this item exists only while rustdoc is collecting doctests: it is absent
// from every real build, adds nothing to the public surface, and never appears in the rendered
// docs. What it buys is that `cargo test --doc` compiles the code blocks in README.md itself.
//
// A crate whose deliverable is its trait surface has a README whose example *is* part of that
// surface, and a README example that no longer compiles is worse than none — it documents a
// signature the crate no longer has. Two crates used to check this by hand-copying the example
// into `tests/readme_check.rs`, which compiles the copy and lets the original drift. This checks
// the original.
//
// Illustrative fragments in the README are marked ```rust,ignore, which is the honest marking:
// they reference names that are deliberately undefined.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;
