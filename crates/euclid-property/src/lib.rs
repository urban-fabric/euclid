//! # euclid-property
//!
//! Randomized property testing with shrinking, built on a **replayable choice tape**.
//!
//! ```
//! use euclid_property::{check_eq, generate, property};
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

pub mod generate;
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
