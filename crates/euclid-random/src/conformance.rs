//! Conformance suites — the key and sampler contracts as functions you can check a backend
//! against.
//!
//! This crate's whole promise is reproducibility: *a draw is a pure function of its key,
//! distribution parameters, and result shape.* Nothing in the type system enforces that. A
//! backend that threads hidden state through a draw, or derives `split` children that collide,
//! compiles and type-checks exactly like one that does not — and then silently produces
//! irreproducible results, which is the one failure that makes a sampler useless.
//!
//! ```
//! # use euclid_random::conformance;
//! # use euclid_random::{RandomKey, RandomSampler};
//! # struct Toy;
//! # impl RandomKey for Toy {
//! #     type Key = u64;
//! #     fn key(seed: u64) -> u64 { seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1 }
//! #     fn split(key: &u64, count: usize) -> Vec<u64> {
//! #         (0..count as u64).map(|i| Self::fold_in(&(key ^ 0x5F11), i)).collect()
//! #     }
//! #     fn fold_in(key: &u64, data: u64) -> u64 {
//! #         let mut z = key.wrapping_add(data).wrapping_add(0x9E37_79B9_7F4A_7C15);
//! #         z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
//! #         z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
//! #         (z ^ (z >> 31)) | 1
//! #     }
//! # }
//! # impl RandomSampler for Toy {
//! #     type Real = f64;
//! #     type Bool = bool;
//! #     type Int = i64;
//! #     fn uniform(key: &u64, _: &[usize], low: f64, high: f64) -> f64 {
//! #         low + (high - low) * ((key >> 11) as f64 / (1u64 << 53) as f64)
//! #     }
//! #     fn normal(key: &u64, _: &[usize], mean: f64, stddev: f64) -> f64 {
//! #         mean + stddev * (((key >> 11) as f64 / (1u64 << 53) as f64) - 0.5)
//! #     }
//! #     fn bernoulli(key: &u64, _: &[usize], p: f64) -> bool {
//! #         ((key >> 11) as f64 / (1u64 << 53) as f64) < p
//! #     }
//! #     fn randint(key: &u64, _: &[usize], low: i64, high: i64) -> i64 {
//! #         if high <= low { low } else { low + ((key >> 11) % (high - low) as u64) as i64 }
//! #     }
//! # }
//! // A backend author checks the promise rather than asserting it.
//! conformance::key_laws::<Toy>(&[0, 1, 7, 42], "Toy");
//! ```
//!
//! # Why the laws are stated through draws
//!
//! [`RandomKey::Key`](crate::RandomKey::Key) is only `Clone`. The contract gives no way to compare two keys, so
//! "`split` is deterministic" and "same key, same value" are **not directly observable** — a
//! key is only visible through what it draws. Every law below therefore compares *draws*, which
//! is both the only option and the right one: a backend whose keys compare equal but draw
//! differently has still broken the contract, and one whose keys are traced graph values can
//! still be checked.
//!
//! That is worth knowing if you implement this trait. Keys need no `PartialEq`, and these suites
//! do not ask for one.
//!
//! # Bounds, and what they imply
//!
//! [`key_laws`] needs only `PartialEq` on the draw type, so it applies to any carrier that can
//! compare two draws for equality. [`sampler_bound_laws`] additionally needs `PartialOrd`,
//! because a range check is an ordering: a deferred carrier whose comparisons yield masks rather
//! than host booleans cannot satisfy that bound, and should check its parameter handling against
//! its own mask surface instead.
//!
//! # Availability
//!
//! Gated behind the `conformance` feature, so a consumer that only draws samples pays nothing
//! for it, and always compiled under `cfg(test)` so this crate's own suites exercise the same
//! code a backend calls.
//!
//! The `generated` submodule sits behind a **second** feature, `conformance-generated`, which
//! enables this one and additionally pulls the property engine. The division: these functions
//! are the contract, so `conformance` resolves no engine at all and is what a consumer binding
//! the laws should take; the generated search is tooling for locating where a carrier fails
//! them, and costs a dependency.
//!
//! # Seeds, not generated input
//!
//! You supply the seeds. These functions generate nothing, so a failure is reproducible from the
//! test that called them — the seed is in the source, with no sidecar file to consult. Four or
//! five seeds including `0` is plenty; the laws run over every pair, and `0` is worth including
//! because a multiplicative mixing function that forgets to handle it degenerates there.

//! # The split
//!
//! This module passed the crate's 400-line budget, so each surface's laws now sit beside the trait
//! they check and this file is the index: the carrier-supplied sample, and the re-exports that keep
//! every public path exactly where it was.

mod key;
mod sampler;

/// The same laws over **generated** seeds, for a backend that wants the search rather than the
/// five-seed sample.
///
/// # Why this is exported rather than left for each backend to write
///
/// The laws above are unusual in euclid: most of them are *inequality* claims about a mixing
/// function — distinct seeds draw distinct values, children are pairwise independent, `split` and
/// `fold_in` are different derivations. An equality law over a small carrier either holds
/// structurally or fails on the first awkward triple, which is the argument for a fixed sample
/// and a good one. An inequality law about a hash is a claim about a *large state space*, and
/// five seeds sample it five times.
///
/// A backend writing its own version has to rediscover one precondition that is not obvious and
/// not optional: [`key_laws`] asserts that every *pair* of seeds draws differently, so **a sample
/// containing the same seed twice fails it** — and that failure says nothing about the backend,
/// since law 1 requires the same seed to draw the same value. Distinctness is a precondition on
/// the sample. A hand-written list satisfies it by inspection; a generator has to be told, and
/// the obvious spelling (generate a `Vec`, then `filter` for distinctness) is the wrong one,
/// because the shrinker's job is to make values more alike and it will spend its budget walking
/// into collisions it then rejects.
///
/// So the generator deduplicates by construction. That is the knowledge this module exists to
/// ship.
#[cfg(any(feature = "conformance-generated", test))]
pub mod generated;

// Re-exported so every public path this module has ever had still resolves: the
// split is an internal rearrangement and no carrier should have to notice it.
pub use key::{key_laws, key_laws_for};
pub use sampler::sampler_bound_laws;

use crate::RandomSampler;
use euclid_scalar::{Element, Integer, Real};

/// The shape every law below draws at: a single scalar.
///
/// Scalar backends require `&[]`; array and dataframe backends interpret a non-empty shape in
/// their own way. The key contract is independent of shape, so the suites use the one shape
/// every backend must support.
const SCALAR: &[usize] = &[];

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
pub trait ConformanceSample {
    /// The seeds to derive keys from. Include `0` — a multiplicative mixing function that
    /// forgets to handle it degenerates there.
    fn seeds() -> Vec<u64>;
}
