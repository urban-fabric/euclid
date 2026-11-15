//! The key and sampler contracts over **generated** seeds and parameters.
//!
//! The companion to [`key_contract`](super::key_contract), which runs the same
//! [`crate::conformance`] laws over `ToySampler`'s fixed five-seed
//! [`ConformanceSample`](crate::conformance::ConformanceSample). Nothing is restated: the laws
//! are the crate's own public functions taking a slice of seeds, and this file changes only
//! where that slice comes from.
//!
//! ## What generated seeds reach that five do not
//!
//! The laws here are unusual in euclid: most of them are **inequality** claims about a mixing
//! function — distinct seeds draw distinct values, children are pairwise independent, `split`
//! and `fold_in` are different derivations. An equality law over a small carrier "either holds
//! structurally or fails on the first awkward triple", which is the argument for fixed samples
//! and a good one. An inequality law about a hash does not work that way: it is a claim about a
//! *large state space*, and five seeds sample it five times.
//!
//! Concretely, a mixing function that decorrelates the low bits and not the high ones, or that
//! loses a domain separator above some magnitude, passes `{0, 1, 7, 42, 0xDEADBEEF}` and fails
//! across the seed space. That is the class this suite hunts, and it is the same reason the
//! algebra crate generates: the laws are about structure, the bugs are about representation.
//!
//! ## Why the seeds are deduplicated
//!
//! [`key_laws`](crate::conformance::key_laws) asserts that every *pair* of seeds in its slice
//! draws differently. A slice containing the same seed twice therefore fails it — and that
//! failure says nothing about the backend, since the same seed is required by law 1 to draw the
//! same value. Distinctness is a **precondition on the sample**, which a hand-written list
//! satisfies by inspection and a generator has to be told about.
//!
//! It is enforced by construction rather than by a filter: the generator sorts and dedups, so
//! every draw is a valid sample and the shrinker never discards a candidate. A `filter` would
//! have shrunk toward collisions and spent its budget rejecting them.
//!
//! ## Beyond the laws: parameters the fixed suite hard-codes
//!
//! [`sampler_bound_laws`](crate::conformance::sampler_bound_laws) checks `uniform` against
//! `[0, 2]`, `normal(1, 0)`, and `randint` over one interval, because a law that names its own
//! parameters has to name *some*. The properties below run the same forced answers over
//! generated parameters, which is where an implementation that rescales with the wrong sign, or
//! that derives a range as `low + u * high` rather than `low + u * (high - low)`, shows up.

use euclid_stability::{Config, Gen, check, classify, generate, property};

use super::toy_sampler::ToySampler;
use crate::{RandomKey, RandomSampler};

type S = ToySampler;

/// A rank-0 draw: this crate's contracts take a shape, and a scalar is the empty one.
const SCALAR: &[usize] = &[];

/// Enough cases to sample the seed space meaningfully; the laws are `O(n²)` in the sample size,
/// which is why the samples stay small rather than the case count low.
fn settings(suite: &str) -> Config {
    Config::default()
        .cases(300)
        .regressions(format!("src/tests/regressions/{suite}.txt"))
}

/// A seed, weighted toward the magnitudes a mixing function gets wrong.
///
/// `0` first, for the reason the fixed sample gives: a multiplicative mixer that forgets to
/// handle it degenerates there and every derivation law then passes vacuously. The high-bit
/// entries are the other half — a mixer that only shifts right loses them, and a uniform draw
/// over `i64` would never produce the small values that make the first failure readable.
fn seed() -> Gen<u64> {
    generate::weighted(vec![
        (2, generate::just(0i64)),
        (2, generate::just(1i64)),
        (2, generate::just(-1i64)),
        (3, generate::int(0, 64)),
        (3, generate::int(i64::MIN, i64::MAX)),
    ])
    .map(|v| v as u64)
}

/// A finite value well inside the exponent range, so a difference or a product in a sampler's
/// rescaling cannot overflow and turn a parameter law into an arithmetic question.
fn parameter() -> Gen<f64> {
    generate::float::in_range(-1e6, 1e6)
}

/// An ordered pair `(low, high)` with `low ≤ high`, built by adding a non-negative width rather
/// than by filtering an unordered pair — so every draw is a valid range.
fn range() -> Gen<(f64, f64)> {
    generate::pair(parameter(), generate::float::in_range(0.0, 1e6)).map(|(low, width)| {
        let high = low + width;
        (low, high)
    })
}

/// A half-open integer interval `[low, high)`, non-empty by construction.
fn int_range() -> Gen<(i64, i64)> {
    generate::pair(
        generate::int(-1_000_000, 1_000_000),
        generate::int(1, 10_000),
    )
    .map(|(low, span)| (low, low + span))
}

/// A rank-0 uniform draw in `[0, 1)`, which is how every law here observes a key.
///
/// Keys are compared through what they draw rather than directly, matching the conformance
/// laws: `ToySampler::Key` happens to be `Eq + Hash`, but [`RandomKey`] does not require it, so
/// a law written against a key's identity would not be runnable by another backend.
fn draw(key: &<S as RandomKey>::Key) -> f64 {
    S::uniform(key, SCALAR, 0.0, 1.0)
}

// ===== The exported generated laws, run against the reference backend =====
//
// These call `conformance::generated` rather than re-deriving the seed strategy, which is the
// point: the strategy is now public API a backend uses, so the only way to know it works is to
// be its first caller. A local copy would compile and drift.

/// `ToySampler` against every derivation law, over the generated seeds euclid ships.
#[test]
fn the_toy_sampler_satisfies_the_generated_key_laws() {
    crate::conformance::generated::key_laws_generated::<S>("ToySampler");
}

/// The parameter laws, likewise.
///
/// Run alongside the derivation laws rather than instead of them: the two are independent halves
/// of the contract, and `sampler_bound_laws` is the one a carrier with a host order can run while
/// a masked carrier cannot.
#[test]
fn the_toy_sampler_satisfies_the_generated_sampler_bound_laws() {
    crate::conformance::generated::sampler_bound_laws_generated::<S>("ToySampler");
}

property! {
    #![config(settings("key-mixing"))]

    /// Flipping **one bit** of a seed changes the draw.
    ///
    /// The avalanche property, and the strongest statement about a mixing function available
    /// without measuring a distribution. `key_laws` can only check the seeds it was handed;
    /// this checks the adjacent seed, at every bit position, which is exactly where a mixer
    /// that shifts in one direction only fails to decorrelate.
    fn flipping_one_bit_of_a_seed_changes_the_draw(
        s in seed(),
        bit in generate::index(0, 63),
    ) {
        classify!("low half", bit < 32);
        classify!("high half", bit >= 32);

        let flipped = s ^ (1u64 << bit);
        check!(
            draw(&S::key(s)) != draw(&S::key(flipped)),
            "key({s}) and key({flipped}) (bit {bit} flipped) drew the same value"
        );
    }

    /// `split` yields the requested count, deterministically, with no child reproducing another.
    ///
    /// `key_laws` fixes the count at 4 and 0 — the two it can name. A generated count is what
    /// exercises a backend that derives a child from an index with an off-by-one, or that
    /// special-cases a small fan-out.
    fn split_is_deterministic_at_any_count(s in seed(), n in generate::index(0, 12)) {
        classify!("empty", n == 0);
        classify!("singleton", n == 1);

        let root = S::key(s);
        let children = S::split(&root, n);
        check!(
            children.len() == n,
            "split(key({s}), {n}) returned {} keys",
            children.len()
        );

        let again = S::split(&root, n);
        for (i, (a, b)) in children.iter().zip(again.iter()).enumerate() {
            check!(
                draw(a) == draw(b),
                "split is not a pure function of its key: child {i} of key({s}) differs"
            );
        }

        let draws: Vec<f64> = children.iter().map(draw).collect();
        for (i, a) in draws.iter().enumerate() {
            for (j, b) in draws.iter().enumerate().skip(i + 1) {
                check!(a != b, "children {i} and {j} of key({s}) drew the same value");
            }
        }
    }

    /// `fold_in` depends on its data, at every magnitude of data.
    ///
    /// `key_laws` folds `11` and `12` — adjacent small values. A pair drawn across the whole
    /// `u64` range is what catches a backend that mixes the data in by addition and loses the
    /// high bits, which `11` versus `12` cannot see.
    fn fold_in_depends_on_its_data(s in seed(), a in seed(), b in seed()) {
        classify!("equal data", a == b);

        let root = S::key(s);
        check!(
            draw(&S::fold_in(&root, a)) == draw(&S::fold_in(&root, a)),
            "fold_in(key({s}), {a}) is not deterministic"
        );

        if a != b {
            check!(
                draw(&S::fold_in(&root, a)) != draw(&S::fold_in(&root, b)),
                "fold_in(key({s}), {a}) and fold_in(key({s}), {b}) drew the same value"
            );
        }
    }
}

property! {
    #![config(settings("sampler-parameters"))]

    /// `uniform` stays inside a **generated** range, and collapses on a degenerate one.
    ///
    /// `sampler_bound_laws` checks `[0, 2]`, where `low + u * high` and
    /// `low + u * (high - low)` agree because `low` is zero. A generated range separates them.
    fn uniform_respects_a_generated_range(s in seed(), (low, high) in range()) {
        classify!("degenerate", low == high);
        classify!("spans zero", low < 0.0 && high > 0.0);

        let k = S::key(s);
        let u = S::uniform(&k, SCALAR, low, high);
        check!(
            u >= low && u <= high,
            "uniform({low}, {high}) drew {u}, outside the range"
        );
        check!(
            S::uniform(&k, SCALAR, low, low) == low,
            "uniform({low}, {low}) must be exactly {low}"
        );
    }

    /// `normal(mean, 0)` is exactly the mean, at any mean.
    ///
    /// Zero variance has one forced answer and needs no distributional reasoning, which is what
    /// makes it checkable as a property. A generated mean catches a Box–Muller implementation
    /// that adds the mean before scaling by the standard deviation.
    fn zero_variance_is_exactly_the_mean(s in seed(), mean in parameter()) {
        classify!("zero mean", mean == 0.0);

        let drawn = S::normal(&S::key(s), SCALAR, mean, 0.0);
        check!(
            drawn == mean,
            "normal({mean}, 0) drew {drawn}, not exactly the mean"
        );
    }

    /// `randint` respects a **generated** half-open interval.
    ///
    /// The half-open end is the one an implementation gets wrong, and it is invisible on a wide
    /// interval — a single off-by-one at the top of `[-1000000, 1000000)` almost never appears.
    /// A generated span includes the narrow intervals where it does, and `[x, x+1)` has exactly
    /// one legal answer.
    fn randint_respects_a_generated_interval(s in seed(), (low, high) in int_range()) {
        classify!("unit interval", high - low == 1);
        classify!("narrow", high - low <= 4);

        let k = S::key(s);
        let n = S::randint(&k, SCALAR, low, high);
        check!(
            n >= low && n < high,
            "randint({low}, {high}) drew {n}, outside [{low}, {high})"
        );
        check!(
            S::randint(&k, SCALAR, low, low + 1) == low,
            "randint({low}, {}) must be exactly {low}",
            low + 1
        );
    }
}
