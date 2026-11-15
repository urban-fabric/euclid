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
//! # Seeds, not generated input
//!
//! You supply the seeds. These functions generate nothing, so a failure is reproducible from the
//! test that called them — the seed is in the source, with no sidecar file to consult. Four or
//! five seeds including `0` is plenty; the laws run over every pair, and `0` is worth including
//! because a multiplicative mixing function that forgets to handle it degenerates there.

use euclid_numeric::Element;

use crate::RandomSampler;

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

/// [`key_laws`] over a backend's own [`ConformanceSample`].
pub fn key_laws_for<R>(what: &str)
where
    R: RandomSampler + ConformanceSample,
    R::Real: PartialEq + core::fmt::Debug,
{
    key_laws::<R>(&R::seeds(), what);
}

/// The derivation laws: reproducibility, and that `split`/`fold_in` produce independent children
/// deterministically.
///
/// Nine laws, all observed through draws:
///
/// 1. the same seed draws the same value, twice over — the headline promise
/// 2. drawing does not disturb the key (it is a value, not a stream)
/// 3. distinct seeds draw distinct values
/// 4. `split` is a pure function of its key
/// 5. `split(k, n)` returns exactly `n` children
/// 6. `split(k, 0)` is empty
/// 7. children are pairwise independent, and none reproduces its parent
/// 8. `fold_in` is deterministic and depends on its data
/// 9. `split` and `fold_in` are different derivations
///
/// Laws 3 and 7 are *inequality* claims, so a correct-but-unlucky backend could in principle
/// fail them — a collision in the draw space is not a contract violation. In practice a
/// mixing function that decorrelates at all will not collide on a handful of seeds, and a
/// failure here is far more likely to be a missing domain separator than bad luck. If you hit
/// one, change the seeds and see whether it persists.
pub fn key_laws<R>(seeds: &[u64], what: &str)
where
    R: RandomSampler,
    R::Real: PartialEq + core::fmt::Debug,
{
    let draw = |k: &R::Key| R::uniform(k, SCALAR, R::Real::zero(), R::Real::one());

    for &s in seeds {
        // 1. same seed, same draw
        let k = R::key(s);
        assert!(
            draw(&k) == draw(&R::key(s)),
            "{what}: key({s}) must draw the same value every time"
        );

        // 2. the key is a value, not a consumed stream
        let first = draw(&k);
        let second = draw(&k);
        assert!(
            first == second,
            "{what}: drawing from a key must not disturb it"
        );

        // 4-7. split
        let children = R::split(&k, 4);
        assert_eq!(children.len(), 4, "{what}: split(k, 4) must return 4 keys");
        assert!(
            R::split(&k, 0).is_empty(),
            "{what}: split(k, 0) must be empty"
        );
        let again = R::split(&k, 4);
        for (i, (a, b)) in children.iter().zip(again.iter()).enumerate() {
            assert!(
                draw(a) == draw(b),
                "{what}: split must be a pure function of its key (child {i} differs)"
            );
        }
        for (i, c) in children.iter().enumerate() {
            assert!(
                draw(c) != first,
                "{what}: child {i} must not reproduce its parent's draw"
            );
            for (j, d) in children.iter().enumerate().skip(i + 1) {
                assert!(
                    draw(c) != draw(d),
                    "{what}: children {i} and {j} must be independent"
                );
            }
        }

        // 8. fold_in
        let folded = R::fold_in(&k, 11);
        assert!(
            draw(&folded) == draw(&R::fold_in(&k, 11)),
            "{what}: fold_in must be deterministic"
        );
        assert!(
            draw(&folded) != draw(&R::fold_in(&k, 12)),
            "{what}: fold_in must depend on its data"
        );
        assert!(
            draw(&folded) != first,
            "{what}: fold_in must not reproduce its parent's draw"
        );

        // 9. the two derivations are distinct
        assert!(
            draw(&R::split(&k, 1)[0]) != draw(&R::fold_in(&k, 0)),
            "{what}: split and fold_in must be different derivations"
        );
    }

    // 3. distinct seeds draw distinctly
    for (i, &a) in seeds.iter().enumerate() {
        for &b in seeds.iter().skip(i + 1) {
            assert!(
                draw(&R::key(a)) != draw(&R::key(b)),
                "{what}: seeds {a} and {b} must draw different values"
            );
        }
    }
}

/// The parameter laws: a draw must respect the distribution parameters it was given.
///
/// These are the cases where the answer is forced regardless of the algorithm, so they hold for
/// any correct backend:
///
/// - `uniform(low, high)` lies within `[low, high]`
/// - `uniform(x, x)` is exactly `x` — a degenerate range has one answer
/// - `normal(mean, 0)` is exactly `mean` — zero variance has one answer
/// - `bernoulli(0)` is always false and `bernoulli(1)` always true
/// - `randint(low, high)` lies within `[low, high)`, and `randint(x, x+1)` is exactly `x`
///
/// Requires `PartialOrd` on the draw type, so this suite is for carriers with a host order. A
/// deferred carrier whose comparisons yield masks should check the same properties through its
/// own mask surface.
pub fn sampler_bound_laws<R>(seeds: &[u64], what: &str)
where
    R: RandomSampler,
    R::Real: PartialEq + PartialOrd + core::fmt::Debug,
    R::Bool: PartialEq + core::fmt::Debug,
    R::Int: PartialEq + PartialOrd + core::fmt::Debug,
{
    let zero = R::Real::zero();
    let one = R::Real::one();
    let two = one.clone() + one.clone();

    for &s in seeds {
        let k = R::key(s);

        // uniform stays inside its range, and collapses on a degenerate one
        let u = R::uniform(&k, SCALAR, zero.clone(), two.clone());
        assert!(
            u >= zero && u <= two,
            "{what}: uniform(0, 2) drew {u:?}, outside [0, 2]"
        );
        assert!(
            R::uniform(&k, SCALAR, one.clone(), one.clone()) == one,
            "{what}: uniform(x, x) must be exactly x"
        );

        // zero variance has exactly one answer
        assert!(
            R::normal(&k, SCALAR, one.clone(), zero.clone()) == one,
            "{what}: normal(mean, 0) must be exactly mean"
        );

        // the Bernoulli boundaries are forced
        assert!(
            R::bernoulli(&k, SCALAR, zero.clone()) == R::Bool::zero(),
            "{what}: bernoulli(0) must be false"
        );
        assert!(
            R::bernoulli(&k, SCALAR, one.clone()) == R::Bool::one(),
            "{what}: bernoulli(1) must be true"
        );

        // randint respects its half-open interval
        let lo = R::Int::zero();
        let hi = R::Int::one() + R::Int::one() + R::Int::one();
        let n = R::randint(&k, SCALAR, lo.clone(), hi.clone());
        assert!(
            n >= lo && n < hi,
            "{what}: randint(0, 3) drew {n:?}, outside [0, 3)"
        );
        let single = R::Int::zero();
        let next = R::Int::one();
        assert!(
            R::randint(&k, SCALAR, single.clone(), next) == single,
            "{what}: randint(x, x+1) must be exactly x"
        );
    }
}

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
#[cfg(any(feature = "conformance", test))]
pub mod generated {
    use euclid_stability::{Config, Gen, check, classify, generate};

    use super::{key_laws, sampler_bound_laws};
    use crate::RandomSampler;

    /// A seed, weighted toward the magnitudes a mixing function gets wrong.
    ///
    /// `0` first, because a multiplicative mixer that forgets to handle it degenerates there and
    /// every derivation law then passes vacuously. The full-range draw is the other half — a
    /// mixer that only shifts right loses the high bits — and the small values are what make a
    /// first failure readable.
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

    /// Distinct seeds, deduplicated by construction rather than by rejection — see the module
    /// docs for why that distinction matters here.
    ///
    /// Four is the widest worth running: the laws are quadratic in the sample and draw several
    /// times per pair, so breadth comes from the case count instead.
    fn seeds() -> Gen<Vec<u64>> {
        generate::distinct(seed(), 1, 4)
    }

    fn settings() -> Config {
        Config::default()
            .cases(300)
            .require("contains zero", 5.0)
            .require("a high-bit seed", 10.0)
    }

    /// [`key_laws`] over generated seeds.
    ///
    /// The coverage floors are part of the contract being checked, not decoration: a run that
    /// never drew `0` would pass every derivation law vacuously on a degenerate mixer, and one
    /// that never drew a high-bit seed would miss a mixer that only decorrelates the low half.
    pub fn key_laws_generated<R>(what: &str)
    where
        R: RandomSampler,
        R::Real: PartialEq + core::fmt::Debug,
    {
        check(what, &settings(), &seeds(), |xs: Vec<u64>| {
            classify!("contains zero", xs.contains(&0));
            classify!(
                "a high-bit seed",
                xs.iter().any(|s| *s > u64::from(u32::MAX))
            );
            key_laws::<R>(&xs, what);
            Ok(())
        });
    }

    /// [`sampler_bound_laws`] over generated seeds.
    pub fn sampler_bound_laws_generated<R>(what: &str)
    where
        R: RandomSampler,
        R::Real: PartialEq + PartialOrd + core::fmt::Debug,
        R::Bool: PartialEq + core::fmt::Debug,
        R::Int: PartialEq + PartialOrd + core::fmt::Debug,
    {
        check(what, &settings(), &seeds(), |xs: Vec<u64>| {
            classify!("contains zero", xs.contains(&0));
            classify!(
                "a high-bit seed",
                xs.iter().any(|s| *s > u64::from(u32::MAX))
            );
            sampler_bound_laws::<R>(&xs, what);
            Ok(())
        });
    }
}
