//! A property search over the law suites, for locating *where* a carrier fails them.
//!
//! Behind a second feature, `conformance-generated`, which enables the parent's and additionally
//! pulls the property engine. The suites in [`super`] are the contract and resolve no engine at
//! all; this is the tooling for finding a counterexample, and it costs a dependency.
//!
//! A file of its own since the split, rather than an inline `pub mod` — it was the largest item
//! in a module that had passed the 400-line budget.

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

#[cfg(test)]
#[test]
fn generated_search_requires_seed_coverage() {
    let config = settings();
    assert!(config.cases >= 300);
    for label in ["contains zero", "a high-bit seed"] {
        assert!(
            config
                .requirements
                .iter()
                .any(|(name, floor)| name == label && *floor > 0.0)
        );
    }
}
