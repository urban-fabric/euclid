//! A property search over the law suites, for locating *where* a carrier fails them.
//!
//! Behind a second feature, `conformance-generated`, which enables the parent's and additionally
//! pulls the property engine. The suites in [`super`] are the contract and resolve no engine at
//! all; this is the tooling for finding a counterexample, and it costs a dependency.
//!
//! A file of its own since the split, rather than an inline `pub mod` — it was by some distance
//! the largest item in a module that had passed the 400-line budget.

use euclid_stability::{Config, Gen, check, classify};

use super::{commutative_semiring_laws, euclidean_laws, field_laws, ring_laws, semiring_laws};
use crate::{CommutativeSemiring, EuclideanDomain, Field, Ring, Semiring};

/// A carrier's own generator, declared once — the generated peer of
/// [`ConformanceSample`](super::ConformanceSample).
///
/// Weight it toward the values that break things rather than uniformly: `0` and `±1` are the
/// identities and the annihilator, so they are what a law is most likely to get wrong, and a
/// uniform draw over a wide range would almost never produce them.
///
/// Bound the magnitudes by what your arithmetic can hold. The laws run over *triples* —
/// distributivity alone forms `(a ⊕ b) ⊗ c` — so a carrier backed by a fixed-width integer
/// should keep a triple's worst case well inside it. A debug-build overflow panics, and that
/// panic is indistinguishable from a law failure at the call site.
pub trait GeneratedSample: Sized {
    /// Values to draw from.
    fn generator() -> Gen<Self>;

    /// How many cases to run. Defaulted at 300: enough that the shrinker has something to
    /// work with, few enough that every tier law over a triple stays quick.
    fn cases() -> usize {
        300
    }
}

/// Three is the minimum that exercises a three-operand law at all; six matches what a fixed
/// sample uses, and the laws are `O(n³)`.
fn sample<T: GeneratedSample + 'static>() -> Gen<Vec<T>> {
    euclid_stability::generate::vec(T::generator(), 3, 6)
}

/// `pub(crate)` only so the vacuity suite can assert the contract below directly.
///
/// The thing worth pinning is that a carrier's declared [`GeneratedSample::cases`] is what
/// the search actually runs. Replacing this body with `Config::default()` silently drops
/// that — `Config`'s own default is 256 cases — and no behavioural test separates the two
/// without counting draws, which would be a statistical assertion about a generator rather
/// than a statement about this function. One `assert_eq!` on the field says it exactly.
pub(crate) fn settings<T: GeneratedSample>() -> Config {
    Config::default().cases(T::cases())
}

/// [`semiring_laws`] over generated elements — the trunk every other tier refines.
pub fn semiring_laws_generated<T>(what: &str)
where
    T: Semiring + GeneratedSample + core::fmt::Debug + Copy + 'static,
{
    check(what, &settings::<T>(), &sample::<T>(), |xs| {
        classify!("contains zero", xs.contains(&T::ZERO));
        semiring_laws(&xs, what);
        Ok(())
    });
}

/// [`commutative_semiring_laws`] over generated elements.
///
/// Separate from [`semiring_laws_generated`] rather than folded into it, because commutative
/// `⊗` is a strictly narrower claim — matrix multiplication is a semiring and is not
/// commutative, so a single entry point would exclude it.
pub fn commutative_semiring_laws_generated<T>(what: &str)
where
    T: CommutativeSemiring + GeneratedSample + core::fmt::Debug + Copy + 'static,
{
    check(what, &settings::<T>(), &sample::<T>(), |xs| {
        classify!("contains zero", xs.contains(&T::ZERO));
        commutative_semiring_laws(&xs, what);
        Ok(())
    });
}

/// [`ring_laws`] over generated elements — the additive-inverse tier.
pub fn ring_laws_generated<T>(what: &str)
where
    T: Ring + GeneratedSample + core::fmt::Debug + Copy + 'static,
{
    check(what, &settings::<T>(), &sample::<T>(), |xs| {
        classify!("contains zero", xs.contains(&T::ZERO));
        ring_laws(&xs, what);
        Ok(())
    });
}

/// [`field_laws`] over generated elements.
///
/// The division laws skip `ZERO` internally, so a generator that produces it
/// is not a problem — and should, since a field law that silently never divided by anything
/// awkward is the one most worth running.
pub fn field_laws_generated<T>(what: &str)
where
    T: Field + GeneratedSample + core::fmt::Debug + Copy + 'static,
{
    check(what, &settings::<T>(), &sample::<T>(), |xs| {
        classify!("contains zero", xs.contains(&T::ZERO));
        field_laws(&xs, what);
        Ok(())
    });
}

/// [`euclidean_laws`] over generated elements — division with remainder.
pub fn euclidean_laws_generated<T>(what: &str)
where
    T: EuclideanDomain + GeneratedSample + core::fmt::Debug + Copy + 'static,
{
    check(what, &settings::<T>(), &sample::<T>(), |xs| {
        classify!("contains zero", xs.contains(&T::ZERO));
        euclidean_laws(&xs, what);
        Ok(())
    });
}
