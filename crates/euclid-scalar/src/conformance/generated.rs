//! A property search over the law suites, for locating *where* a carrier fails them.
//!
//! Behind a second feature, `conformance-generated`, which enables the parent's and additionally
//! pulls the property engine. The suites in [`super`] are the contract and resolve no engine at
//! all; this is the tooling for finding a counterexample, and it costs a dependency.
//!
//! A file of its own since the split, rather than an inline `pub mod` — it was the largest item
//! in a module that had passed the 400-line budget.

use euclid_stability::{Config, Gen, check, classify};

use super::{boolean_laws, mask_laws, real_laws};
use crate::{Boolean, Compare, Real};

/// A carrier's own generator, declared once — the generated peer of
/// [`ConformanceSample`](super::ConformanceSample).
///
/// Follow the sample requirements of the slice-taking laws. For [`super::real_laws`],
/// generate values in `(0, π/2)`, including values near `1` to exercise series crossovers.
pub trait GeneratedSample: Sized {
    /// Values to draw from.
    fn generator() -> Gen<Self>;

    /// How many cases to run. Defaulted at 400 — these laws are per-value rather than over
    /// triples, so a wider search is affordable here than in the algebra tower.
    fn cases() -> usize {
        400
    }
}

fn sample<T: GeneratedSample + 'static>() -> Gen<Vec<T>> {
    euclid_stability::generate::vec(T::generator(), 1, 6)
}

/// `pub(crate)` only so the vacuity suite can assert the contract below directly.
///
/// The claim worth pinning is that a carrier's declared [`GeneratedSample::cases`] is what
/// the search actually runs. Replacing this body with `Config::default()` silently drops it
/// — `Config`'s own default is 256 — and no behavioural test separates the two without
/// counting draws, which would be a statistical assertion about a generator rather than a
/// statement about this function. The same reasoning, and the same fix, as the algebra
/// tower's peer.
pub(crate) fn settings<T: GeneratedSample>() -> Config {
    Config::default().cases(T::cases())
}

/// [`real_laws`] over generated values.
///
/// `tolerance` is relative and is the carrier's to choose from its own precision — see
/// [`super::real_laws`], which explains why a bound that grows with the condition
/// number is honest and a bound loosened until the suite passes is not.
pub fn real_laws_generated<T>(tolerance: f64, what: &str)
where
    T: Real + GeneratedSample + PartialEq + core::fmt::Debug + 'static,
    T::Mask: PartialEq + core::fmt::Debug,
{
    check(what, &settings::<T>(), &sample::<T>(), |xs| {
        real_laws(&xs, tolerance, what);
        Ok(())
    });
}

/// [`mask_laws`] over generated values — the comparison surface.
pub fn mask_laws_generated<T>(what: &str)
where
    T: Compare + GeneratedSample + PartialEq + core::fmt::Debug + 'static,
    T::Mask: PartialEq + core::fmt::Debug,
{
    check(what, &settings::<T>(), &sample::<T>(), |xs| {
        classify!("more than one value", xs.len() > 1);
        mask_laws(&xs, what);
        Ok(())
    });
}

/// [`boolean_laws`] over generated values — the mask carrier's own algebra.
pub fn boolean_laws_generated<B>(what: &str)
where
    B: Boolean + GeneratedSample + PartialEq + core::fmt::Debug + 'static,
{
    check(what, &settings::<B>(), &sample::<B>(), |xs| {
        boolean_laws(&xs, what);
        Ok(())
    });
}
