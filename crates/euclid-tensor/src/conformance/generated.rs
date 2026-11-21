//! A property search over the law suites, for locating *where* a carrier fails them.
//!
//! Behind a second feature, `conformance-generated`, which enables the parent's and additionally
//! pulls the property engine. The suites in [`super`] are the contract and resolve no engine at
//! all; this is the tooling for finding a counterexample, and it costs a dependency.
//!
//! A file of its own since the split, rather than an inline `pub mod` — it was the largest item
//! in a module that had passed the 400-line budget.

use euclid_scalar::Real;
use euclid_stability::{Config, Gen, check, classify, generate};

use super::{shape_laws, structural_laws, tensor_laws};
use crate::Tensor;

/// Rank 4 is one above where a hand-written sample usually stops, which is the point.
const MAX_RANK: usize = 4;

/// An extent, weighted toward the two that break things.
///
/// `1` because a unit axis makes a permutation unobservable in the shape; `2` and `3` because
/// a small alphabet is what produces *repeated* extents. A uniform draw over a wide range
/// would make every extent distinct and give away every permutation bug for free.
fn extent() -> Gen<usize> {
    generate::weighted(vec![
        (3, generate::just(1usize)),
        (4, generate::just(2usize)),
        (3, generate::just(3usize)),
        (1, generate::just(4usize)),
    ])
}

/// A shape, including the rank-0 case the laws must skip rather than fail on.
fn shape() -> Gen<Vec<usize>> {
    generate::vec(extent(), 0, MAX_RANK)
}

/// Finite values bounded well away from the exponent limits.
///
/// These laws are structural — permutations, reversals, reshapes — so no arithmetic runs on
/// the data and its magnitude is irrelevant. What matters is that it is finite: the laws
/// compare with `PartialEq`, and a `NaN` would fail every identity by never equalling itself,
/// which says nothing about the backend.
fn value() -> Gen<f64> {
    generate::float::in_range(-1e6, 1e6)
}

/// A shape paired with exactly enough row-major data to fill it — the dependent draw
/// [`Tensor::from_host`] needs.
fn shaped_data() -> Gen<(Vec<usize>, Vec<f64>)> {
    shape().and_then(|s| {
        let count: usize = s.iter().product();
        generate::vec(value(), count, count).map(move |data| (s.clone(), data))
    })
}

/// Build the carriers for one generated case.
///
/// # Panics
/// If `parts` is empty, or if it yields no carriers.
///
/// The assertion is the point. A law suite over an empty sample asserts nothing and reports
/// a clean pass, and mutation testing found this function replaceable with `vec![]` —
/// returning no tensors at all — with every generated law in the crate still green. The
/// generator is `vec(shaped_data(), 1, 3)`, so a case can never legitimately be empty; if one
/// is, the sample construction is broken rather than the laws being satisfied.
fn sample<T: Tensor + Real>(parts: &[(Vec<usize>, Vec<f64>)]) -> Vec<T> {
    let xs: Vec<T> = parts
        .iter()
        .map(|(shape, data)| T::from_host(data, shape))
        .collect();
    assert!(
        !xs.is_empty(),
        "the generated sample is empty, so the laws below would pass vacuously"
    );
    xs
}

fn has_repeated_extents(shape: &[usize]) -> bool {
    let mut sorted = shape.to_vec();
    sorted.sort_unstable();
    let len = sorted.len();
    sorted.dedup();
    sorted.len() < len
}

fn settings() -> Config {
    Config::default()
        .cases(300)
        .require("a repeated-extent tensor", 10.0)
        .require("a rank-0 tensor", 1.0)
        .require("a rank-4 tensor", 1.0)
}

/// [`tensor_laws`] over generated shapes.
///
/// The coverage floors are the contract this function is making: without them a run could
/// report a clean pass having never produced a repeated extent, which is the case the whole
/// search exists for.
pub fn tensor_laws_generated<T>(what: &str)
where
    T: Tensor + Real + PartialEq + core::fmt::Debug + 'static,
{
    let generator = generate::vec(shaped_data(), 1, 3);
    check(what, &settings(), &generator, |parts| {
        let xs: Vec<T> = sample(&parts);
        classify!(
            "a repeated-extent tensor",
            parts.iter().any(|(s, _)| has_repeated_extents(s))
        );
        classify!("a rank-0 tensor", parts.iter().any(|(s, _)| s.is_empty()));
        classify!(
            "a rank-4 tensor",
            parts.iter().any(|(s, _)| s.len() == MAX_RANK)
        );
        tensor_laws(&xs, what);
        Ok(())
    });
}

/// [`shape_laws`] and [`structural_laws`] separately, over the same generator.
///
/// Worth running alongside the aggregate rather than instead of it: if [`tensor_laws`] ever
/// stopped calling one of its halves, the aggregate would still pass while that half went
/// unchecked, and this is what notices.
pub fn shape_and_structural_laws_generated<T>(what: &str)
where
    T: Tensor + Real + PartialEq + core::fmt::Debug + 'static,
{
    let generator = generate::vec(shaped_data(), 1, 3);
    let config = Config::default().cases(300);
    check(what, &config, &generator, |parts| {
        let xs: Vec<T> = sample(&parts);
        shape_laws(&xs, what);
        structural_laws(&xs, what);
        Ok(())
    });
}
