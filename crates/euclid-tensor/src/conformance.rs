//! Conformance suites — the structural half of the [`Tensor`] contract, as functions you can
//! check a backend against.
//!
//! euclid-tensor ships no tensor, so every implementation of this trait lives somewhere else.
//! A backend whose `transpose_axes` is not an involution, whose `reshape` to its own shape is
//! not the identity, or whose `slice_in_dim` over the full extent drops an element compiles and
//! type-checks exactly like a correct one — and then produces wrong answers in a kernel written
//! once against the contract.
//!
//! # What these suites assert, and what they deliberately do not
//!
//! The trait docs are explicit that *"axis naming, mask representation, broadcast/gather/scatter
//! details, and precision policy are outside this contract"*. Conventions the contract leaves
//! open are not laws, so nothing here asserts one. In particular:
//!
//! - **Reductions are not checked.** Whether `sum(&[])` reduces over all axes or none, and
//!   whether a reduced axis is removed or kept at length 1, is a convention each backend picks.
//!   A suite that asserted either would fail a correct backend that chose the other.
//! - **Broadcast, gather and scatter semantics are not checked**, for the same reason.
//! - **Numeric results are not checked.** That is `euclid-scalar`'s subject; here the question
//!   is whether the *structure* holds.
//!
//! What is left is the part the contract does fix: shape algebra, and the operations that must
//! be identities or involutions whatever the backend. Those are enough to catch the errors that
//! actually happen — a transposed axis order, an off-by-one extent, a lost rank.
//!
//! # Availability
//!
//! Gated behind the `conformance` feature, so a consumer pays nothing unless it asks, and always
//! compiled under `cfg(test)` so this crate's own suites exercise the same code a backend calls.
//!
//! The `generated` submodule sits behind a **second** feature, `conformance-generated`, which
//! enables this one and additionally pulls the property engine. The division: these functions
//! are the contract, so `conformance` resolves no engine at all and is what a consumer binding
//! the laws should take; the generated search is tooling for locating where a carrier fails
//! them, and costs a dependency.
//!
//! These docs are the module's own (`//!`), with no outer `///` on the `mod` declaration: the
//! two forms are concatenated, and an outer comment resolves intra-doc links in the *parent's*
//! scope, so a link to a function in here would silently fail from there.
//!
//! # Samples
//!
//! Supply tensors of **rank 1 or higher**, with at least one of rank 2 or more so the transpose
//! laws are exercised. A rank-0 sample is accepted and simply skips the laws that need an axis.

use crate::Tensor;

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
pub trait ConformanceSample: Sized {
    /// Tensors to check. Include one of **rank 2 or more** so the transpose laws are
    /// exercised, and one of rank 3 or more so the 3-cycle law is — a swap cannot distinguish
    /// a permutation from its inverse.
    fn sample() -> Vec<Self>;
}

/// [`tensor_laws`] over a carrier's own [`ConformanceSample`].
pub fn tensor_laws_for<T>(what: &str)
where
    T: Tensor + ConformanceSample + PartialEq + core::fmt::Debug,
{
    tensor_laws(&T::sample(), what);
}

/// The shape algebra: `rank`, `reshape`, and the shape of each constructor.
///
/// - `rank() == shape().len()`
/// - `reshape` to a tensor's own shape is the identity
/// - `reshape` round-trips through a flattened shape
/// - `full_like` preserves shape
/// - `concatenate` sums the extent along its axis; `stack` adds an axis
///
/// **`zeros_like` is deliberately not checked here**, and this list used to claim it was. It
/// cannot be: `zeros_like` is bounded `where Self: Element`, and these laws take `T: Tensor`
/// precisely so that shape and reshape carry no arithmetic demand — that is why a boolean tensor
/// can run them without pretending to be a ring. Adding the bound to check one defaulted
/// constructor would exclude every carrier the looseness exists for.
///
/// It is also nearly free: `zeros_like` defaults through `zeros` to `full`, so for any carrier
/// using the defaults it preserves shape exactly because `full_like` does. A carrier that
/// **overrides** it has to check that itself, which is the one case this omission matters in.
pub fn shape_laws<T>(xs: &[T], what: &str)
where
    T: Tensor + PartialEq + core::fmt::Debug,
{
    for x in xs {
        let shape = x.shape();

        assert_eq!(
            x.rank(),
            shape.len(),
            "{what}: rank() must equal shape().len()"
        );

        // reshaping to the shape you already have changes nothing
        assert_eq!(
            &x.clone().reshape(&shape),
            x,
            "{what}: reshape to its own shape must be the identity"
        );

        // and flattening then restoring round-trips
        let count: usize = shape.iter().product();
        assert_eq!(
            &x.clone().reshape(&[count]).reshape(&shape),
            x,
            "{what}: reshape must round-trip through a flattened shape"
        );
        assert_eq!(
            x.clone().reshape(&[count]).shape(),
            vec![count],
            "{what}: reshape must report the shape it was given"
        );

        // `*_like` constructors preserve shape
        assert_eq!(
            x.full_like(x.clone()).shape(),
            shape,
            "{what}: full_like must preserve shape"
        );

        if !shape.is_empty() {
            // concatenating a tensor with itself doubles the extent it was joined along
            let joined = T::concatenate(&[x.clone(), x.clone()], 0);
            let mut want = shape.clone();
            want[0] = shape[0] * 2;
            assert_eq!(
                joined.shape(),
                want,
                "{what}: concatenate must sum the extent along its axis"
            );

            // stacking adds an axis
            let stacked = T::stack(&[x.clone(), x.clone()], 0);
            assert_eq!(
                stacked.rank(),
                x.rank() + 1,
                "{what}: stack must add exactly one axis"
            );
            assert_eq!(
                stacked.shape()[0],
                2,
                "{what}: the stacked axis must have the input count as its extent"
            );
        }
    }
}

/// The structural identities and involutions.
///
/// - `transpose_axes` with the identity permutation is the identity
/// - `transpose_axes` with a swap is an involution: doing it twice restores the input
/// - `reverse` is an involution, is the identity over an empty axis set, and reversing a set of
///   axes equals reversing them one at a time
/// - `slice_in_dim` over an axis's full extent with stride 1 is the identity
///
/// An involution is the strongest shape-level law available without fixing a convention: it
/// holds for every correct backend regardless of layout, and it fails for exactly the mistakes
/// that occur in practice — a permutation applied in the wrong direction, or a reversal that
/// forgets a stride.
pub fn structural_laws<T>(xs: &[T], what: &str)
where
    T: Tensor + PartialEq + core::fmt::Debug,
{
    for x in xs {
        let shape = x.shape();
        let rank = shape.len();

        if rank == 0 {
            continue; // no axes to permute, reverse or slice
        }

        // the identity permutation is the identity
        let identity: Vec<usize> = (0..rank).collect();
        assert_eq!(
            &x.transpose_axes(&identity),
            x,
            "{what}: transpose with the identity permutation must be the identity"
        );

        // reversing the same axes twice restores the input
        assert_eq!(
            &x.reverse(&identity).reverse(&identity),
            x,
            "{what}: reverse must be an involution"
        );

        // reversing no axes changes nothing
        assert_eq!(
            &x.reverse(&[]),
            x,
            "{what}: reverse over an empty axis set must be the identity"
        );

        // Reversing a set of axes must equal reversing them one at a time. The involution law
        // above cannot see a backend that silently handles only the first axis — doing that
        // twice is still the identity — so this is what catches a partial implementation.
        let one_at_a_time = identity
            .iter()
            .fold(x.clone(), |acc, &axis| acc.reverse(&[axis]));
        assert_eq!(
            x.reverse(&identity),
            one_at_a_time,
            "{what}: reverse(axes) must equal reversing each axis in turn"
        );

        // a full-extent, unit-stride slice is the identity, on every axis
        for (axis, &extent) in shape.iter().enumerate() {
            assert_eq!(
                &x.slice_in_dim(0, extent, 1, axis),
                x,
                "{what}: a full slice of axis {axis} must be the identity"
            );
        }

        if rank >= 2 {
            // swapping two axes twice restores the input
            let mut swapped = identity.clone();
            swapped.swap(0, 1);
            assert_eq!(
                &x.transpose_axes(&swapped).transpose_axes(&swapped),
                x,
                "{what}: an axis swap must be an involution"
            );

            // and the swap actually permutes the reported shape
            let mut want = shape.clone();
            want.swap(0, 1);
            assert_eq!(
                x.transpose_axes(&swapped).shape(),
                want,
                "{what}: transpose must permute the shape it reports"
            );
        }

        if rank >= 3 {
            // A swap is its own inverse, so the laws above cannot tell a permutation from its
            // inverse — both read the same. A 3-cycle can: `[1, 2, 0]` sends axis 1 to
            // position 0, where its inverse `[2, 0, 1]` sends axis 2 there.
            //
            // Cycle the first three axes and fix the rest, so this is a permutation of `0..rank`
            // at every rank ≥ 3 — which `transpose_axes` requires. Note the range is `0..3` and
            // not `0..rank`: the latter emits `rank` entries before the chain appends the tail,
            // giving a longer-than-rank list with a repeated axis at rank ≥ 4. It read as
            // correct for as long as the only sample reaching this branch was rank 3, where the
            // two ranges coincide and the chain is empty.
            let cycle: Vec<usize> = (0..3).map(|i| (i + 1) % 3).chain(3..rank).collect();
            let want: Vec<usize> = cycle.iter().map(|&a| shape[a]).collect();
            assert_eq!(
                x.transpose_axes(&cycle).shape(),
                want,
                "{what}: transpose(axes) must report shape[axes[i]] at position i — a cycle \
                 distinguishes the permutation from its inverse, where a swap cannot"
            );

            // Applying a 3-cycle three times returns to the start.
            assert_eq!(
                &x.transpose_axes(&cycle)
                    .transpose_axes(&cycle)
                    .transpose_axes(&cycle),
                x,
                "{what}: a 3-cycle applied three times must be the identity"
            );
        }
    }
}

/// Both structural suites, for a carrier that implements the whole core contract.
pub fn tensor_laws<T>(xs: &[T], what: &str)
where
    T: Tensor + PartialEq + core::fmt::Debug,
{
    shape_laws(xs, what);
    structural_laws(xs, what);
}

/// The same laws over **generated shapes**, for a backend that wants the search rather than its
/// own fixed sample.
///
/// # What this ships that a fixed sample cannot
///
/// A [`ConformanceSample`] proves the laws are *satisfiable* and reaches every rank they need.
/// What it cannot do is vary the **extents**, and the extents are where a structural bug hides:
///
/// - **Repeated extents.** At `[2, 3, 4]` a permutation applied in the wrong direction produces
///   the wrong *shape* and `shape()` alone catches it. At `[3, 3, 3]` the wrong direction
///   produces the **right shape and the wrong data**, so only the values catch it. That is the
///   one case where a backend can be wrong and still look right to every shape assertion.
/// - **Unit axes.** An extent of `1` makes transpose and reverse no-ops along that axis, so a
///   backend that drops such an axis — or computes its stride as `0` — passes an involution law
///   while corrupting the rank.
/// - **Rank 4 and up.** The laws are written for arbitrary rank, and a sample that stops at 3
///   leaves the generic path type-checked but never executed.
///
/// That last one is not hypothetical. [`structural_laws`] built a malformed 3-cycle at rank ≥ 4
/// for as long as euclid's own reference sample stopped at rank 3, and the defect surfaced the
/// first time a generated shape reached rank 4. A backend whose own sample stops where euclid's
/// did inherits exactly that blind spot, which is the reason to ship the search rather than
/// describe it.
///
/// # No new trait method was needed
///
/// The carrier is built through [`Tensor::from_host`], which already takes row-major host data
/// and a shape — so euclid generates the shape and the data and the backend supplies nothing. The
/// `Real` bound comes from `from_host` itself and is why this covers real tensors and not boolean
/// ones; a boolean carrier should run the fixed suite.
#[cfg(any(feature = "conformance-generated", test))]
pub mod generated {
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
}
