//! The tensor contract over **generated** shapes.
//!
//! The companion to [`conformance_suites`](super::conformance_suites), which runs the same
//! [`crate::conformance`] laws over `Dense`'s fixed five-tensor
//! [`ConformanceSample`](crate::conformance::ConformanceSample). Nothing is restated: the laws
//! are the crate's own public functions taking a slice, and this file changes only where the
//! slice comes from.
//!
//! ## What a generated shape reaches that five tensors do not
//!
//! The fixed sample's job is to prove the suites are *satisfiable* and that every rank the laws
//! need is present — its header says the rank-3 entry is load-bearing because the 3-cycle law
//! only applies at rank ≥ 3. It does that well. What it cannot do is vary the *extents*, and
//! the extents are where a structural bug hides:
//!
//! - **Repeated extents.** The sample's rank-3 case is `[2, 3, 4]`, every extent distinct, so a
//!   permutation applied in the wrong direction produces the wrong *shape* and is caught by
//!   `shape()` alone. At `[3, 3, 3]` a reversed permutation produces the **right shape and the
//!   wrong data**, so only the values catch it. That is the one case where a backend can be
//!   wrong and still look right to every shape assertion.
//! - **Unit axes.** An extent of `1` makes transpose and reverse no-ops along that axis, so a
//!   backend that drops such an axis — or that computes a stride as `0` — passes an involution
//!   law while corrupting the shape. The fixed sample has one unit extent, in one position.
//! - **Rank 4.** The laws are written for arbitrary rank and the sample stops at 3, so the
//!   generic path above the 3-cycle is type-checked but never executed.
//!
//! ## Beyond the laws: an arbitrary permutation
//!
//! `structural_laws` checks the identity permutation, a swap, and a 3-cycle — the three it can
//! name without knowing the rank. A generated permutation and its inverse covers every element
//! of the symmetric group the rank admits, which is the strongest statement available here and
//! still needs no reference value: `transpose(p)` followed by `transpose(p⁻¹)` is the input, for
//! whichever direction the backend composes in, because the two are inverse group elements
//! either way.
//!
//! ## The shape comes first, and everything else depends on it
//!
//! A tensor's data length is its shape's product and a permutation's length is its rank, so both
//! generators are *dependent* on a value drawn earlier. This is the case the choice tape exists
//! for: shrinking a shape re-draws the data and the permutation under the new rank, so every
//! candidate stays well-formed by construction and there is no filter to exhaust.

use euclid_property::{Config, Gen, check, classify, generate, property};

use super::dense::Dense;
use crate::Tensor;
use crate::conformance::{self as laws};

/// Few enough that each case running every law over a rank-4 tensor stays quick, and enough
/// that the weighted extents actually produce a repeated-extent rank-3 case.
fn settings(suite: &str) -> Config {
    Config::default()
        .cases(300)
        .regressions(format!("src/tests/regressions/{suite}.txt"))
}

/// Rank 4 is one above the fixed sample's maximum, which is what exercises the generic path
/// rather than the three permutations the laws can name.
const MAX_RANK: usize = 4;

/// An extent, weighted toward the two that break things.
///
/// `1` because a unit axis makes a permutation unobservable in the shape; `2` and `3` because
/// a small alphabet is what produces *repeated* extents, where a wrong permutation preserves
/// the shape. A uniform draw over a wide range would make every extent distinct and give away
/// every permutation bug for free.
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

/// Finite values, bounded well away from the exponent limits.
///
/// The laws here are structural — permutations, reversals, reshapes — so no arithmetic is
/// performed on these and their magnitude is irrelevant. What matters is that they are finite
/// and distinct: the laws compare with `PartialEq`, so a `NaN` would fail every identity by
/// never equalling itself, which says nothing about the backend.
fn value() -> Gen<f64> {
    generate::float::in_range(-1e6, 1e6)
}

/// A `Dense` with exactly the right number of elements for `shape`.
fn dense_of(shape: Vec<usize>) -> Gen<Dense> {
    let count: usize = shape.iter().product();
    generate::vec(value(), count, count).map(move |data| Dense::new(&shape, data))
}

fn tensor() -> Gen<Dense> {
    shape().and_then(dense_of)
}

/// A permutation of `0..rank`, by argsort of generated keys.
///
/// Built this way rather than by rejection so that every draw is a valid permutation and the
/// shrinker never has to discard a candidate. Ties broken by index, so the result is a total
/// order and therefore a bijection.
fn permutation(rank: usize) -> Gen<Vec<usize>> {
    generate::vec(generate::index(0, 32), rank, rank).map(|keys| {
        let mut axes: Vec<usize> = (0..keys.len()).collect();
        axes.sort_by_key(|&i| (keys[i], i));
        axes
    })
}

/// A tensor paired with a permutation of its own axes — the dependent draw.
fn tensor_and_permutation() -> Gen<(Dense, Vec<usize>)> {
    shape().and_then(|s| {
        let rank = s.len();
        generate::pair(dense_of(s), permutation(rank))
    })
}

/// The inverse of a permutation: `inv[p[i]] = i`.
fn invert(p: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; p.len()];
    for (i, &axis) in p.iter().enumerate() {
        inv[axis] = i;
    }
    inv
}

/// Whether two axes share an extent — the case where a permutation applied backwards still
/// produces the right shape, so only the data can catch it.
fn has_repeated_extents(shape: &[usize]) -> bool {
    let mut sorted = shape.to_vec();
    sorted.sort_unstable();
    let len = sorted.len();
    sorted.dedup();
    sorted.len() < len
}

/// How the extents of a shape relate to each other — the distinction the fixed sample cannot
/// vary, and the reason this suite exists.
fn extents(shape: &[usize]) -> &'static str {
    if shape.is_empty() {
        return "rank 0";
    }
    match (has_repeated_extents(shape), shape.contains(&1)) {
        (true, _) => "repeated extents",
        (false, true) => "a unit axis",
        (false, false) => "all extents distinct",
    }
}

property! {
    #![config(
        settings("tensor-laws")
            .require("a repeated-extent tensor", 10.0)
            .require("a rank-0 tensor", 1.0)
            .require("a rank-4 tensor", 1.0)
    )]

    /// `Dense` against every law the crate states, over generated shapes.
    ///
    /// `tensor_laws` is the aggregate a backend author calls, so this is the same entry point
    /// a consumer would use — over generated shapes rather than five fixed tensors.
    ///
    /// The four labels are independent rather than exclusive, and each has a floor: a run that
    /// never produced a repeated extent, a rank-0 skip or a rank-4 tensor would be reporting
    /// coverage this suite exists to provide and did not actually reach.
    fn dense_satisfies_the_tensor_laws(xs in generate::vec(tensor(), 1, 3)) {
        classify!("a repeated-extent tensor", xs.iter().any(|x| has_repeated_extents(&x.shape())));
        classify!("a rank-0 tensor", xs.iter().any(|x| x.rank() == 0));
        classify!("a rank-4 tensor", xs.iter().any(|x| x.rank() == MAX_RANK));
        classify!("a unit axis", xs.iter().any(|x| x.shape().contains(&1)));
        laws::tensor_laws(&xs, "Dense");
    }
}

property! {
    #![config(settings("tensor-structure"))]

    /// The shape and structural halves, separately.
    ///
    /// Run alongside the aggregate rather than instead of it: if `tensor_laws` ever stopped
    /// calling one of its parts, the aggregate would still pass while that part went
    /// unchecked, and these are what notice.
    ///
    /// In its own block with no coverage floors, because it draws from the same generator as
    /// the aggregate above — the floors are a claim about `tensor()`, so asserting them twice
    /// would duplicate the claim rather than check anything further.
    fn dense_satisfies_the_shape_and_structural_laws(xs in generate::vec(tensor(), 1, 3)) {
        laws::shape_laws(&xs, "Dense");
        laws::structural_laws(&xs, "Dense");
    }
}

property! {
    #![config(
        settings("tensor-permutations")
            .require("repeated extents", 10.0)
    )]

    /// An **arbitrary** permutation composed with its inverse is the identity.
    ///
    /// The generalisation of the three permutations `structural_laws` can name. A backend that
    /// applies a permutation in the wrong direction survives the identity and the swap, and is
    /// caught by the 3-cycle only at rank ≥ 3 — this catches it at every rank, and at the
    /// repeated extents where the shape alone cannot tell the two directions apart.
    fn a_permutation_composed_with_its_inverse_is_the_identity(
        (x, p) in tensor_and_permutation(),
    ) {
        classify!(extents(&x.shape()));
        classify!("identity permutation", p.iter().copied().eq(0..p.len()));

        let inverse = invert(&p);
        let round_trip = x.clone().transpose_axes(&p).transpose_axes(&inverse);
        check!(
            round_trip == x,
            "transpose({p:?}) then transpose({inverse:?}) changed the tensor with shape {:?}",
            x.shape()
        );
    }

    /// A permutation preserves the element count, and permutes the shape.
    ///
    /// The shape-level statement underneath the round trip above: a transpose rearranges axes
    /// and creates or destroys nothing. Worth stating separately because it fails differently —
    /// a backend that silently drops a unit axis keeps the data intact and loses the rank.
    fn a_permutation_rearranges_the_shape_without_changing_the_count(
        (x, p) in tensor_and_permutation(),
    ) {
        classify!(extents(&x.shape()));

        let before = x.shape();
        let after = x.clone().transpose_axes(&p).shape();
        check!(
            after.len() == before.len(),
            "transpose({p:?}) changed the rank of {before:?} to {after:?}"
        );
        check!(
            after.iter().product::<usize>() == before.iter().product::<usize>(),
            "transpose({p:?}) changed the element count of {before:?} to {after:?}"
        );

        let mut want: Vec<usize> = before.clone();
        want.sort_unstable();
        let mut got = after.clone();
        got.sort_unstable();
        check!(
            got == want,
            "transpose({p:?}) of {before:?} gave extents {after:?}, not a rearrangement"
        );
    }
}

property! {
    #![config(settings("tensor-reshape"))]

    /// Reshape round-trips through a **non-trivial** intermediate shape.
    ///
    /// `shape_laws` round-trips through the flattened `[count]`, which a backend can satisfy by
    /// storing a flat buffer and ignoring the shape entirely. Going out through the reversed
    /// shape and back requires the intermediate to be a real rank-n shape, so it separates a
    /// backend that tracks shape from one that merely remembers the original.
    fn reshape_round_trips_through_a_reversed_shape(x in tensor()) {
        let shape = x.shape();
        classify!(extents(&shape));

        let mut reversed = shape.clone();
        reversed.reverse();

        let there = x.clone().reshape(&reversed);
        check!(
            there.shape() == reversed,
            "reshape to {reversed:?} reported {:?}",
            there.shape()
        );
        check!(
            there.reshape(&shape) == x,
            "reshape {shape:?} -> {reversed:?} -> {shape:?} did not round-trip"
        );
    }
}
