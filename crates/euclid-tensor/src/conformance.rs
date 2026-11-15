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
//! - **Numeric results are not checked.** That is `euclid-numeric`'s subject; here the question
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
//! These docs are the module's own (`//!`), with no outer `///` on the `mod` declaration: the
//! two forms are concatenated, and an outer comment resolves intra-doc links in the *parent's*
//! scope, so a link to a function in here would silently fail from there.
//!
//! # Samples
//!
//! Supply tensors of **rank 1 or higher**, with at least one of rank 2 or more so the transpose
//! laws are exercised. A rank-0 sample is accepted and simply skips the laws that need an axis.

use crate::Tensor;

/// The shape algebra: `rank`, `reshape`, and the shape of each constructor.
///
/// - `rank() == shape().len()`
/// - `reshape` to a tensor's own shape is the identity
/// - `reshape` round-trips through a flattened shape
/// - `full_like`/`zeros_like` preserve shape
/// - `concatenate` sums the extent along its axis; `stack` adds an axis
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
            let cycle: Vec<usize> = (0..rank).map(|i| (i + 1) % 3).chain(3..rank).collect();
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
