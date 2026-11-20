//! The structural laws: what the data does when the shape changes.
//!
//! Reshape, transpose, slice, concatenate — the operations that move elements without computing
//! on them, so every law here is an identity about *where a value ended up*. A backend can
//! satisfy every shape law and still permute the contents, which is what these catch.

use super::*;

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
