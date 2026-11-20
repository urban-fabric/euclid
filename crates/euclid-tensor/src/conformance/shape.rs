//! The shape algebra: ranks, extents, and the arithmetic of broadcasting.
//!
//! Separated from the structural laws because a shape is checkable without ever holding an
//! element. Everything here is about agreement between a tensor's declared shape and what its
//! operations do to it — which is the half of the contract a backend gets wrong by arithmetic
//! rather than by data movement.

use super::*;

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
