use crate::Ring;

use super::Measure;

/// An absolute position on a [`Measure`]'s axis — an *instant*, as opposed to a displacement.
///
/// Epoch timestamps are the motivating case: two instants subtract to an elapsed amount (a raw
/// carrier delta), and an instant shifted by a delta is another instant. The carrier must be a
/// [`Ring`] so the difference can be signed (e.g. `i64`). Unlike a [`LinearQuantity`], instants do
/// not add to each other or scale — those operations are meaningless for an absolute position.
///
/// [`LinearQuantity`]: super::LinearQuantity
pub trait Instant: Measure
where
    Self::Repr: Ring,
{
    /// The elapsed carrier amount from `origin` to `self` (`self - origin`).
    fn since(self, origin: Self) -> Self::Repr {
        self.raw() - origin.raw()
    }

    /// This instant shifted later by `elapsed` (`self + elapsed`).
    fn advance(self, elapsed: Self::Repr) -> Self {
        Self::from_raw(self.raw() + elapsed)
    }
}
