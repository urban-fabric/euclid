use crate::Semiring;

use super::Measure;

/// A [`Measure`] whose carrier is a [`Semiring`] — counts and sizes that add but need not negate.
///
/// This is the contract for integer-backed counting quantities (byte sizes, item counts, raw
/// centisecond/decimeter storage). Addition is the semiring sum; subtraction saturates at zero
/// because the carrier (e.g. `u64`) has no negatives.
pub trait DiscreteQuantity: Measure
where
    Self::Repr: Semiring,
{
    /// The sum of two values, in the same unit.
    fn add(self, other: Self) -> Self {
        Self::from_raw(self.raw() + other.raw())
    }

    /// `self - other`, saturating at the carrier's zero rather than wrapping or underflowing.
    fn saturating_sub(self, other: Self) -> Self;
}
