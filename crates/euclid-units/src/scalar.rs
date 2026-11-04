use super::{Mask, Quantity};

/// Uniform scalar helpers for quantities.
pub trait ScalarQuantity: Quantity {
    /// The comparison-free lesser (`Real::min`, NaN-aware on the reals).
    fn min(self, other: Self) -> Self;

    /// The comparison-free greater (`Real::max`).
    fn max(self, other: Self) -> Self;

    /// Clamp into `[lo, hi]` (`Real::clamp`).
    fn clamp(self, lo: Self, hi: Self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;

    /// Finiteness as a [`Mask`]. Compose it with `select` rather than branching on it.
    fn is_finite(&self) -> Mask<Self>;
    fn is_nan(&self) -> Mask<Self>;
    fn is_infinite(&self) -> Mask<Self>;

    /// Approximate equality within a same-unit tolerance, as a [`Mask`].
    fn approx_eq(&self, other: &Self, tolerance: &Self) -> Mask<Self>;
}
