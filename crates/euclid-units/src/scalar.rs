use euclid_scalar::{Compare, Real};

use super::{Mask, Quantity};

/// Uniform scalar helpers for quantities.
///
/// Every method has a default body written over the carrier's own
/// [`Real`] surface, so a carrier opts in with an empty impl block:
///
/// ```ignore
/// impl ScalarQuantity for LengthMeters {}
/// ```
///
/// The defaults are not a lowest-common-denominator fallback: each one delegates straight to the
/// carrier's own operation, so a tensor or expression carrier's `min` *is* this `min`. Overriding
/// one is possible but rarely useful — there is little a carrier could supply that would beat
/// delegating to itself.
pub trait ScalarQuantity: Quantity {
    /// The comparison-free lesser ([`Real::min`], NaN-aware on the reals).
    #[inline]
    fn min(self, other: Self) -> Self {
        Self::from_raw(self.raw().min(other.raw()))
    }

    /// The comparison-free greater ([`Real::max`]).
    #[inline]
    fn max(self, other: Self) -> Self {
        Self::from_raw(self.raw().max(other.raw()))
    }

    /// Clamp into `[lo, hi]` ([`Real::clamp`]).
    #[inline]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        Self::from_raw(self.raw().clamp(lo.raw(), hi.raw()))
    }

    /// Round toward negative infinity, staying in this unit.
    #[inline]
    fn floor(self) -> Self {
        Self::from_raw(self.raw().floor())
    }

    /// Round toward positive infinity, staying in this unit.
    #[inline]
    fn ceil(self) -> Self {
        Self::from_raw(self.raw().ceil())
    }

    /// Round to nearest, staying in this unit.
    #[inline]
    fn round(self) -> Self {
        Self::from_raw(self.raw().round())
    }

    /// Truncate toward zero, staying in this unit.
    #[inline]
    fn trunc(self) -> Self {
        Self::from_raw(self.raw().trunc())
    }

    /// Finiteness as a [`Mask`]. Compose it with `select` rather than branching on it.
    #[inline]
    fn is_finite(&self) -> Mask<Self> {
        self.raw().is_finite()
    }

    /// Whether the value is `NaN`, as a [`Mask`].
    #[inline]
    fn is_nan(&self) -> Mask<Self> {
        self.raw().is_nan()
    }

    /// Whether the value is an infinity, as a [`Mask`].
    #[inline]
    fn is_infinite(&self) -> Mask<Self> {
        self.raw().is_infinite()
    }

    /// Approximate equality within a same-unit tolerance, as a [`Mask`].
    ///
    /// The tolerance is itself a quantity rather than a bare carrier value, so its dimension
    /// cannot be confused with another unit's. Its magnitude is used, so a negative tolerance
    /// behaves as its absolute value rather than rejecting everything.
    #[inline]
    fn approx_eq(&self, other: &Self, tolerance: &Self) -> Mask<Self> {
        (self.raw() - other.raw()).abs().le(&tolerance.raw().abs())
    }
}
