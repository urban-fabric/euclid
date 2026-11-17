use euclid_scalar::{Element, Real};

use super::ScalarQuantity;

/// A quantity that behaves like a displacement/vector — addition, negation, scaling,
/// interpolation. Opt-in per unit: an absolute scalar such as a temperature does **not**
/// implement it.
///
/// Every method has a default body over the carrier's own [`Real`] surface,
/// so a linear unit opts in with an empty impl block:
///
/// ```ignore
/// impl LinearQuantity for LengthMeters {}
/// ```
///
/// Unlike [`ScalarQuantity`], this trait is deliberately *not* blanket-implemented. The empty
/// block is the point: it is the per-unit claim that this quantity is a displacement rather than
/// an absolute position, and that claim is exactly what a temperature in Celsius must be able to
/// withhold. Adding 20 °C to 20 °C is meaningless in a way that adding 20 m to 20 m is not, so
/// the compiler is given the chance to say so.
pub trait LinearQuantity: ScalarQuantity {
    /// Same-unit sum.
    #[inline]
    fn add(self, other: Self) -> Self {
        Self::from_raw(self.raw() + other.raw())
    }

    /// Same-unit difference.
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self::from_raw(self.raw() - other.raw())
    }

    /// Negation.
    #[inline]
    fn neg(self) -> Self {
        Self::from_raw(-self.raw())
    }

    /// The magnitude, staying in this unit.
    #[inline]
    fn abs(self) -> Self {
        Self::from_raw(self.raw().abs())
    }

    /// The sign of the carrier ([`Real::signum`]).
    #[inline]
    fn signum(self) -> Self::Repr {
        self.raw().signum()
    }

    /// The dimensionless ratio `self / other`.
    #[inline]
    fn ratio(self, other: Self) -> Self::Repr {
        self.raw() / other.raw()
    }

    /// Linear interpolation with a carrier-valued `t`: `t = 0` → `self`, `t = 1` → `other`.
    #[inline]
    fn lerp(self, other: Self, t: Self::Repr) -> Self {
        let one_minus_t = <Self::Repr as Element>::one() - t.clone();
        Self::from_raw(self.raw() * one_minus_t + other.raw() * t)
    }

    /// Scale by a carrier-valued dimensionless factor (e.g. a graph-computed gain).
    #[inline]
    fn scale(self, factor: Self::Repr) -> Self {
        Self::from_raw(self.raw() * factor)
    }

    /// Scale by a host literal, lifted into the carrier via
    /// [`scalar_like`](euclid_scalar::Real::scalar_like) so the carrier can preserve any needed
    /// context.
    #[inline]
    fn scaled(self, k: f64) -> Self {
        let factor = self.raw().scalar_like(k);
        Self::from_raw(self.raw() * factor)
    }
}
