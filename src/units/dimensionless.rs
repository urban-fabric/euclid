//! The [`Dimensionless`] quantity contract — pure ratios: factors, gains, percentages.

use super::LinearQuantity;

/// A dimensionless quantity: a pure number that carries no SI base dimension (a scale factor, a
/// gain, a percentage, an efficiency). Its base signature is `[0; 7]`, like
/// [`Angle`](super::Angle) and [`SolidAngle`](super::SolidAngle) — but unlike those, it is *not*
/// a tracked geometric name; it is the multiplicative identity dimension itself.
///
/// Unlike the other dimension contracts, [`Dimensionless`] is **closed under multiplication and
/// reciprocal**: a ratio times a ratio is still a ratio, and so is its inverse. (Multiplying a
/// *dimensioned* quantity by a factor is [`LinearQuantity::scale`], which stays in the dimensioned
/// quantity's own type.)
pub trait Dimensionless: LinearQuantity {
    /// The multiplicative identity, `1`.
    fn unit() -> Self;

    /// `ratio × ratio = ratio` — dimensionless quantities compose under multiplication.
    fn times(self, other: Self) -> Self;

    /// The reciprocal `1 / self`.
    fn recip(self) -> Self;

    /// `ratio / ratio = ratio`.
    #[inline]
    fn per(self, other: Self) -> Self {
        self.times(other.recip())
    }
}
