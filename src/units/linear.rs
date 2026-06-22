use super::ScalarQuantity;

/// A quantity that behaves like a displacement/vector — addition, negation, scaling,
/// interpolation. Opt-in per unit: an absolute scalar such as a temperature does **not**
/// implement it.
pub trait LinearQuantity: ScalarQuantity {
    /// Same-unit sum.
    fn add(self, other: Self) -> Self;

    /// Same-unit difference.
    fn sub(self, other: Self) -> Self;

    /// Negation.
    fn neg(self) -> Self;

    fn abs(self) -> Self;

    /// The sign of the carrier (`Real::signum`).
    fn signum(self) -> Self::Repr;

    /// The dimensionless ratio `self / other`.
    fn ratio(self, other: Self) -> Self::Repr;

    /// Linear interpolation with a carrier-valued `t`: `t = 0` → `self`, `t = 1` → `other`.
    fn lerp(self, other: Self, t: Self::Repr) -> Self;

    /// Scale by a carrier-valued dimensionless factor (e.g. a graph-computed gain).
    fn scale(self, factor: Self::Repr) -> Self;

    /// Scale by a host literal, lifted into the carrier via
    /// [`scalar_like`](crate::Real::scalar_like) so the carrier can preserve any needed context.
    fn scaled(self, k: f64) -> Self;
}
