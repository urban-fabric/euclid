use super::LinearQuantity;

/// An angle-dimensioned quantity, conventionally carried in radians.
pub trait Angle: LinearQuantity {
    /// Wrap a carrier value already in radians.
    fn from_radians(radians: Self::Repr) -> Self;

    /// Wrap a carrier value in degrees, converting to radians.
    fn from_degrees(degrees: Self::Repr) -> Self;

    /// `π` radians (a half turn).
    fn half_turn() -> Self;

    /// `2π` radians (a full turn).
    fn full_turn() -> Self;

    /// The angle in radians.
    fn radians(&self) -> Self::Repr;

    /// The angle in degrees.
    fn degrees(&self) -> Self::Repr;

    /// Sine of the angle.
    fn sin(&self) -> Self::Repr;

    /// Cosine of the angle.
    fn cos(&self) -> Self::Repr;

    /// Tangent of the angle.
    fn tan(&self) -> Self::Repr;

    /// Normalize into `[0, 2π)`.
    fn normalized_unsigned(self) -> Self;

    /// Normalize into `(-π, π]`.
    fn normalized_signed(self) -> Self;
}
