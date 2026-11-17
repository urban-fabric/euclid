//! Scalar-parameterized interpolation contracts.
//!
//! Interpolation is kept separate from the algebra tiers: some values interpolate linearly,
//! while others need a domain-specific path or a numerically stable formula. The trait
//! therefore requires both `lerp` and `midpoint` instead of deriving one from the other. It is
//! deliberately **opt-in** rather than a blanket impl over
//! [`NormedDivisionAlgebra`](crate::NormedDivisionAlgebra): the affine `lerp` is the
//! wrong interpolation for some division algebras (a unit quaternion wants `slerp`), and a
//! blanket impl would prevent any carrier from supplying a stabler or domain-specific path. The
//! `f64` and `f32` reference impls live in the numeric `impls` module.

use crate::Real;

/// Values that can interpolate between two points using a real scalar parameter.
pub trait Interpolate: Clone {
    /// The real scalar field used for interpolation parameters.
    type Scalar: Real;

    /// Interpolate from `self` to `target` at parameter `t`.
    ///
    /// The usual affine convention is `t = 0` at `self` and `t = 1` at `target`, but the
    /// implementation owns the exact numeric formula.
    fn lerp(self, target: Self, t: Self::Scalar) -> Self;

    /// Midpoint between `self` and `target`.
    ///
    /// This is required separately from [`lerp`](Self::lerp) so implementations can use a
    /// more stable or better-defined midpoint formula.
    fn midpoint(self, target: Self) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_interpolation_has_explicit_midpoints() {
        assert_eq!(0.0_f64.lerp(10.0, 0.25), 2.5);
        assert_eq!(0.0_f64.midpoint(10.0), 5.0);
        assert_eq!(0.0_f32.lerp(10.0, 0.25), 2.5);
        assert_eq!(0.0_f32.midpoint(10.0), 5.0);
    }
}
