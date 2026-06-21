//! Scalar-parameterized interpolation contracts.
//!
//! Interpolation is separate from [`VectorSpace`](crate::numeric::VectorSpace): some values
//! interpolate linearly, while others need a domain-specific path or a numerically stable
//! formula. The trait therefore requires both `lerp` and `midpoint` instead of deriving one
//! from the other.

use crate::numeric::{LazyNatural, LazyReal};

/// Values that can interpolate between two points using a real scalar parameter.
pub trait Interpolate: Clone {
    /// The real scalar field used for interpolation parameters.
    type Scalar: LazyReal;

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

/// Interpolate at a natural-number sample index.
///
/// `segments` is the denominator of the interpolation parameter and should be nonzero.
pub fn sample_at<P, N>(start: P, end: P, index: N, segments: N) -> P
where
    P: Interpolate,
    N: LazyNatural<Real = P::Scalar>,
{
    start.lerp(end, index.to_real() / segments.to_real())
}

/// Return `segments + 1` samples including both endpoints.
///
/// When `segments == 0`, this returns only `start`.
pub fn sample<P>(start: P, end: P, segments: u64) -> Vec<P>
where
    P: Interpolate,
{
    if segments == 0 {
        return vec![start];
    }

    let mut values = Vec::with_capacity(segments as usize + 1);
    for i in 0..=segments {
        let t = P::Scalar::scalar(i as f64 / segments as f64);
        values.push(start.clone().lerp(end.clone(), t));
    }
    values
}

impl Interpolate for f64 {
    type Scalar = f64;

    #[inline]
    fn lerp(self, target: Self, t: Self::Scalar) -> Self {
        self + (target - self) * t
    }

    #[inline]
    fn midpoint(self, target: Self) -> Self {
        self + (target - self) * Self::half()
    }
}

impl Interpolate for f32 {
    type Scalar = f32;

    #[inline]
    fn lerp(self, target: Self, t: Self::Scalar) -> Self {
        self + (target - self) * t
    }

    #[inline]
    fn midpoint(self, target: Self) -> Self {
        self + (target - self) * Self::half()
    }
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

    #[test]
    fn natural_sample_index_maps_to_real_parameter() {
        assert_eq!(sample_at(0.0_f64, 10.0, 1_u64, 4_u64), 2.5);
        assert_eq!(sample_at(0.0_f32, 10.0, 1_u32, 4_u32), 2.5);
    }

    #[test]
    fn sampling_includes_both_endpoints() {
        let values = sample(0.0_f64, 1.0, 4);
        assert_eq!(values, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(sample(3.0_f64, 7.0, 0), vec![3.0]);
    }
}
