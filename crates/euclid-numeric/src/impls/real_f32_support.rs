//! Supporting `f32` numeric impls outside the main `Real` block.

use crate::interpolate::Interpolate;
use crate::{Arithmetic, Element, NormedDivisionAlgebra};

impl Element for f32 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
}

impl Arithmetic for f32 {}

impl Interpolate for f32 {
    type Scalar = f32;

    #[inline]
    fn lerp(self, target: Self, t: Self::Scalar) -> Self {
        self + (target - self) * t
    }

    #[inline]
    fn midpoint(self, target: Self) -> Self {
        // `f32::midpoint` (std, stabilized in 1.85) is overflow- and sign-stable; the
        // earlier `self + (target - self) * ½` overflowed to ±∞ for large opposite-sign
        // operands. This is exactly the "more stable midpoint" the trait doc anticipates.
        f32::midpoint(self, target)
    }
}

impl NormedDivisionAlgebra for f32 {
    type RealPart = f32;
    #[inline]
    fn conjugate(&self) -> Self {
        *self
    }
    #[inline]
    fn norm_sqr(&self) -> f32 {
        self * self
    }
    #[inline]
    fn real_part(&self) -> f32 {
        *self
    }
    #[inline]
    fn from_real(re: f32) -> Self {
        re
    }
    #[inline]
    fn magnitude(&self) -> f32 {
        f32::abs(*self)
    }
}
