//! Supporting `f32` numeric impls outside the main `Real` block.

use crate::interpolate::Interpolate;
use crate::numeric::{Arithmetic, Element, NormedDivisionAlgebra, Real};

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
        self + (target - self) * Self::half()
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
