//! Supporting `f64` numeric impls outside the main `Real` block.

use crate::interpolate::Interpolate;
use crate::{Arithmetic, Element, NormedDivisionAlgebra};

impl Element for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
}

impl Arithmetic for f64 {}

impl Interpolate for f64 {
    type Scalar = f64;

    #[inline]
    fn lerp(self, target: Self, t: Self::Scalar) -> Self {
        self + (target - self) * t
    }

    #[inline]
    fn midpoint(self, target: Self) -> Self {
        // `f64::midpoint` is overflow- and sign-stable; the earlier
        // `self + (target - self) * ½` overflowed to ±∞ for large opposite-sign operands.
        f64::midpoint(self, target)
    }
}

pub(super) fn powi_i64(base: f64, exponent: i64) -> f64 {
    let mut n = exponent.unsigned_abs();
    let mut factor = base;
    let mut acc = 1.0;

    while n > 0 {
        if n & 1 == 1 {
            acc *= factor;
        }
        n >>= 1;
        if n > 0 {
            factor *= factor;
        }
    }

    if exponent < 0 { acc.recip() } else { acc }
}

impl NormedDivisionAlgebra for f64 {
    type RealPart = f64;
    #[inline]
    fn conjugate(&self) -> Self {
        *self
    }
    #[inline]
    fn norm_sqr(&self) -> f64 {
        self * self
    }
    #[inline]
    fn real_part(&self) -> f64 {
        *self
    }
    #[inline]
    fn from_real(re: f64) -> Self {
        re
    }
    #[inline]
    fn magnitude(&self) -> f64 {
        f64::abs(*self)
    }
}

#[cfg(test)]
mod tests {
    use crate::interpolate::Interpolate;

    #[test]
    fn midpoint_is_overflow_stable() {
        // Regression: `self + (target - self) * ½` returned -inf for large opposite-sign
        // operands; the stable `f64::midpoint` returns the true (finite) midpoint.
        assert_eq!(Interpolate::midpoint(1e308_f64, -1e308_f64), 0.0);
        assert_eq!(Interpolate::midpoint(f64::MAX, f64::MAX), f64::MAX);
        assert_eq!(Interpolate::midpoint(2.0_f64, 4.0), 3.0);
        // Agrees with lerp at the midpoint for ordinary values.
        assert_eq!(
            Interpolate::midpoint(3.0_f64, 9.0),
            Interpolate::lerp(3.0_f64, 9.0, 0.5)
        );
    }
}
