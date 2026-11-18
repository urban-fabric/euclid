//! Supporting `f64` numeric impls outside the main `Real` block.

use crate::Interpolate;
use crate::{Arithmetic, Element, NormedDivisionAlgebra};

impl Element for f64 {
    #[inline]
    fn zero() -> Self {
        Default::default()
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
    let magnitude = exponent.unsigned_abs();
    let power = pow_unsigned(base, magnitude);
    if exponent.is_negative() {
        if power.is_infinite() {
            // The reciprocal of an overflowing power can still be subnormal.
            return pow_unsigned(base.recip(), magnitude);
        }
        return power.recip();
    }
    power
}

fn pow_unsigned(base: f64, mut n: u64) -> f64 {
    let mut factor = base;
    let mut acc = 1.0;

    for _ in 0..u64::BITS {
        if n == 0 {
            break;
        }
        if n & 1 == 1 {
            acc *= factor;
        }
        n >>= 1;
        if n != 0 {
            factor *= factor;
        }
    }

    acc
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
    use crate::Interpolate;

    #[test]
    fn unsigned_powers_follow_the_bits_of_the_exponent() {
        use super::pow_unsigned;

        for n in 0..=64_u64 {
            let expected = f64::from_bits((1023 + n) << 52);
            assert_eq!(pow_unsigned(2.0, n), expected);
            let negative = if n % 2 == 0 { expected } else { -expected };
            assert_eq!(pow_unsigned(-2.0, n), negative);
        }
        assert_eq!(pow_unsigned(-1.0, u64::MAX), -1.0);
        assert_eq!(pow_unsigned(f64::NAN, 0), 1.0);
    }

    #[test]
    fn integer_powers_preserve_representable_subnormal_reciprocals() {
        for magnitude in 1023_i64..=1074 {
            let expected = f64::from_bits(1 << (1074 - magnitude));
            assert_eq!(
                crate::Real::powi(2.0_f64, -magnitude).to_bits(),
                expected.to_bits()
            );
            let signed = if magnitude % 2 == 0 {
                expected
            } else {
                -expected
            };
            assert_eq!(
                crate::Real::powi(-2.0_f64, -magnitude).to_bits(),
                signed.to_bits()
            );
        }
        assert_eq!(
            crate::Real::powi(-2.0_f64, -1075).to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_eq!(crate::Real::powi(-1.0_f64, i64::MIN), 1.0);
    }

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
