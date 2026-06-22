//! Supporting `f64` numeric impls outside the main `Real` block.

use crate::numeric::{Arithmetic, Element, NormedDivisionAlgebra};

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

    if exponent < 0 {
        acc.recip()
    } else {
        acc
    }
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
