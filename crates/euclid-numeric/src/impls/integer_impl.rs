//! `i32`/`i64`/`i128` implementations of the integer tier — [`Element`] and [`Integer`]. The
//! trait *definition* lives in [`integer`](crate::integer); this holds the concrete
//! carrier impls (each width-matched to the real it casts into: `i32 → f32`, `i64`/`i128 → f64`).

use crate::{Element, Integer};

impl Integer for i32 {
    type Real = f32;
    #[inline]
    fn from_i64(x: i64) -> Self {
        x as i32
    }
    #[inline]
    fn from_real(r: f32) -> Self {
        r.round() as i32
    }
    #[inline]
    fn to_real(&self) -> f32 {
        *self as f32
    }
    #[inline]
    fn signum(self) -> Self {
        i32::signum(self)
    }
    #[inline]
    fn abs(self) -> Self {
        i32::abs(self)
    }
    #[inline]
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self.div_euclid(rhs), self.rem_euclid(rhs))
    }
    #[inline]
    fn gcd(self, other: Self) -> Self {
        // On the magnitudes, so the result is the non-negative representative.
        let (mut a, mut b) = (self.abs(), other.abs());
        while b != 0 {
            let r = a % b;
            a = b;
            b = r;
        }
        a
    }
    #[inline]
    fn lcm(self, other: Self) -> Self {
        if self == 0 || other == 0 {
            return 0;
        }
        let g = Integer::gcd(self, other);
        ((self / g) * other).abs()
    }
}

impl Integer for i64 {
    type Real = f64;
    #[inline]
    fn from_i64(x: i64) -> Self {
        x
    }
    #[inline]
    fn from_real(r: f64) -> Self {
        r.round() as i64
    }
    #[inline]
    fn to_real(&self) -> f64 {
        *self as f64
    }
    #[inline]
    fn signum(self) -> Self {
        i64::signum(self)
    }
    #[inline]
    fn abs(self) -> Self {
        i64::abs(self)
    }
    #[inline]
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self.div_euclid(rhs), self.rem_euclid(rhs))
    }
    #[inline]
    fn gcd(self, other: Self) -> Self {
        // On the magnitudes, so the result is the non-negative representative.
        let (mut a, mut b) = (self.abs(), other.abs());
        while b != 0 {
            let r = a % b;
            a = b;
            b = r;
        }
        a
    }
    #[inline]
    fn lcm(self, other: Self) -> Self {
        if self == 0 || other == 0 {
            return 0;
        }
        let g = Integer::gcd(self, other);
        ((self / g) * other).abs()
    }
}

impl Integer for i128 {
    type Real = f64;
    #[inline]
    fn from_i64(x: i64) -> Self {
        x as i128
    }
    #[inline]
    fn from_real(r: f64) -> Self {
        r.round() as i128
    }
    #[inline]
    fn to_real(&self) -> f64 {
        *self as f64
    }
    #[inline]
    fn signum(self) -> Self {
        i128::signum(self)
    }
    #[inline]
    fn abs(self) -> Self {
        i128::abs(self)
    }
    #[inline]
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self.div_euclid(rhs), self.rem_euclid(rhs))
    }
    #[inline]
    fn gcd(self, other: Self) -> Self {
        // On the magnitudes, so the result is the non-negative representative.
        let (mut a, mut b) = (self.abs(), other.abs());
        while b != 0 {
            let r = a % b;
            a = b;
            b = r;
        }
        a
    }
    #[inline]
    fn lcm(self, other: Self) -> Self {
        if self == 0 || other == 0 {
            return 0;
        }
        let g = Integer::gcd(self, other);
        ((self / g) * other).abs()
    }
}

impl Element for i32 {
    #[inline]
    fn zero() -> Self {
        0
    }
    #[inline]
    fn one() -> Self {
        1
    }
}

impl Element for i64 {
    #[inline]
    fn zero() -> Self {
        0
    }
    #[inline]
    fn one() -> Self {
        1
    }
}

impl Element for i128 {
    #[inline]
    fn zero() -> Self {
        0
    }
    #[inline]
    fn one() -> Self {
        1
    }
}
