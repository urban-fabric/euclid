//! `u32`/`u64`/`u128` implementations of the natural tier — [`Element`] and [`Natural`]. The
//! trait *definition* lives in [`natural`](crate::natural); this holds the concrete
//! carrier impls (each width-matched to the real it casts into and the signed ring it widens
//! into: `u32 → f32`/`i64`, `u64`/`u128 → f64`/`i128`).

use crate::{Element, Natural};

impl Natural for u32 {
    type Real = f32;
    type Integer = i64;
    #[inline]
    fn from_u64(x: u64) -> Self {
        x as u32
    }
    #[inline]
    fn from_real(r: f32) -> Self {
        r.round() as u32
    }
    #[inline]
    fn to_real(&self) -> f32 {
        *self as f32
    }
    #[inline]
    fn to_integer(&self) -> i64 {
        *self as i64
    }
    #[inline]
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self / rhs, self % rhs)
    }
    #[inline]
    fn ln_factorial(&self) -> f32 {
        crate::kernels::host::factorial::ln_factorial_u32(*self)
    }
    #[inline]
    fn factorial(&self) -> f32 {
        crate::kernels::host::factorial::factorial_u32(*self)
    }
    #[inline]
    fn ln_binomial(&self, k: &Self) -> f32 {
        crate::kernels::host::factorial::ln_binomial_u32(*self, *k)
    }
    #[inline]
    fn binomial(&self, k: &Self) -> f32 {
        crate::kernels::host::factorial::binomial_u32(*self, *k)
    }
    #[inline]
    fn gcd(self, other: Self) -> Self {
        // Euclid. Total on the naturals: no overflow case, since the result only ever shrinks.
        let (mut a, mut b) = (self, other);
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
        // Divide before multiplying so a representable answer never overflows on the way.
        (self / Natural::gcd(self, other)) * other
    }
    #[inline]
    fn isqrt(self) -> Self {
        u32::isqrt(self)
    }
}

impl Natural for u64 {
    type Real = f64;
    type Integer = i128;
    #[inline]
    fn from_u64(x: u64) -> Self {
        x
    }
    #[inline]
    fn from_real(r: f64) -> Self {
        r.round() as u64
    }
    #[inline]
    fn to_real(&self) -> f64 {
        *self as f64
    }
    #[inline]
    fn to_integer(&self) -> i128 {
        *self as i128
    }
    #[inline]
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self / rhs, self % rhs)
    }
    #[inline]
    fn ln_factorial(&self) -> f64 {
        crate::kernels::host::factorial::ln_factorial(*self)
    }
    #[inline]
    fn factorial(&self) -> f64 {
        crate::kernels::host::factorial::factorial(*self)
    }
    #[inline]
    fn ln_binomial(&self, k: &Self) -> f64 {
        crate::kernels::host::factorial::ln_binomial(*self, *k)
    }
    #[inline]
    fn binomial(&self, k: &Self) -> f64 {
        crate::kernels::host::factorial::binomial(*self, *k)
    }
    #[inline]
    fn gcd(self, other: Self) -> Self {
        // Euclid. Total on the naturals: no overflow case, since the result only ever shrinks.
        let (mut a, mut b) = (self, other);
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
        // Divide before multiplying so a representable answer never overflows on the way.
        (self / Natural::gcd(self, other)) * other
    }
    #[inline]
    fn isqrt(self) -> Self {
        u64::isqrt(self)
    }
}

impl Natural for u128 {
    type Real = f64;
    type Integer = i128;
    #[inline]
    fn from_u64(x: u64) -> Self {
        x as u128
    }
    #[inline]
    fn from_real(r: f64) -> Self {
        r.round() as u128
    }
    #[inline]
    fn to_real(&self) -> f64 {
        *self as f64
    }
    #[inline]
    fn to_integer(&self) -> i128 {
        *self as i128
    }
    #[inline]
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self / rhs, self % rhs)
    }
    // The combinatorial four delegate to the same `u128` host bodies the `u64` carrier
    // reaches, rather than restating them here. They were restated once, and the copy drifted:
    // it differenced `ln_binomial` straight off `ln_gamma` where the shared body uses the
    // exact small sum, and its `binomial` carried the `f64` accumulator whose intermediate
    // rounds a factor of `k` below `2^53`. Two carriers of one contract answering the same
    // combinatorial question differently is the disagreement `conformance` exists to prevent.
    #[inline]
    fn ln_factorial(&self) -> f64 {
        crate::kernels::host::factorial::ln_factorial_u128(*self)
    }
    #[inline]
    fn factorial(&self) -> f64 {
        crate::kernels::host::factorial::factorial_u128(*self)
    }
    #[inline]
    fn ln_binomial(&self, k: &Self) -> f64 {
        crate::kernels::host::factorial::ln_binomial_u128(*self, *k)
    }
    #[inline]
    fn binomial(&self, k: &Self) -> f64 {
        crate::kernels::host::factorial::binomial_u128(*self, *k)
    }
    #[inline]
    fn gcd(self, other: Self) -> Self {
        // Euclid. Total on the naturals: no overflow case, since the result only ever shrinks.
        let (mut a, mut b) = (self, other);
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
        // Divide before multiplying so a representable answer never overflows on the way.
        (self / Natural::gcd(self, other)) * other
    }
    #[inline]
    fn isqrt(self) -> Self {
        u128::isqrt(self)
    }
}

impl Element for u32 {
    #[inline]
    fn zero() -> Self {
        0
    }
    #[inline]
    fn one() -> Self {
        1
    }
}

impl Element for u64 {
    #[inline]
    fn zero() -> Self {
        0
    }
    #[inline]
    fn one() -> Self {
        1
    }
}

impl Element for u128 {
    #[inline]
    fn zero() -> Self {
        0
    }
    #[inline]
    fn one() -> Self {
        1
    }
}
