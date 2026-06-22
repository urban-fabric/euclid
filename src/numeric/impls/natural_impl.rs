//! `u32`/`u64`/`u128` implementations of the natural tier — [`Element`] and [`Natural`]. The
//! trait *definition* lives in [`natural`](crate::numeric::natural); this holds the concrete
//! carrier impls (each width-matched to the real it casts into and the signed ring it widens
//! into: `u32 → f32`/`i64`, `u64`/`u128 → f64`/`i128`).

use crate::numeric::{Element, Natural};

impl Natural for u32 {
    type Real = f32;
    type Integer = i64;
    #[inline]
    fn from_u64(x: u64) -> Self {
        x as u32
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
        crate::numeric::special::host::factorial::ln_factorial_u32(*self)
    }
    #[inline]
    fn factorial(&self) -> f32 {
        crate::numeric::special::host::factorial::factorial_u32(*self)
    }
    #[inline]
    fn ln_binomial(&self, k: &Self) -> f32 {
        crate::numeric::special::host::factorial::ln_binomial_u32(*self, *k)
    }
    #[inline]
    fn binomial(&self, k: &Self) -> f32 {
        crate::numeric::special::host::factorial::binomial_u32(*self, *k)
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
        crate::numeric::special::host::factorial::ln_factorial(*self)
    }
    #[inline]
    fn factorial(&self) -> f64 {
        crate::numeric::special::host::factorial::factorial(*self)
    }
    #[inline]
    fn ln_binomial(&self, k: &Self) -> f64 {
        crate::numeric::special::host::factorial::ln_binomial(*self, *k)
    }
    #[inline]
    fn binomial(&self, k: &Self) -> f64 {
        crate::numeric::special::host::factorial::binomial(*self, *k)
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
        if *self <= 20 {
            let mut sum = 0.0;
            for k in 2..=*self {
                sum += (k as f64).ln();
            }
            sum
        } else {
            let n = *self as f64;
            crate::numeric::special::gamma::ln_gamma(n + 1.0)
        }
    }
    #[inline]
    fn factorial(&self) -> f64 {
        if *self <= 20 {
            let mut acc = 1.0;
            for k in 2..=*self {
                acc *= k as f64;
            }
            acc
        } else {
            self.ln_factorial().exp()
        }
    }
    #[inline]
    fn ln_binomial(&self, k: &Self) -> f64 {
        if k > self {
            return f64::NEG_INFINITY;
        }
        let n = *self as f64;
        let k = *k as f64;
        crate::numeric::special::gamma::ln_gamma(n + 1.0)
            - crate::numeric::special::gamma::ln_gamma(k + 1.0)
            - crate::numeric::special::gamma::ln_gamma(n - k + 1.0)
    }
    #[inline]
    fn binomial(&self, k: &Self) -> f64 {
        if k > self {
            0.0
        } else {
            self.ln_binomial(k).exp().round()
        }
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
