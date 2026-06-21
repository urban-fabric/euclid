//! `f32` implementations of the numeric traits.

use crate::numeric::{Arithmetic, LazyReal, NormedDivisionAlgebra};

impl Arithmetic for f32 {}

impl LazyReal for f32 {
    type Integer = i32;
    #[inline]
    fn scalar(x: f64) -> Self {
        x as f32
    }
    #[inline]
    fn exp(&self) -> Self {
        f32::exp(*self)
    }
    #[inline]
    fn ln(&self) -> Self {
        f32::ln(*self)
    }
    #[inline]
    fn sqrt(&self) -> Self {
        f32::sqrt(*self)
    }
    #[inline]
    fn abs(&self) -> Self {
        f32::abs(*self)
    }
    #[inline]
    fn powf(&self, p: Self) -> Self {
        f32::powf(*self, p)
    }
    #[inline]
    fn ln_1p_exp(&self) -> Self {
        let x = *self;
        if x > 0.0 {
            x + f32::ln_1p((-x).exp())
        } else {
            f32::ln_1p(x.exp())
        }
    }
    #[inline]
    fn sin(self) -> Self {
        f32::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        f32::cos(self)
    }
    #[inline]
    fn tan(self) -> Self {
        f32::tan(self)
    }
    #[inline]
    fn asin(self) -> Self {
        f32::asin(self)
    }
    #[inline]
    fn acos(self) -> Self {
        f32::acos(self)
    }
    #[inline]
    fn atan(self) -> Self {
        f32::atan(self)
    }
    #[inline]
    fn atan2(self, x: Self) -> Self {
        f32::atan2(self, x)
    }
    #[inline]
    fn sinh(self) -> Self {
        f32::sinh(self)
    }
    #[inline]
    fn cosh(self) -> Self {
        f32::cosh(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        f32::tanh(self)
    }
    #[inline]
    fn asinh(self) -> Self {
        f32::asinh(self)
    }
    #[inline]
    fn acosh(self) -> Self {
        f32::acosh(self)
    }
    #[inline]
    fn atanh(self) -> Self {
        f32::atanh(self)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        f32::hypot(self, other)
    }
    #[inline]
    fn cbrt(self) -> Self {
        f32::cbrt(self)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        f32::powi(self, n)
    }
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f32::mul_add(self, a, b)
    }
    #[inline]
    fn recip(self) -> Self {
        f32::recip(self)
    }
    #[inline]
    fn log2(self) -> Self {
        f32::log2(self)
    }
    #[inline]
    fn floor(self) -> Self {
        f32::floor(self)
    }
    #[inline]
    fn ceil(self) -> Self {
        f32::ceil(self)
    }
    #[inline]
    fn round(self) -> Self {
        f32::round(self)
    }
    #[inline]
    fn trunc(self) -> Self {
        f32::trunc(self)
    }
    #[inline]
    fn fract(self) -> Self {
        f32::fract(self)
    }
    #[inline]
    fn signum(self) -> Self {
        f32::signum(self)
    }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        f32::copysign(self, sign)
    }
    #[inline]
    fn rem_euclid(self, m: Self) -> Self {
        f32::rem_euclid(self, m)
    }
    #[inline]
    fn to_radians(self) -> Self {
        f32::to_radians(self)
    }
    #[inline]
    fn to_degrees(self) -> Self {
        f32::to_degrees(self)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        f32::min(self, other)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }
    #[inline]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        f32::clamp(self, lo, hi)
    }
    // `f32` has no native special functions: evaluate in `f64` and narrow, the same
    // way `scalar` already narrows a host constant.
    #[inline]
    fn erf(self) -> Self {
        crate::numeric::special::erf::erf(self as f64) as f32
    }
    #[inline]
    fn erfc(self) -> Self {
        crate::numeric::special::erf::erfc(self as f64) as f32
    }
    #[inline]
    fn erf_inv(self) -> Self {
        crate::numeric::special::erf::erf_inv(self as f64) as f32
    }
    #[inline]
    fn ln_gamma(self) -> Self {
        crate::numeric::special::gamma::ln_gamma(self as f64) as f32
    }
    #[inline]
    fn digamma(self) -> Self {
        crate::numeric::special::gamma::digamma(self as f64) as f32
    }
    #[inline]
    fn trigamma(self) -> Self {
        crate::numeric::special::gamma::trigamma(self as f64) as f32
    }
    #[inline]
    fn gamma_lr(self, shape: Self) -> Self {
        crate::numeric::special::gamma::gamma_lr(shape as f64, self as f64) as f32
    }
    #[inline]
    fn gamma_ur(self, shape: Self) -> Self {
        crate::numeric::special::gamma::gamma_ur(shape as f64, self as f64) as f32
    }
    #[inline]
    fn beta_reg(self, a: Self, b: Self) -> Self {
        crate::numeric::special::beta::beta_reg(a as f64, b as f64, self as f64) as f32
    }
    #[inline]
    fn ln_1m_exp(&self) -> Self {
        let x = *self;
        if x >= 0.0 {
            f32::NAN
        } else if x > -core::f32::consts::LN_2 {
            (-x.exp_m1()).ln()
        } else {
            (-x.exp()).ln_1p()
        }
    }
    #[inline]
    fn erfcx(self) -> Self {
        crate::numeric::special::erf::erfcx(self as f64) as f32
    }
    #[inline]
    fn bessel_i0e(self) -> Self {
        crate::numeric::special::bessel::i0e(self as f64) as f32
    }
    #[inline]
    fn bessel_i1e(self) -> Self {
        crate::numeric::special::bessel::i1e(self as f64) as f32
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
