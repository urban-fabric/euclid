//! `f64` implementations of the numeric traits.

use crate::numeric::{Arithmetic, LazyReal, NormedDivisionAlgebra};

impl Arithmetic for f64 {}

impl LazyReal for f64 {
    type Integer = i64;
    #[inline]
    fn scalar(x: f64) -> Self {
        x
    }
    #[inline]
    fn exp(&self) -> Self {
        f64::exp(*self)
    }
    #[inline]
    fn ln(&self) -> Self {
        f64::ln(*self)
    }
    #[inline]
    fn sqrt(&self) -> Self {
        f64::sqrt(*self)
    }
    #[inline]
    fn abs(&self) -> Self {
        f64::abs(*self)
    }
    #[inline]
    fn powf(&self, p: Self) -> Self {
        f64::powf(*self, p)
    }
    #[inline]
    fn ln_1p_exp(&self) -> Self {
        // Stable softplus: for x > 0 factor out x to avoid overflow in e^x; for
        // x <= 0 evaluate ln(1 + e^x) directly.
        let x = *self;
        if x > 0.0 {
            x + f64::ln_1p((-x).exp())
        } else {
            f64::ln_1p(x.exp())
        }
    }
    #[inline]
    fn sin(self) -> Self {
        f64::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }
    #[inline]
    fn tan(self) -> Self {
        f64::tan(self)
    }
    #[inline]
    fn asin(self) -> Self {
        f64::asin(self)
    }
    #[inline]
    fn acos(self) -> Self {
        f64::acos(self)
    }
    #[inline]
    fn atan(self) -> Self {
        f64::atan(self)
    }
    #[inline]
    fn atan2(self, x: Self) -> Self {
        f64::atan2(self, x)
    }
    #[inline]
    fn sinh(self) -> Self {
        f64::sinh(self)
    }
    #[inline]
    fn cosh(self) -> Self {
        f64::cosh(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        f64::tanh(self)
    }
    #[inline]
    fn asinh(self) -> Self {
        f64::asinh(self)
    }
    #[inline]
    fn acosh(self) -> Self {
        f64::acosh(self)
    }
    #[inline]
    fn atanh(self) -> Self {
        f64::atanh(self)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        f64::hypot(self, other)
    }
    #[inline]
    fn cbrt(self) -> Self {
        f64::cbrt(self)
    }
    #[inline]
    fn powi(self, n: i64) -> Self {
        f64::powi(self, n as i32)
    }
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f64::mul_add(self, a, b)
    }
    #[inline]
    fn recip(self) -> Self {
        f64::recip(self)
    }
    #[inline]
    fn log2(self) -> Self {
        f64::log2(self)
    }
    #[inline]
    fn floor(self) -> Self {
        f64::floor(self)
    }
    #[inline]
    fn ceil(self) -> Self {
        f64::ceil(self)
    }
    #[inline]
    fn round(self) -> Self {
        f64::round(self)
    }
    #[inline]
    fn trunc(self) -> Self {
        f64::trunc(self)
    }
    #[inline]
    fn fract(self) -> Self {
        f64::fract(self)
    }
    #[inline]
    fn signum(self) -> Self {
        f64::signum(self)
    }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        f64::copysign(self, sign)
    }
    #[inline]
    fn rem_euclid(self, m: Self) -> Self {
        f64::rem_euclid(self, m)
    }
    #[inline]
    fn to_radians(self) -> Self {
        f64::to_radians(self)
    }
    #[inline]
    fn to_degrees(self) -> Self {
        f64::to_degrees(self)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }
    #[inline]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        f64::clamp(self, lo, hi)
    }
    #[inline]
    fn erf(self) -> Self {
        crate::numeric::special::erf::erf(self)
    }
    #[inline]
    fn erfc(self) -> Self {
        crate::numeric::special::erf::erfc(self)
    }
    #[inline]
    fn erf_inv(self) -> Self {
        crate::numeric::special::erf::erf_inv(self)
    }
    #[inline]
    fn ln_gamma(self) -> Self {
        crate::numeric::special::gamma::ln_gamma(self)
    }
    #[inline]
    fn digamma(self) -> Self {
        crate::numeric::special::gamma::digamma(self)
    }
    #[inline]
    fn trigamma(self) -> Self {
        crate::numeric::special::gamma::trigamma(self)
    }
    #[inline]
    fn gamma_lr(self, shape: Self) -> Self {
        crate::numeric::special::gamma::gamma_lr(shape, self)
    }
    #[inline]
    fn gamma_ur(self, shape: Self) -> Self {
        crate::numeric::special::gamma::gamma_ur(shape, self)
    }
    #[inline]
    fn beta_reg(self, a: Self, b: Self) -> Self {
        crate::numeric::special::beta::beta_reg(a, b, self)
    }
    #[inline]
    fn ln_1m_exp(&self) -> Self {
        // Stable log1mexp: ln(-expm1(x)) near 0, ln1p(-exp(x)) in the tail.
        let x = *self;
        if x >= 0.0 {
            f64::NAN
        } else if x > -core::f64::consts::LN_2 {
            (-x.exp_m1()).ln()
        } else {
            (-x.exp()).ln_1p()
        }
    }
    #[inline]
    fn erfcx(self) -> Self {
        crate::numeric::special::erf::erfcx(self)
    }
    #[inline]
    fn bessel_i0e(self) -> Self {
        crate::numeric::special::bessel::i0e(self)
    }
    #[inline]
    fn bessel_i1e(self) -> Self {
        crate::numeric::special::bessel::i1e(self)
    }
    #[inline]
    fn xlogy(self, y: Self) -> Self {
        // 0·ln 0 = 0 (and 0·ln y = 0): short-circuit the absorbing zero so the
        // multiply never forms 0·(−∞) = NaN.
        if self == 0.0 { 0.0 } else { self * y.ln() }
    }
    #[inline]
    fn gamma(self) -> Self {
        // The exact Lanczos reference: more accurate than `exp(ln Γ)` and valid for
        // negative `x` (via reflection), which the defaulted form is not.
        crate::numeric::special::gamma::gamma(self)
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
