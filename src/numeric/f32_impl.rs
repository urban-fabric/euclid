//! `f32` implementations of the numeric traits.

use crate::numeric::{Arithmetic, LazyReal, NormedDivisionAlgebra, Real};

impl Arithmetic for f32 {}

impl LazyReal for f32 {
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
    fn powf(&self, p: f64) -> Self {
        f32::powf(*self, p as f32)
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
    // way `scalar`/`powf` already do.
    #[inline]
    fn erf(self) -> Self {
        crate::special::erf::erf(self as f64) as f32
    }
    #[inline]
    fn erfc(self) -> Self {
        crate::special::erf::erfc(self as f64) as f32
    }
    #[inline]
    fn erf_inv(self) -> Self {
        crate::special::erf::erf_inv(self as f64) as f32
    }
    #[inline]
    fn ln_gamma(self) -> Self {
        crate::special::gamma::ln_gamma(self as f64) as f32
    }
    #[inline]
    fn digamma(self) -> Self {
        crate::special::gamma::digamma(self as f64) as f32
    }
    #[inline]
    fn trigamma(self) -> Self {
        crate::special::gamma::trigamma(self as f64) as f32
    }
    #[inline]
    fn gamma_lr(self, shape: Self) -> Self {
        crate::special::gamma::gamma_lr(shape as f64, self as f64) as f32
    }
    #[inline]
    fn gamma_ur(self, shape: Self) -> Self {
        crate::special::gamma::gamma_ur(shape as f64, self as f64) as f32
    }
    #[inline]
    fn beta_reg(self, a: Self, b: Self) -> Self {
        crate::special::beta::beta_reg(a as f64, b as f64, self as f64) as f32
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
        crate::special::erf::erfcx(self as f64) as f32
    }
    #[inline]
    fn bessel_i0e(self) -> Self {
        crate::special::bessel::i0e(self as f64) as f32
    }
    #[inline]
    fn bessel_i1e(self) -> Self {
        crate::special::bessel::i1e(self as f64) as f32
    }
}

// The concrete `Real` impls. `ZERO`/`ONE` come from `algebra::{HasZero, HasOne}`,
// `÷`/`conjugate`/ordering from `algebra::OrderedField`, and the whole elementwise
// analysis (`scalar`/`exp`/`ln`/`sqrt`/trig/rounding/`min`/`max`/`clamp`) from
// `LazyReal`; only the constants and the `bool` predicates are written here.

impl Real for f32 {
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const PI: Self = core::f32::consts::PI;
    const TAU: Self = core::f32::consts::TAU;
    const FRAC_PI_2: Self = core::f32::consts::FRAC_PI_2;
    const FRAC_PI_4: Self = core::f32::consts::FRAC_PI_4;
    const SQRT_2: Self = core::f32::consts::SQRT_2;
    const FRAC_1_SQRT_2: Self = core::f32::consts::FRAC_1_SQRT_2;
    const E: Self = core::f32::consts::E;
    const EPSILON: Self = f32::EPSILON;
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    const NAN: Self = f32::NAN;
    const MAX: Self = f32::MAX;
    const MIN: Self = f32::MIN;
    const MIN_POSITIVE: Self = f32::MIN_POSITIVE;

    #[inline]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
    #[inline]
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
    #[inline]
    fn is_sign_negative(self) -> bool {
        f32::is_sign_negative(self)
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
