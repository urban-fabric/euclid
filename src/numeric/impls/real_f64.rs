//! `f64` implementations of the numeric traits.

use crate::numeric::{Arithmetic, Element, NormedDivisionAlgebra, Real, RealConstants};

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

impl Real for f64 {
    type Integer = i64;
    #[inline]
    fn scalar(x: f64) -> Self {
        x
    }
    #[inline]
    fn scalar_like(&self, x: f64) -> Self {
        x
    }
    #[inline]
    fn two() -> Self {
        2.0
    }
    #[inline]
    fn half() -> Self {
        0.5
    }
    #[inline]
    fn pi() -> Self {
        core::f64::consts::PI
    }
    #[inline]
    fn tau() -> Self {
        core::f64::consts::TAU
    }
    #[inline]
    fn frac_pi_2() -> Self {
        core::f64::consts::FRAC_PI_2
    }
    #[inline]
    fn frac_pi_4() -> Self {
        core::f64::consts::FRAC_PI_4
    }
    #[inline]
    fn sqrt_2() -> Self {
        core::f64::consts::SQRT_2
    }
    #[inline]
    fn frac_1_sqrt_2() -> Self {
        core::f64::consts::FRAC_1_SQRT_2
    }
    #[inline]
    fn ln_2() -> Self {
        core::f64::consts::LN_2
    }
    #[inline]
    fn ln_pi() -> Self {
        core::f64::consts::PI.ln()
    }
    #[inline]
    fn ln_tau() -> Self {
        core::f64::consts::TAU.ln()
    }
    #[inline]
    fn half_ln_tau() -> Self {
        0.5 * core::f64::consts::TAU.ln()
    }
    #[inline]
    fn e() -> Self {
        core::f64::consts::E
    }
    #[inline]
    fn log2_e() -> Self {
        core::f64::consts::LOG2_E
    }
    #[inline]
    fn log10_e() -> Self {
        core::f64::consts::LOG10_E
    }
    #[inline]
    fn ln_10() -> Self {
        core::f64::consts::LN_10
    }
    #[inline]
    fn log2_10() -> Self {
        core::f64::consts::LOG2_10
    }
    #[inline]
    fn log10_2() -> Self {
        core::f64::consts::LOG10_2
    }
    #[inline]
    fn euler_gamma() -> Self {
        0.577_215_664_901_532_9
    }
    #[inline]
    fn infinity() -> Self {
        f64::INFINITY
    }
    #[inline]
    fn neg_infinity() -> Self {
        f64::NEG_INFINITY
    }
    #[inline]
    fn nan() -> Self {
        f64::NAN
    }
    #[inline]
    fn debug_assert_finite(self) -> Self {
        // The eager reference real traps non-finite values in debug builds; release builds
        // erase the check (it carries no semantic meaning).
        debug_assert!(
            self.is_finite(),
            "Real::debug_assert_finite: non-finite f64 {self}"
        );
        self
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
    fn ln_1p(&self) -> Self {
        f64::ln_1p(*self)
    }
    #[inline]
    fn exp_m1(&self) -> Self {
        f64::exp_m1(*self)
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
    fn square(&self) -> Self {
        *self * *self
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
    fn is_nan(&self) -> bool {
        f64::is_nan(*self)
    }
    #[inline]
    fn is_infinite(&self) -> bool {
        f64::is_infinite(*self)
    }
    #[inline]
    fn is_finite(&self) -> bool {
        f64::is_finite(*self)
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
    fn sin_cos(self) -> (Self, Self) {
        f64::sin_cos(self)
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
        powi_i64(self, n)
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
        if self == 0.0 {
            0.0
        } else {
            self * y.ln()
        }
    }
    #[inline]
    fn xlog1py(self, y: Self) -> Self {
        // 0·ln(1 + y) = 0, including y = −1 where the logarithm is −∞.
        if self == 0.0 {
            0.0
        } else {
            self * y.ln_1p()
        }
    }
    #[inline]
    fn ln_1m(&self) -> Self {
        (-*self).ln_1p()
    }
    #[inline]
    fn ln_exp_m1(&self) -> Self {
        let x = *self;
        x + (-x).ln_1m_exp()
    }
    #[inline]
    fn sigmoid(&self) -> Self {
        let x = *self;
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let exp_x = x.exp();
            exp_x / (1.0 + exp_x)
        }
    }
    #[inline]
    fn ln_sigmoid(&self) -> Self {
        -(-*self).ln_1p_exp()
    }
    #[inline]
    fn ln_1m_sigmoid(&self) -> Self {
        -self.ln_1p_exp()
    }
    #[inline]
    fn logit(&self) -> Self {
        self.ln() - self.ln_1m()
    }
    #[inline]
    fn xlog1my(self, y: Self) -> Self {
        self.xlog1py(-y)
    }
    #[inline]
    fn ln_add_exp(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            return f64::NAN;
        }
        if self == f64::NEG_INFINITY {
            return other;
        }
        if other == f64::NEG_INFINITY {
            return self;
        }
        if self == f64::INFINITY || other == f64::INFINITY {
            return f64::INFINITY;
        }

        let m = self.max(other);
        let gap = (self - other).abs();
        m + (-gap).ln_1p_exp()
    }
    #[inline]
    fn ln_sub_exp(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() || self < other {
            return f64::NAN;
        }
        if self == f64::INFINITY && other == f64::INFINITY {
            return f64::NAN;
        }
        if other == f64::NEG_INFINITY {
            return self;
        }
        if self == other {
            return f64::NEG_INFINITY;
        }

        self + (other - self).ln_1m_exp()
    }
    #[inline]
    fn gamma(self) -> Self {
        // The exact Lanczos reference: more accurate than `exp(ln Γ)` and valid for
        // negative `x` through reflection.
        crate::numeric::special::gamma::gamma(self)
    }
    #[inline]
    fn ln_beta(self, b: Self) -> Self {
        crate::numeric::special::beta::ln_beta(self, b)
    }
    #[inline]
    fn beta(self, b: Self) -> Self {
        crate::numeric::special::beta::beta(self, b)
    }
    #[inline]
    fn ln_bessel_i0(self) -> Self {
        crate::numeric::special::bessel::ln_i0(self)
    }
    #[inline]
    fn bessel_i1_over_i0(self) -> Self {
        crate::numeric::special::bessel::i1_over_i0(self)
    }
}

impl RealConstants for f64 {
    #[inline]
    fn neg_one() -> Self {
        -1.0
    }

    #[inline]
    fn full_turn_degrees() -> Self {
        360.0
    }

    #[inline]
    fn half_turn_degrees() -> Self {
        180.0
    }

    #[inline]
    fn quarter_turn_degrees() -> Self {
        90.0
    }

    #[inline]
    fn three_half_turn_degrees() -> Self {
        540.0
    }

    #[inline]
    fn radians_per_degree() -> Self {
        core::f64::consts::PI / 180.0
    }

    #[inline]
    fn degrees_per_radian() -> Self {
        180.0 / core::f64::consts::PI
    }
}

fn powi_i64(base: f64, exponent: i64) -> f64 {
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
