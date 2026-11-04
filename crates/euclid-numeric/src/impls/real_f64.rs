//! `f64` implementation of the `Real` trait.

use crate::Real;

use super::real_f64_support::powi_i64;

impl Real for f64 {
    type Integer = i64;
    fn scalar(x: f64) -> Self {
        x
    }
    fn scalar_like(&self, x: f64) -> Self {
        x
    }
    fn two() -> Self {
        2.0
    }
    fn half() -> Self {
        0.5
    }
    fn pi() -> Self {
        core::f64::consts::PI
    }
    fn tau() -> Self {
        core::f64::consts::TAU
    }
    fn frac_pi_2() -> Self {
        core::f64::consts::FRAC_PI_2
    }
    fn frac_pi_4() -> Self {
        core::f64::consts::FRAC_PI_4
    }
    fn sqrt_2() -> Self {
        core::f64::consts::SQRT_2
    }
    fn frac_1_sqrt_2() -> Self {
        core::f64::consts::FRAC_1_SQRT_2
    }
    fn ln_2() -> Self {
        core::f64::consts::LN_2
    }
    fn ln_pi() -> Self {
        core::f64::consts::PI.ln()
    }
    fn ln_tau() -> Self {
        core::f64::consts::TAU.ln()
    }
    fn half_ln_tau() -> Self {
        0.5 * core::f64::consts::TAU.ln()
    }
    fn e() -> Self {
        core::f64::consts::E
    }
    fn log2_e() -> Self {
        core::f64::consts::LOG2_E
    }
    fn log10_e() -> Self {
        core::f64::consts::LOG10_E
    }
    fn ln_10() -> Self {
        core::f64::consts::LN_10
    }
    fn log2_10() -> Self {
        core::f64::consts::LOG2_10
    }
    fn log10_2() -> Self {
        core::f64::consts::LOG10_2
    }
    fn euler_gamma() -> Self {
        0.577_215_664_901_532_9
    }
    fn infinity() -> Self {
        f64::INFINITY
    }
    fn neg_infinity() -> Self {
        f64::NEG_INFINITY
    }
    fn nan() -> Self {
        f64::NAN
    }
    fn debug_assert_finite(self) -> Self {
        // The eager reference real traps non-finite values in debug builds; release builds
        // erase the check (it carries no semantic meaning).
        debug_assert!(
            self.is_finite(),
            "Real::debug_assert_finite: non-finite f64 {self}"
        );
        self
    }
    fn exp(&self) -> Self {
        f64::exp(*self)
    }
    fn ln(&self) -> Self {
        f64::ln(*self)
    }
    fn ln_1p(&self) -> Self {
        f64::ln_1p(*self)
    }
    fn exp_m1(&self) -> Self {
        f64::exp_m1(*self)
    }
    fn sqrt(&self) -> Self {
        f64::sqrt(*self)
    }
    fn abs(&self) -> Self {
        f64::abs(*self)
    }
    fn square(&self) -> Self {
        *self * *self
    }
    fn powf(&self, p: Self) -> Self {
        f64::powf(*self, p)
    }
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
    fn is_nan(&self) -> bool {
        f64::is_nan(*self)
    }
    fn is_infinite(&self) -> bool {
        f64::is_infinite(*self)
    }
    fn is_finite(&self) -> bool {
        f64::is_finite(*self)
    }
    fn sin(self) -> Self {
        f64::sin(self)
    }
    fn cos(self) -> Self {
        f64::cos(self)
    }
    fn tan(self) -> Self {
        f64::tan(self)
    }
    fn asin(self) -> Self {
        f64::asin(self)
    }
    fn acos(self) -> Self {
        f64::acos(self)
    }
    fn atan(self) -> Self {
        f64::atan(self)
    }
    fn atan2(self, x: Self) -> Self {
        f64::atan2(self, x)
    }
    fn sinh(self) -> Self {
        f64::sinh(self)
    }
    fn cosh(self) -> Self {
        f64::cosh(self)
    }
    fn tanh(self) -> Self {
        f64::tanh(self)
    }
    fn asinh(self) -> Self {
        f64::asinh(self)
    }
    fn acosh(self) -> Self {
        f64::acosh(self)
    }
    fn atanh(self) -> Self {
        f64::atanh(self)
    }
    fn sin_cos(self) -> (Self, Self) {
        f64::sin_cos(self)
    }
    fn hypot(self, other: Self) -> Self {
        f64::hypot(self, other)
    }
    fn cbrt(self) -> Self {
        f64::cbrt(self)
    }
    fn powi(self, n: i64) -> Self {
        powi_i64(self, n)
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        f64::mul_add(self, a, b)
    }
    fn recip(self) -> Self {
        f64::recip(self)
    }
    fn log2(self) -> Self {
        f64::log2(self)
    }
    fn floor(self) -> Self {
        f64::floor(self)
    }
    fn ceil(self) -> Self {
        f64::ceil(self)
    }
    fn round(self) -> Self {
        f64::round(self)
    }
    fn trunc(self) -> Self {
        f64::trunc(self)
    }
    fn fract(self) -> Self {
        f64::fract(self)
    }
    fn signum(self) -> Self {
        f64::signum(self)
    }
    fn copysign(self, sign: Self) -> Self {
        f64::copysign(self, sign)
    }
    fn rem_euclid(self, m: Self) -> Self {
        f64::rem_euclid(self, m)
    }
    fn to_radians(self) -> Self {
        f64::to_radians(self)
    }
    fn to_degrees(self) -> Self {
        f64::to_degrees(self)
    }
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }
    fn clamp(self, lo: Self, hi: Self) -> Self {
        f64::clamp(self, lo, hi)
    }
    fn erf(self) -> Self {
        crate::special::erf::erf(self)
    }
    fn erfc(self) -> Self {
        crate::special::erf::erfc(self)
    }
    fn erf_inv(self) -> Self {
        crate::special::erf::erf_inv(self)
    }
    fn ln_gamma(self) -> Self {
        crate::special::gamma::ln_gamma(self)
    }
    fn digamma(self) -> Self {
        crate::special::gamma::digamma(self)
    }
    fn trigamma(self) -> Self {
        crate::special::gamma::trigamma(self)
    }
    fn gamma_lr(self, shape: Self) -> Self {
        crate::special::gamma::gamma_lr(shape, self)
    }
    fn gamma_ur(self, shape: Self) -> Self {
        crate::special::gamma::gamma_ur(shape, self)
    }
    fn beta_reg(self, a: Self, b: Self) -> Self {
        crate::special::beta::beta_reg(a, b, self)
    }
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
    fn erfcx(self) -> Self {
        crate::special::erf::erfcx(self)
    }
    fn bessel_i0e(self) -> Self {
        crate::special::bessel::i0e(self)
    }
    fn bessel_i1e(self) -> Self {
        crate::special::bessel::i1e(self)
    }
    fn xlogy(self, y: Self) -> Self {
        // 0·ln 0 = 0 (and 0·ln y = 0): short-circuit the absorbing zero so the
        // multiply never forms 0·(−∞) = NaN.
        if self == 0.0 { 0.0 } else { self * y.ln() }
    }
    fn xlog1py(self, y: Self) -> Self {
        // 0·ln(1 + y) = 0, including y = −1 where the logarithm is −∞.
        if self == 0.0 { 0.0 } else { self * y.ln_1p() }
    }
    fn ln_1m(&self) -> Self {
        (-*self).ln_1p()
    }
    fn ln_exp_m1(&self) -> Self {
        let x = *self;
        x + (-x).ln_1m_exp()
    }
    fn sigmoid(&self) -> Self {
        let x = *self;
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let exp_x = x.exp();
            exp_x / (1.0 + exp_x)
        }
    }
    fn ln_sigmoid(&self) -> Self {
        -(-*self).ln_1p_exp()
    }
    fn ln_1m_sigmoid(&self) -> Self {
        -self.ln_1p_exp()
    }
    fn logit(&self) -> Self {
        self.ln() - self.ln_1m()
    }
    fn xlog1my(self, y: Self) -> Self {
        self.xlog1py(-y)
    }
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
    fn gamma(self) -> Self {
        // The exact Lanczos reference: more accurate than `exp(ln Γ)` and valid for
        // negative `x` through reflection.
        crate::special::gamma::gamma(self)
    }
    fn ln_beta(self, b: Self) -> Self {
        crate::special::beta::ln_beta(self, b)
    }
    fn beta(self, b: Self) -> Self {
        crate::special::beta::beta(self, b)
    }
    fn ln_bessel_i0(self) -> Self {
        crate::special::bessel::ln_i0(self)
    }
    fn bessel_i1_over_i0(self) -> Self {
        crate::special::bessel::i1_over_i0(self)
    }
}
