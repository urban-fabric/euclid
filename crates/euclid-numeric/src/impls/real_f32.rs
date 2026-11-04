//! `f32` implementation of the `Real` trait.

use crate::Real;

impl Real for f32 {
    type Integer = i32;
    fn scalar(x: f64) -> Self {
        x as f32
    }
    fn scalar_like(&self, x: f64) -> Self {
        x as f32
    }
    fn two() -> Self {
        2.0
    }
    fn half() -> Self {
        0.5
    }
    fn pi() -> Self {
        core::f32::consts::PI
    }
    fn tau() -> Self {
        core::f32::consts::TAU
    }
    fn frac_pi_2() -> Self {
        core::f32::consts::FRAC_PI_2
    }
    fn frac_pi_4() -> Self {
        core::f32::consts::FRAC_PI_4
    }
    fn sqrt_2() -> Self {
        core::f32::consts::SQRT_2
    }
    fn frac_1_sqrt_2() -> Self {
        core::f32::consts::FRAC_1_SQRT_2
    }
    fn ln_2() -> Self {
        core::f32::consts::LN_2
    }
    fn ln_pi() -> Self {
        core::f32::consts::PI.ln()
    }
    fn ln_tau() -> Self {
        core::f32::consts::TAU.ln()
    }
    fn half_ln_tau() -> Self {
        0.5 * core::f32::consts::TAU.ln()
    }
    fn e() -> Self {
        core::f32::consts::E
    }
    fn log2_e() -> Self {
        core::f32::consts::LOG2_E
    }
    fn log10_e() -> Self {
        core::f32::consts::LOG10_E
    }
    fn ln_10() -> Self {
        core::f32::consts::LN_10
    }
    fn log2_10() -> Self {
        core::f32::consts::LOG2_10
    }
    fn log10_2() -> Self {
        core::f32::consts::LOG10_2
    }
    fn euler_gamma() -> Self {
        0.577_215_7
    }
    fn infinity() -> Self {
        f32::INFINITY
    }
    fn neg_infinity() -> Self {
        f32::NEG_INFINITY
    }
    fn nan() -> Self {
        f32::NAN
    }
    fn debug_assert_finite(self) -> Self {
        // The eager reference real traps non-finite values in debug builds; release builds
        // erase the check (it carries no semantic meaning).
        debug_assert!(
            self.is_finite(),
            "Real::debug_assert_finite: non-finite f32 {self}"
        );
        self
    }
    fn exp(&self) -> Self {
        f32::exp(*self)
    }
    fn ln(&self) -> Self {
        f32::ln(*self)
    }
    fn ln_1p(&self) -> Self {
        f32::ln_1p(*self)
    }
    fn exp_m1(&self) -> Self {
        f32::exp_m1(*self)
    }
    fn sqrt(&self) -> Self {
        f32::sqrt(*self)
    }
    fn abs(&self) -> Self {
        f32::abs(*self)
    }
    fn square(&self) -> Self {
        *self * *self
    }
    fn powf(&self, p: Self) -> Self {
        f32::powf(*self, p)
    }
    fn ln_1p_exp(&self) -> Self {
        let x = *self;
        if x > 0.0 {
            x + f32::ln_1p((-x).exp())
        } else {
            f32::ln_1p(x.exp())
        }
    }
    fn is_nan(&self) -> bool {
        f32::is_nan(*self)
    }
    fn is_infinite(&self) -> bool {
        f32::is_infinite(*self)
    }
    fn is_finite(&self) -> bool {
        f32::is_finite(*self)
    }
    fn sin(self) -> Self {
        f32::sin(self)
    }
    fn cos(self) -> Self {
        f32::cos(self)
    }
    fn tan(self) -> Self {
        f32::tan(self)
    }
    fn asin(self) -> Self {
        f32::asin(self)
    }
    fn acos(self) -> Self {
        f32::acos(self)
    }
    fn atan(self) -> Self {
        f32::atan(self)
    }
    fn atan2(self, x: Self) -> Self {
        f32::atan2(self, x)
    }
    fn sinh(self) -> Self {
        f32::sinh(self)
    }
    fn cosh(self) -> Self {
        f32::cosh(self)
    }
    fn tanh(self) -> Self {
        f32::tanh(self)
    }
    fn asinh(self) -> Self {
        f32::asinh(self)
    }
    fn acosh(self) -> Self {
        f32::acosh(self)
    }
    fn atanh(self) -> Self {
        f32::atanh(self)
    }
    fn sin_cos(self) -> (Self, Self) {
        f32::sin_cos(self)
    }
    fn hypot(self, other: Self) -> Self {
        f32::hypot(self, other)
    }
    fn cbrt(self) -> Self {
        f32::cbrt(self)
    }
    fn powi(self, n: i32) -> Self {
        f32::powi(self, n)
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        f32::mul_add(self, a, b)
    }
    fn recip(self) -> Self {
        f32::recip(self)
    }
    fn log2(self) -> Self {
        f32::log2(self)
    }
    fn floor(self) -> Self {
        f32::floor(self)
    }
    fn ceil(self) -> Self {
        f32::ceil(self)
    }
    fn round(self) -> Self {
        f32::round(self)
    }
    fn trunc(self) -> Self {
        f32::trunc(self)
    }
    fn fract(self) -> Self {
        f32::fract(self)
    }
    fn signum(self) -> Self {
        f32::signum(self)
    }
    fn copysign(self, sign: Self) -> Self {
        f32::copysign(self, sign)
    }
    fn rem_euclid(self, m: Self) -> Self {
        f32::rem_euclid(self, m)
    }
    fn to_radians(self) -> Self {
        f32::to_radians(self)
    }
    fn to_degrees(self) -> Self {
        f32::to_degrees(self)
    }
    fn min(self, other: Self) -> Self {
        f32::min(self, other)
    }
    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }
    fn clamp(self, lo: Self, hi: Self) -> Self {
        f32::clamp(self, lo, hi)
    }
    // Native `f32` special functions (`numeric::special::f32`): evaluated in single
    // precision throughout, with no `f64` round-trip.
    fn erf(self) -> Self {
        crate::special::f32::erf::erf(self)
    }
    fn erfc(self) -> Self {
        crate::special::f32::erf::erfc(self)
    }
    fn erf_inv(self) -> Self {
        crate::special::f32::erf::erf_inv(self)
    }
    fn ln_gamma(self) -> Self {
        crate::special::f32::gamma::ln_gamma(self)
    }
    fn digamma(self) -> Self {
        crate::special::f32::gamma::digamma(self)
    }
    fn trigamma(self) -> Self {
        crate::special::f32::gamma::trigamma(self)
    }
    fn gamma_lr(self, shape: Self) -> Self {
        crate::special::f32::gamma::gamma_lr(shape, self)
    }
    fn gamma_ur(self, shape: Self) -> Self {
        crate::special::f32::gamma::gamma_ur(shape, self)
    }
    fn beta_reg(self, a: Self, b: Self) -> Self {
        crate::special::f32::beta::beta_reg(a, b, self)
    }
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
    fn erfcx(self) -> Self {
        crate::special::f32::erf::erfcx(self)
    }
    fn bessel_i0e(self) -> Self {
        crate::special::f32::bessel::i0e(self)
    }
    fn bessel_i1e(self) -> Self {
        crate::special::f32::bessel::i1e(self)
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
            return f32::NAN;
        }
        if self == f32::NEG_INFINITY {
            return other;
        }
        if other == f32::NEG_INFINITY {
            return self;
        }
        if self == f32::INFINITY || other == f32::INFINITY {
            return f32::INFINITY;
        }

        let m = self.max(other);
        let gap = (self - other).abs();
        m + (-gap).ln_1p_exp()
    }
    fn ln_sub_exp(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() || self < other {
            return f32::NAN;
        }
        if self == f32::INFINITY && other == f32::INFINITY {
            return f32::NAN;
        }
        if other == f32::NEG_INFINITY {
            return self;
        }
        if self == other {
            return f32::NEG_INFINITY;
        }

        self + (other - self).ln_1m_exp()
    }
    fn gamma(self) -> Self {
        // The native single-precision Lanczos reference (accurate + valid for negative `x`).
        crate::special::f32::gamma::gamma(self)
    }
    fn ln_beta(self, b: Self) -> Self {
        crate::special::f32::beta::ln_beta(self, b)
    }
    fn beta(self, b: Self) -> Self {
        crate::special::f32::beta::beta(self, b)
    }
    fn ln_bessel_i0(self) -> Self {
        crate::special::f32::bessel::ln_i0(self)
    }
    fn bessel_i1_over_i0(self) -> Self {
        crate::special::f32::bessel::i1_over_i0(self)
    }

    // --- compensated arithmetic ---

    fn two_sum(self, rhs: Self) -> (Self, Self) {
        // Knuth's two-sum: exact under round-to-nearest, no FMA needed.
        let s = self + rhs;
        let b_virtual = s - self;
        let err = (self - (s - b_virtual)) + (rhs - b_virtual);
        (s, err)
    }

    fn two_product(self, rhs: Self) -> (Self, Self) {
        // With an FMA the discarded remainder is exactly `fma(a, b, -p)`.
        let p = self * rhs;
        let err = f32::mul_add(self, rhs, -p);
        (p, err)
    }

    // --- cancellation-prone elementary forms ---

    fn cos_m1(self) -> Self {
        // cos x - 1 = -2 sin^2(x/2): accurate near zero, where `cos(x) - 1` is not.
        let h = f32::sin(self * 0.5);
        -2.0 * h * h
    }

    fn sinc(self) -> Self {
        if self == 0.0 {
            return 1.0;
        }
        // The Taylor series is better conditioned than sin(x)/x for small |x|, and avoids
        // the 0/0 form entirely.
        if f32::abs(self) < 1e-2 {
            let x2 = self * self;
            return 1.0 - x2 / 6.0 + x2 * x2 / 120.0 - x2 * x2 * x2 / 5040.0;
        }
        f32::sin(self) / self
    }

    fn hypot3(self, y: Self, z: Self) -> Self {
        // Scale by the largest magnitude so no square can overflow or underflow.
        let (a, b, c) = (f32::abs(self), f32::abs(y), f32::abs(z));
        let m = f32::max(f32::max(a, b), c);
        if m == 0.0 {
            return 0.0;
        }
        if !m.is_finite() {
            // An infinite component dominates; a NaN component must still propagate.
            return if a.is_nan() || b.is_nan() || c.is_nan() {
                f32::NAN
            } else {
                f32::INFINITY
            };
        }
        let (a, b, c) = (a / m, b / m, c / m);
        m * f32::sqrt(a * a + b * b + c * c)
    }

    fn ln_rising_factorial(self, n: Self) -> Self {
        crate::special::f32::gamma::ln_rising_factorial(self, n)
    }

    // --- elementary completeness ---

    fn log10(self) -> Self {
        f32::log10(self)
    }
    fn exp2(self) -> Self {
        f32::exp2(self)
    }
    fn erfc_inv(self) -> Self {
        crate::special::f32::normal::erfc_inv(self)
    }

    // --- the standard normal distribution ---

    fn ndtr(self) -> Self {
        crate::special::f32::normal::ndtr(self)
    }
    fn log_ndtr(self) -> Self {
        crate::special::f32::normal::log_ndtr(self)
    }
    fn ndtri(self) -> Self {
        crate::special::f32::normal::ndtri(self)
    }
}
