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
        crate::constants::f64::LN_PI
    }
    fn ln_tau() -> Self {
        crate::constants::f64::LN_TAU
    }
    fn half_ln_tau() -> Self {
        crate::constants::f64::HALF_LN_TAU
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
        crate::constants::f64::EULER_GAMMA
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
    fn sin_cos_degrees(self) -> (Self, Self) {
        // GeographicLib's `sincosd`. Reducing in degrees keeps the residual
        // exactly zero at every multiple of 90, which is what makes the quadrant
        // values exact; `to_radians` first cannot, since `pi/2` is not an f64.
        let r = if f64::is_finite(self) {
            self % 360.0
        } else {
            f64::NAN
        };
        let q = if f64::is_nan(r) {
            0
        } else {
            ((r / 90.0).round()) as i64
        };
        let r = (r - 90.0 * q as f64).to_radians();
        let (s, c) = f64::sin_cos(r);
        // The quadrant rotation. `rem_euclid(4)` is in 0..=3, so the last arm is
        // the `3` case rather than an unreachable one.
        let (mut sinx, cosx) = match q.rem_euclid(4) {
            0 => (s, c),
            1 => (c, -s),
            2 => (-s, -c),
            _ => (-c, s),
        };
        // IEEE 754 F.10.1.12/13: a vanished cosine is `+0`, and a vanished sine
        // carries the argument's sign.
        let cosx = cosx + 0.0;
        if sinx == 0.0 {
            sinx = f64::copysign(sinx, self);
        }
        (sinx, cosx)
    }
    fn polyval(self, coefficients: &[Self]) -> Self {
        // Horner, highest power first. Deliberately not `mul_add`: see the trait.
        let Some((highest, rest)) = coefficients.split_first() else {
            return 0.0;
        };
        let mut acc = *highest;
        for c in rest {
            acc = acc * self + *c;
        }
        acc
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
        crate::kernels::erf::erf(self)
    }
    fn erfc(self) -> Self {
        crate::kernels::erf::erfc(self)
    }
    fn erf_inv(self) -> Self {
        crate::kernels::erf::erf_inv(self)
    }
    fn ln_gamma(self) -> Self {
        crate::kernels::gamma::ln_gamma(self)
    }
    fn digamma(self) -> Self {
        crate::kernels::gamma::digamma(self)
    }
    fn trigamma(self) -> Self {
        crate::kernels::gamma::trigamma(self)
    }
    fn gamma_lr(self, shape: Self) -> Self {
        crate::kernels::gamma::gamma_lr(shape, self)
    }
    fn gamma_ur(self, shape: Self) -> Self {
        crate::kernels::gamma::gamma_ur(shape, self)
    }
    fn beta_reg(self, a: Self, b: Self) -> Self {
        crate::kernels::beta::beta_reg(a, b, self)
    }
    fn ln_1m_exp(&self) -> Self {
        // Stable log1mexp: ln(-expm1(x)) near 0, ln1p(-exp(x)) in the tail.
        let x = *self;
        if x == 0.0 {
            // `ln(1 - e^0) = ln 0`, which has the limit `-inf` from inside the domain —
            // and `ln_1m(1)` already returns `-inf` at the very same mathematical point, so
            // returning `NaN` here was an inconsistency rather than a domain statement.
            // Fixing it also repairs `ln_exp_m1(0)`, which is built from this.
            f64::NEG_INFINITY
        } else if x > 0.0 {
            f64::NAN
        } else if x > -core::f64::consts::LN_2 {
            (-x.exp_m1()).ln()
        } else {
            (-x.exp()).ln_1p()
        }
    }
    fn erfcx(self) -> Self {
        crate::kernels::erf::erfcx(self)
    }
    fn bessel_i0(self) -> Self {
        crate::kernels::bessel::i0(self)
    }
    fn bessel_i1(self) -> Self {
        crate::kernels::bessel::i1(self)
    }
    fn bessel_i0e(self) -> Self {
        crate::kernels::bessel::i0e(self)
    }
    fn bessel_i1e(self) -> Self {
        crate::kernels::bessel::i1e(self)
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
        crate::kernels::gamma::gamma(self)
    }
    fn ln_beta(self, b: Self) -> Self {
        crate::kernels::beta::ln_beta(self, b)
    }
    fn beta(self, b: Self) -> Self {
        crate::kernels::beta::beta(self, b)
    }
    fn ln_bessel_i0(self) -> Self {
        crate::kernels::bessel::ln_i0(self)
    }
    fn bessel_i1_over_i0(self) -> Self {
        crate::kernels::bessel::i1_over_i0(self)
    }

    // --- compensated arithmetic ---

    fn two_sum(self, rhs: Self) -> (Self, Self) {
        // Knuth's two-sum: exact under round-to-nearest, no FMA needed.
        let s = self + rhs;
        if !s.is_finite() {
            // Overflow or a non-finite operand. The exact sum is not representable, so no
            // error term can satisfy `value + error == exact`; reporting zero at least keeps
            // `value + error == value`, so a compensated reduction *saturates* rather than
            // turning to `NaN`. The Knuth sequence would otherwise produce
            // `inf - inf = NaN` here and poison the correction accumulator for good.
            return (s, 0.0);
        }
        let b_virtual = s - self;
        let err = (self - (s - b_virtual)) + (rhs - b_virtual);
        (s, err)
    }

    fn two_product(self, rhs: Self) -> (Self, Self) {
        // With an FMA the discarded remainder is exactly `fma(a, b, -p)`.
        let p = self * rhs;
        if !p.is_finite() {
            // As in `two_sum`: `fma(a, b, -inf)` is `-inf`, which is not a remainder.
            return (p, 0.0);
        }
        let err = f64::mul_add(self, rhs, -p);
        (p, err)
    }

    // --- cancellation-prone elementary forms ---

    fn cos_m1(self) -> Self {
        // cos x - 1 = -2 sin^2(x/2): accurate near zero, where `cos(x) - 1` is not.
        let h = f64::sin(self * 0.5);
        -2.0 * h * h
    }

    fn sinc(self) -> Self {
        if self == 0.0 {
            return 1.0;
        }
        // The Taylor series is better conditioned than sin(x)/x for small |x|, and avoids
        // the 0/0 form entirely.
        if f64::abs(self) < 1e-2 {
            let x2 = self * self;
            return 1.0 - x2 / 6.0 + x2 * x2 / 120.0 - x2 * x2 * x2 / 5040.0;
        }
        f64::sin(self) / self
    }

    fn hypot3(self, y: Self, z: Self) -> Self {
        // Scale by the largest magnitude so no square can overflow or underflow.
        let (a, b, c) = (f64::abs(self), f64::abs(y), f64::abs(z));
        let m = f64::max(f64::max(a, b), c);
        if m == 0.0 {
            return 0.0;
        }
        if !m.is_finite() {
            // An infinite component dominates; a NaN component must still propagate.
            return if a.is_nan() || b.is_nan() || c.is_nan() {
                f64::NAN
            } else {
                f64::INFINITY
            };
        }
        let (a, b, c) = (a / m, b / m, c / m);
        m * f64::sqrt(a * a + b * b + c * c)
    }

    fn ln_rising_factorial(self, n: Self) -> Self {
        crate::kernels::gamma::ln_rising_factorial(self, n)
    }

    // --- elementary completeness ---

    fn log10(self) -> Self {
        f64::log10(self)
    }
    fn exp2(self) -> Self {
        f64::exp2(self)
    }
    fn erfc_inv(self) -> Self {
        crate::kernels::normal::erfc_inv(self)
    }

    // --- the standard normal distribution ---

    fn ndtr(self) -> Self {
        crate::kernels::normal::ndtr(self)
    }
    fn log_ndtr(self) -> Self {
        crate::kernels::normal::log_ndtr(self)
    }
    fn ndtri(self) -> Self {
        crate::kernels::normal::ndtri(self)
    }

    fn gamma_lr_inv(self, shape: Self) -> Self {
        crate::kernels::quantile::gamma_lr_inv(self, shape)
    }
    fn beta_reg_inv(self, a: Self, b: Self) -> Self {
        crate::kernels::quantile::beta_reg_inv(self, a, b)
    }

    // --- floating-point representation ---

    fn frexp(self) -> (Self, Self) {
        crate::kernels::repr::frexp(self)
    }
    fn ldexp(self, exp: Self) -> Self {
        crate::kernels::repr::ldexp(self, exp)
    }
    fn next_after(self, toward: Self) -> Self {
        crate::kernels::repr::next_after(self, toward)
    }
    fn ulp(self) -> Self {
        crate::kernels::repr::ulp(self)
    }
    fn signbit(&self) -> Self::Mask {
        f64::is_sign_negative(*self)
    }
    fn modf(self) -> (Self, Self) {
        (f64::fract(self), f64::trunc(self))
    }

    // --- further analysis ---

    fn exprel(self) -> Self {
        crate::kernels::polygamma::exprel(self)
    }
    fn rgamma(self) -> Self {
        crate::kernels::polygamma::rgamma(self)
    }
    fn polygamma(self, order: u32) -> Self {
        crate::kernels::polygamma::polygamma(order, self)
    }

    fn golden_ratio() -> Self {
        crate::constants::f64::GOLDEN_RATIO
    }
    fn sqrt_pi() -> Self {
        crate::constants::f64::SQRT_PI
    }
    fn sqrt_tau() -> Self {
        crate::constants::f64::SQRT_TAU
    }
    fn apery() -> Self {
        crate::constants::f64::APERY
    }
    fn catalan() -> Self {
        crate::constants::f64::CATALAN
    }
    fn omega() -> Self {
        crate::constants::f64::OMEGA
    }
    // --- classical special functions ---

    fn bessel_j0(self) -> Self {
        crate::kernels::bessel::j0(self)
    }
    fn bessel_j1(self) -> Self {
        crate::kernels::bessel::j1(self)
    }
    fn bessel_y0(self) -> Self {
        crate::kernels::bessel::y0(self)
    }
    fn bessel_y1(self) -> Self {
        crate::kernels::bessel::y1(self)
    }
    fn bessel_k0(self) -> Self {
        crate::kernels::bessel::k0(self)
    }
    fn bessel_k1(self) -> Self {
        crate::kernels::bessel::k1(self)
    }
    fn bessel_k0e(self) -> Self {
        crate::kernels::bessel::k0e(self)
    }
    fn bessel_k1e(self) -> Self {
        crate::kernels::bessel::k1e(self)
    }
    fn sin_int(self) -> Self {
        crate::kernels::integrals::sin_int(self)
    }
    fn cos_int(self) -> Self {
        crate::kernels::integrals::cos_int(self)
    }
    fn exp_int_e1(self) -> Self {
        crate::kernels::integrals::exp_int_e1(self)
    }
    fn exp_int_ei(self) -> Self {
        crate::kernels::integrals::exp_int_ei(self)
    }
    fn lambert_w(self) -> Self {
        crate::kernels::transcendental::lambert_w(self)
    }
    fn zeta(self) -> Self {
        crate::kernels::transcendental::zeta(self)
    }
    fn airy_ai(self) -> Self {
        crate::kernels::airy::ai(self)
    }
    fn airy_ai_prime(self) -> Self {
        crate::kernels::airy::ai_prime(self)
    }
    fn airy_bi(self) -> Self {
        crate::kernels::airy::bi(self)
    }
    fn airy_bi_prime(self) -> Self {
        crate::kernels::airy::bi_prime(self)
    }
    fn airy_ai_e(self) -> Self {
        crate::kernels::airy::aie(self)
    }
    fn airy_ai_e_prime(self) -> Self {
        crate::kernels::airy::aie_prime(self)
    }
    fn airy_bi_e(self) -> Self {
        crate::kernels::airy::bie(self)
    }
    fn airy_bi_e_prime(self) -> Self {
        crate::kernels::airy::bie_prime(self)
    }
}
