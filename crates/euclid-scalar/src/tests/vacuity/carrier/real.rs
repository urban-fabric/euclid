//! The `Real` implementation, where every method is perturbed by index.
//!
//! A child module so the file stays inside the crate's 400-line budget; a single trait impl has
//! no sub-topics of its own, so what makes the split natural is that everything in [`super`] is
//! either structure or a faithful forwarding, while this is the surface under test. Each method
//! calls [`super::perturb`] with its own identifier, so `Faulty<N>` damages exactly method `N`
//! and the conformance suite is asked whether it notices.

use super::*;

impl<const WHICH: u32> Real for Faulty<WHICH> {
    type Integer = i64;
    fn scalar(x: f64) -> Self {
        Self(perturb::<WHICH>(15, <f64 as Real>::scalar(x)))
    }
    fn scalar_like(&self, x: f64) -> Self {
        Self(perturb::<WHICH>(16, <f64 as Real>::scalar_like(&self.0, x)))
    }
    fn two() -> Self {
        Self(perturb::<WHICH>(17, <f64 as Real>::two()))
    }
    fn half() -> Self {
        Self(perturb::<WHICH>(18, <f64 as Real>::half()))
    }
    fn pi() -> Self {
        Self(perturb::<WHICH>(19, <f64 as Real>::pi()))
    }
    fn tau() -> Self {
        Self(perturb::<WHICH>(20, <f64 as Real>::tau()))
    }
    fn frac_pi_2() -> Self {
        Self(perturb::<WHICH>(21, <f64 as Real>::frac_pi_2()))
    }
    fn frac_pi_4() -> Self {
        Self(perturb::<WHICH>(22, <f64 as Real>::frac_pi_4()))
    }
    fn sqrt_2() -> Self {
        Self(perturb::<WHICH>(23, <f64 as Real>::sqrt_2()))
    }
    fn frac_1_sqrt_2() -> Self {
        Self(perturb::<WHICH>(24, <f64 as Real>::frac_1_sqrt_2()))
    }
    fn ln_2() -> Self {
        Self(perturb::<WHICH>(25, <f64 as Real>::ln_2()))
    }
    fn ln_pi() -> Self {
        Self(perturb::<WHICH>(26, <f64 as Real>::ln_pi()))
    }
    fn ln_tau() -> Self {
        Self(perturb::<WHICH>(27, <f64 as Real>::ln_tau()))
    }
    fn half_ln_tau() -> Self {
        Self(perturb::<WHICH>(28, <f64 as Real>::half_ln_tau()))
    }
    fn e() -> Self {
        Self(perturb::<WHICH>(29, <f64 as Real>::e()))
    }
    fn log2_e() -> Self {
        Self(perturb::<WHICH>(30, <f64 as Real>::log2_e()))
    }
    fn log10_e() -> Self {
        Self(perturb::<WHICH>(31, <f64 as Real>::log10_e()))
    }
    fn ln_10() -> Self {
        Self(perturb::<WHICH>(32, <f64 as Real>::ln_10()))
    }
    fn log2_10() -> Self {
        Self(perturb::<WHICH>(33, <f64 as Real>::log2_10()))
    }
    fn log10_2() -> Self {
        Self(perturb::<WHICH>(34, <f64 as Real>::log10_2()))
    }
    fn euler_gamma() -> Self {
        Self(perturb::<WHICH>(35, <f64 as Real>::euler_gamma()))
    }
    fn golden_ratio() -> Self {
        Self(perturb::<WHICH>(36, <f64 as Real>::golden_ratio()))
    }
    fn sqrt_pi() -> Self {
        Self(perturb::<WHICH>(37, <f64 as Real>::sqrt_pi()))
    }
    fn sqrt_tau() -> Self {
        Self(perturb::<WHICH>(38, <f64 as Real>::sqrt_tau()))
    }
    fn apery() -> Self {
        Self(perturb::<WHICH>(39, <f64 as Real>::apery()))
    }
    fn catalan() -> Self {
        Self(perturb::<WHICH>(40, <f64 as Real>::catalan()))
    }
    fn omega() -> Self {
        Self(perturb::<WHICH>(41, <f64 as Real>::omega()))
    }
    fn infinity() -> Self {
        Self(perturb::<WHICH>(42, <f64 as Real>::infinity()))
    }
    fn neg_infinity() -> Self {
        Self(perturb::<WHICH>(43, <f64 as Real>::neg_infinity()))
    }
    fn nan() -> Self {
        Self(perturb::<WHICH>(44, <f64 as Real>::nan()))
    }
    fn is_nan(&self) -> Self::Mask {
        perturb_mask::<WHICH>(45, <f64 as Real>::is_nan(&self.0))
    }
    fn is_infinite(&self) -> Self::Mask {
        perturb_mask::<WHICH>(46, <f64 as Real>::is_infinite(&self.0))
    }
    fn is_finite(&self) -> Self::Mask {
        perturb_mask::<WHICH>(47, <f64 as Real>::is_finite(&self.0))
    }
    fn exp(&self) -> Self {
        Self(perturb::<WHICH>(48, <f64 as Real>::exp(&self.0)))
    }
    fn ln(&self) -> Self {
        Self(perturb::<WHICH>(49, <f64 as Real>::ln(&self.0)))
    }
    fn ln_1p(&self) -> Self {
        Self(perturb::<WHICH>(50, <f64 as Real>::ln_1p(&self.0)))
    }
    fn exp_m1(&self) -> Self {
        Self(perturb::<WHICH>(51, <f64 as Real>::exp_m1(&self.0)))
    }
    fn sqrt(&self) -> Self {
        Self(perturb::<WHICH>(52, <f64 as Real>::sqrt(&self.0)))
    }
    fn abs(&self) -> Self {
        Self(perturb::<WHICH>(53, <f64 as Real>::abs(&self.0)))
    }
    fn square(&self) -> Self {
        Self(perturb::<WHICH>(54, <f64 as Real>::square(&self.0)))
    }
    fn powf(&self, p: Self) -> Self {
        Self(perturb::<WHICH>(55, <f64 as Real>::powf(&self.0, p.0)))
    }
    fn ln_1p_exp(&self) -> Self {
        Self(perturb::<WHICH>(56, <f64 as Real>::ln_1p_exp(&self.0)))
    }
    fn sin(self) -> Self {
        Self(perturb::<WHICH>(57, <f64 as Real>::sin(self.0)))
    }
    fn cos(self) -> Self {
        Self(perturb::<WHICH>(58, <f64 as Real>::cos(self.0)))
    }
    fn tan(self) -> Self {
        Self(perturb::<WHICH>(59, <f64 as Real>::tan(self.0)))
    }
    fn asin(self) -> Self {
        Self(perturb::<WHICH>(60, <f64 as Real>::asin(self.0)))
    }
    fn acos(self) -> Self {
        Self(perturb::<WHICH>(61, <f64 as Real>::acos(self.0)))
    }
    fn atan(self) -> Self {
        Self(perturb::<WHICH>(62, <f64 as Real>::atan(self.0)))
    }
    fn atan2(self, x: Self) -> Self {
        Self(perturb::<WHICH>(63, <f64 as Real>::atan2(self.0, x.0)))
    }
    fn sinh(self) -> Self {
        Self(perturb::<WHICH>(64, <f64 as Real>::sinh(self.0)))
    }
    fn cosh(self) -> Self {
        Self(perturb::<WHICH>(65, <f64 as Real>::cosh(self.0)))
    }
    fn tanh(self) -> Self {
        Self(perturb::<WHICH>(66, <f64 as Real>::tanh(self.0)))
    }
    fn asinh(self) -> Self {
        Self(perturb::<WHICH>(67, <f64 as Real>::asinh(self.0)))
    }
    fn acosh(self) -> Self {
        Self(perturb::<WHICH>(68, <f64 as Real>::acosh(self.0)))
    }
    fn atanh(self) -> Self {
        Self(perturb::<WHICH>(69, <f64 as Real>::atanh(self.0)))
    }
    fn sin_cos(self) -> (Self, Self) {
        let (first, second) = <f64 as Real>::sin_cos(self.0);
        (Self(perturb::<WHICH>(70, first)), Self(second))
    }
    fn sin_cos_degrees(self) -> (Self, Self) {
        let (first, second) = <f64 as Real>::sin_cos_degrees(self.0);
        (Self(perturb::<WHICH>(71, first)), Self(second))
    }
    fn polyval(self, coefficients: &[Self]) -> Self {
        let mut total = <Self as Element>::zero();
        for coefficient in coefficients.iter().rev() {
            total = total * self + *coefficient;
        }
        total
    }
    fn hypot(self, other: Self) -> Self {
        Self(perturb::<WHICH>(72, <f64 as Real>::hypot(self.0, other.0)))
    }
    fn cbrt(self) -> Self {
        Self(perturb::<WHICH>(73, <f64 as Real>::cbrt(self.0)))
    }
    fn powi(self, n: Self::Integer) -> Self {
        Self(perturb::<WHICH>(74, <f64 as Real>::powi(self.0, n)))
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(perturb::<WHICH>(
            75,
            <f64 as Real>::mul_add(self.0, a.0, b.0),
        ))
    }
    fn recip(self) -> Self {
        Self(perturb::<WHICH>(76, <f64 as Real>::recip(self.0)))
    }
    fn log2(self) -> Self {
        Self(perturb::<WHICH>(77, <f64 as Real>::log2(self.0)))
    }
    fn floor(self) -> Self {
        Self(perturb::<WHICH>(78, <f64 as Real>::floor(self.0)))
    }
    fn ceil(self) -> Self {
        Self(perturb::<WHICH>(79, <f64 as Real>::ceil(self.0)))
    }
    fn round(self) -> Self {
        Self(perturb::<WHICH>(80, <f64 as Real>::round(self.0)))
    }
    fn trunc(self) -> Self {
        Self(perturb::<WHICH>(81, <f64 as Real>::trunc(self.0)))
    }
    fn fract(self) -> Self {
        Self(perturb::<WHICH>(82, <f64 as Real>::fract(self.0)))
    }
    fn signum(self) -> Self {
        Self(perturb::<WHICH>(83, <f64 as Real>::signum(self.0)))
    }
    fn copysign(self, sign: Self) -> Self {
        Self(perturb::<WHICH>(
            84,
            <f64 as Real>::copysign(self.0, sign.0),
        ))
    }
    fn rem_euclid(self, m: Self) -> Self {
        Self(perturb::<WHICH>(85, <f64 as Real>::rem_euclid(self.0, m.0)))
    }
    fn to_radians(self) -> Self {
        Self(perturb::<WHICH>(86, <f64 as Real>::to_radians(self.0)))
    }
    fn to_degrees(self) -> Self {
        Self(perturb::<WHICH>(87, <f64 as Real>::to_degrees(self.0)))
    }
    fn min(self, other: Self) -> Self {
        Self(perturb::<WHICH>(88, <f64 as Real>::min(self.0, other.0)))
    }
    fn max(self, other: Self) -> Self {
        Self(perturb::<WHICH>(89, <f64 as Real>::max(self.0, other.0)))
    }
    fn clamp(self, lo: Self, hi: Self) -> Self {
        Self(perturb::<WHICH>(
            90,
            <f64 as Real>::clamp(self.0, lo.0, hi.0),
        ))
    }
    fn erf(self) -> Self {
        Self(perturb::<WHICH>(91, <f64 as Real>::erf(self.0)))
    }
    fn erfc(self) -> Self {
        Self(perturb::<WHICH>(92, <f64 as Real>::erfc(self.0)))
    }
    fn erf_inv(self) -> Self {
        Self(perturb::<WHICH>(93, <f64 as Real>::erf_inv(self.0)))
    }
    fn ln_gamma(self) -> Self {
        Self(perturb::<WHICH>(94, <f64 as Real>::ln_gamma(self.0)))
    }
    fn digamma(self) -> Self {
        Self(perturb::<WHICH>(95, <f64 as Real>::digamma(self.0)))
    }
    fn trigamma(self) -> Self {
        Self(perturb::<WHICH>(96, <f64 as Real>::trigamma(self.0)))
    }
    fn gamma_lr(self, shape: Self) -> Self {
        Self(perturb::<WHICH>(
            97,
            <f64 as Real>::gamma_lr(self.0, shape.0),
        ))
    }
    fn gamma_ur(self, shape: Self) -> Self {
        Self(perturb::<WHICH>(
            98,
            <f64 as Real>::gamma_ur(self.0, shape.0),
        ))
    }
    fn beta_reg(self, a: Self, b: Self) -> Self {
        Self(perturb::<WHICH>(
            99,
            <f64 as Real>::beta_reg(self.0, a.0, b.0),
        ))
    }
    fn ln_1m_exp(&self) -> Self {
        Self(perturb::<WHICH>(100, <f64 as Real>::ln_1m_exp(&self.0)))
    }
    fn erfcx(self) -> Self {
        Self(perturb::<WHICH>(101, <f64 as Real>::erfcx(self.0)))
    }
    fn bessel_i0(self) -> Self {
        Self(perturb::<WHICH>(102, <f64 as Real>::bessel_i0(self.0)))
    }
    fn bessel_i1(self) -> Self {
        Self(perturb::<WHICH>(103, <f64 as Real>::bessel_i1(self.0)))
    }
    fn bessel_i0e(self) -> Self {
        Self(perturb::<WHICH>(104, <f64 as Real>::bessel_i0e(self.0)))
    }
    fn bessel_i1e(self) -> Self {
        Self(perturb::<WHICH>(105, <f64 as Real>::bessel_i1e(self.0)))
    }
    fn xlogy(self, y: Self) -> Self {
        Self(perturb::<WHICH>(106, <f64 as Real>::xlogy(self.0, y.0)))
    }
    fn xlog1py(self, y: Self) -> Self {
        Self(perturb::<WHICH>(107, <f64 as Real>::xlog1py(self.0, y.0)))
    }
    fn ln_1m(&self) -> Self {
        Self(perturb::<WHICH>(108, <f64 as Real>::ln_1m(&self.0)))
    }
    fn ln_exp_m1(&self) -> Self {
        Self(perturb::<WHICH>(109, <f64 as Real>::ln_exp_m1(&self.0)))
    }
    fn sigmoid(&self) -> Self {
        Self(perturb::<WHICH>(110, <f64 as Real>::sigmoid(&self.0)))
    }
    fn ln_sigmoid(&self) -> Self {
        Self(perturb::<WHICH>(111, <f64 as Real>::ln_sigmoid(&self.0)))
    }
    fn ln_1m_sigmoid(&self) -> Self {
        Self(perturb::<WHICH>(112, <f64 as Real>::ln_1m_sigmoid(&self.0)))
    }
    fn logit(&self) -> Self {
        Self(perturb::<WHICH>(113, <f64 as Real>::logit(&self.0)))
    }
    fn xlog1my(self, y: Self) -> Self {
        Self(perturb::<WHICH>(114, <f64 as Real>::xlog1my(self.0, y.0)))
    }
    fn ln_add_exp(self, other: Self) -> Self {
        Self(perturb::<WHICH>(
            115,
            <f64 as Real>::ln_add_exp(self.0, other.0),
        ))
    }
    fn ln_sub_exp(self, other: Self) -> Self {
        Self(perturb::<WHICH>(
            116,
            <f64 as Real>::ln_sub_exp(self.0, other.0),
        ))
    }
    fn gamma(self) -> Self {
        Self(perturb::<WHICH>(117, <f64 as Real>::gamma(self.0)))
    }
    fn ln_beta(self, b: Self) -> Self {
        Self(perturb::<WHICH>(118, <f64 as Real>::ln_beta(self.0, b.0)))
    }
    fn beta(self, b: Self) -> Self {
        Self(perturb::<WHICH>(119, <f64 as Real>::beta(self.0, b.0)))
    }
    fn ln_bessel_i0(self) -> Self {
        Self(perturb::<WHICH>(120, <f64 as Real>::ln_bessel_i0(self.0)))
    }
    fn bessel_i1_over_i0(self) -> Self {
        Self(perturb::<WHICH>(
            121,
            <f64 as Real>::bessel_i1_over_i0(self.0),
        ))
    }
    fn two_sum(self, rhs: Self) -> (Self, Self) {
        let (first, second) = <f64 as Real>::two_sum(self.0, rhs.0);
        (Self(perturb::<WHICH>(122, first)), Self(second))
    }
    fn two_product(self, rhs: Self) -> (Self, Self) {
        let (first, second) = <f64 as Real>::two_product(self.0, rhs.0);
        (Self(perturb::<WHICH>(123, first)), Self(second))
    }
    fn cos_m1(self) -> Self {
        Self(perturb::<WHICH>(124, <f64 as Real>::cos_m1(self.0)))
    }
    fn sinc(self) -> Self {
        Self(perturb::<WHICH>(125, <f64 as Real>::sinc(self.0)))
    }
    fn hypot3(self, y: Self, z: Self) -> Self {
        Self(perturb::<WHICH>(
            126,
            <f64 as Real>::hypot3(self.0, y.0, z.0),
        ))
    }
    fn ln_rising_factorial(self, n: Self) -> Self {
        Self(perturb::<WHICH>(
            127,
            <f64 as Real>::ln_rising_factorial(self.0, n.0),
        ))
    }
    fn log10(self) -> Self {
        Self(perturb::<WHICH>(128, <f64 as Real>::log10(self.0)))
    }
    fn exp2(self) -> Self {
        Self(perturb::<WHICH>(129, <f64 as Real>::exp2(self.0)))
    }
    fn erfc_inv(self) -> Self {
        Self(perturb::<WHICH>(130, <f64 as Real>::erfc_inv(self.0)))
    }
    fn ndtr(self) -> Self {
        Self(perturb::<WHICH>(131, <f64 as Real>::ndtr(self.0)))
    }
    fn log_ndtr(self) -> Self {
        Self(perturb::<WHICH>(132, <f64 as Real>::log_ndtr(self.0)))
    }
    fn gamma_lr_inv(self, shape: Self) -> Self {
        Self(perturb::<WHICH>(
            133,
            <f64 as Real>::gamma_lr_inv(self.0, shape.0),
        ))
    }
    fn beta_reg_inv(self, a: Self, b: Self) -> Self {
        Self(perturb::<WHICH>(
            134,
            <f64 as Real>::beta_reg_inv(self.0, a.0, b.0),
        ))
    }
    fn frexp(self) -> (Self, Self) {
        let (first, second) = <f64 as Real>::frexp(self.0);
        (Self(perturb::<WHICH>(135, first)), Self(second))
    }
    fn ldexp(self, exp: Self) -> Self {
        Self(perturb::<WHICH>(136, <f64 as Real>::ldexp(self.0, exp.0)))
    }
    fn next_after(self, toward: Self) -> Self {
        Self(perturb::<WHICH>(
            137,
            <f64 as Real>::next_after(self.0, toward.0),
        ))
    }
    fn ulp(self) -> Self {
        Self(perturb::<WHICH>(138, <f64 as Real>::ulp(self.0)))
    }
    fn signbit(&self) -> Self::Mask {
        perturb_mask::<WHICH>(139, <f64 as Real>::signbit(&self.0))
    }
    fn modf(self) -> (Self, Self) {
        let (first, second) = <f64 as Real>::modf(self.0);
        (Self(perturb::<WHICH>(140, first)), Self(second))
    }
    fn exprel(self) -> Self {
        Self(perturb::<WHICH>(141, <f64 as Real>::exprel(self.0)))
    }
    fn rgamma(self) -> Self {
        Self(perturb::<WHICH>(142, <f64 as Real>::rgamma(self.0)))
    }
    fn polygamma(self, order: u32) -> Self {
        Self(perturb::<WHICH>(
            143,
            <f64 as Real>::polygamma(self.0, order),
        ))
    }
    fn ndtri(self) -> Self {
        Self(perturb::<WHICH>(144, <f64 as Real>::ndtri(self.0)))
    }
    fn bessel_j0(self) -> Self {
        Self(perturb::<WHICH>(145, <f64 as Real>::bessel_j0(self.0)))
    }
    fn bessel_j1(self) -> Self {
        Self(perturb::<WHICH>(146, <f64 as Real>::bessel_j1(self.0)))
    }
    fn bessel_y0(self) -> Self {
        Self(perturb::<WHICH>(147, <f64 as Real>::bessel_y0(self.0)))
    }
    fn bessel_y1(self) -> Self {
        Self(perturb::<WHICH>(148, <f64 as Real>::bessel_y1(self.0)))
    }
    fn bessel_k0(self) -> Self {
        Self(perturb::<WHICH>(149, <f64 as Real>::bessel_k0(self.0)))
    }
    fn bessel_k1(self) -> Self {
        Self(perturb::<WHICH>(150, <f64 as Real>::bessel_k1(self.0)))
    }
    fn bessel_k0e(self) -> Self {
        Self(perturb::<WHICH>(151, <f64 as Real>::bessel_k0e(self.0)))
    }
    fn bessel_k1e(self) -> Self {
        Self(perturb::<WHICH>(152, <f64 as Real>::bessel_k1e(self.0)))
    }
    fn sin_int(self) -> Self {
        Self(perturb::<WHICH>(153, <f64 as Real>::sin_int(self.0)))
    }
    fn cos_int(self) -> Self {
        Self(perturb::<WHICH>(154, <f64 as Real>::cos_int(self.0)))
    }
    fn exp_int_e1(self) -> Self {
        Self(perturb::<WHICH>(155, <f64 as Real>::exp_int_e1(self.0)))
    }
    fn exp_int_ei(self) -> Self {
        Self(perturb::<WHICH>(156, <f64 as Real>::exp_int_ei(self.0)))
    }
    fn lambert_w(self) -> Self {
        Self(perturb::<WHICH>(157, <f64 as Real>::lambert_w(self.0)))
    }
    fn zeta(self) -> Self {
        Self(perturb::<WHICH>(158, <f64 as Real>::zeta(self.0)))
    }
    fn airy_ai(self) -> Self {
        Self(perturb::<WHICH>(159, <f64 as Real>::airy_ai(self.0)))
    }
    fn airy_ai_prime(self) -> Self {
        Self(perturb::<WHICH>(160, <f64 as Real>::airy_ai_prime(self.0)))
    }
    fn airy_bi(self) -> Self {
        Self(perturb::<WHICH>(161, <f64 as Real>::airy_bi(self.0)))
    }
    fn airy_bi_prime(self) -> Self {
        Self(perturb::<WHICH>(162, <f64 as Real>::airy_bi_prime(self.0)))
    }
    fn airy_ai_e(self) -> Self {
        Self(perturb::<WHICH>(163, <f64 as Real>::airy_ai_e(self.0)))
    }
    fn airy_ai_e_prime(self) -> Self {
        Self(perturb::<WHICH>(
            164,
            <f64 as Real>::airy_ai_e_prime(self.0),
        ))
    }
    fn airy_bi_e(self) -> Self {
        Self(perturb::<WHICH>(165, <f64 as Real>::airy_bi_e(self.0)))
    }
    fn airy_bi_e_prime(self) -> Self {
        Self(perturb::<WHICH>(
            166,
            <f64 as Real>::airy_bi_e_prime(self.0),
        ))
    }
}
