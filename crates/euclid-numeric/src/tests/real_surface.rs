//! **Does a formula written once against [`Real`] run on every real carrier?**
//!
//! The analysis tier's portability suite. Each test writes a formula with no mention of a
//! concrete number type — `shifted_quadratic`, `softplus`, an inverse-hyperbolic round trip —
//! and then evaluates it on both `f64` and `f32`, asserting they agree to each format's
//! precision.
//!
//! That is a weaker claim than accuracy and a different one: [`super::accuracy`] asks whether
//! `erf` is close to the true `erf`, while this asks whether *every method the trait names* is
//! actually callable on a generic `T: Real` and returns something sane. A method that compiles
//! only for `f64` — because an implementation reached for a host intrinsic the trait does not
//! expose — fails here and nowhere else.
//!
//! The peer suites cover the other tiers: [`super::masks`] the comparison/selection tier,
//! [`super::discrete`] the semiring and ring tiers, [`super::control_flow`] the loop contract.

use super::*;

/// A formula written once over the real analysis base — analysis plus a scalar constant.
fn shifted_quadratic<T: Real>(x: T) -> T {
    let half = T::half();
    x.clone() * x.clone() + half * x
}

#[test]
fn real_formula_runs_on_f64_and_f32() {
    assert!((shifted_quadratic(2.5_f64) - 7.5).abs() < 1e-12);
    assert!((shifted_quadratic(2.5_f32) - 7.5).abs() < 1e-5);
}

#[test]
fn element_identities_are_defined_directly() {
    assert_eq!(<f64 as Element>::zero(), 0.0);
    assert_eq!(<f64 as Element>::one(), 1.0);
    assert_eq!(<f32 as Element>::zero(), 0.0);
    assert_eq!(<f32 as Element>::one(), 1.0);
}

#[test]
fn lazy_real_constants_lift_scalars() {
    assert_eq!(f64::half(), 0.5);
    assert_eq!(f32::half(), 0.5);
    assert!((f64::pi() - core::f64::consts::PI).abs() < 1e-12);
    assert!((f32::tau() - core::f32::consts::TAU).abs() < 1e-6);
    assert!((f64::half_ln_tau() - 0.5 * core::f64::consts::TAU.ln()).abs() < 1e-12);
}

#[test]
fn euler_constants_lift_on_every_real_tier() {
    // e lifted must agree with exp(1).
    assert!((f64::e() - 1.0_f64.exp()).abs() < 1e-12);
    assert!((f32::e() - 1.0_f32.exp()).abs() < 1e-6);

    // The log-base conversions are mutual inverses.
    assert!((f64::log2_e() * f64::ln_2() - 1.0).abs() < 1e-12);
    assert!((f64::log10_e() * f64::ln_10() - 1.0).abs() < 1e-12);
    assert!((f64::log2_10() * f64::log10_2() - 1.0).abs() < 1e-12);

    // Against `euclid_utils`' independent copy of the digits rather than a literal retyped
    // here. The crate's own catalogue takes `EULER_GAMMA` from `core` as of Rust 1.99, so a
    // literal in this file would now be comparing `core` against itself; the helper crate
    // keeps a deliberately separate copy so this stays a real check. See its doc comment.
    let gamma = euclid_utils::EULER_GAMMA;
    assert!((f64::euler_gamma() - gamma).abs() < 1e-15);
    assert!((f64::from(f32::euler_gamma()) - gamma).abs() < 1e-6);
}

/// `powf` takes its exponent in the same real substrate, so a power law is write-once over
/// [`Real`] with the exponent itself a value of the surface.
fn power_law<T: Real>(base: T, exponent: T) -> T {
    base.powf(exponent)
}

#[test]
fn lazy_real_powf_takes_a_real_exponent() {
    // Exponent lifted as a `Self`, not an `f64`.
    assert!((power_law(2.0_f64, 2.0_f64.scalar_like(10.0)) - 1024.0).abs() < 1e-9);
    assert!((power_law(2.0_f32, 2.0_f32.scalar_like(10.0)) - 1024.0).abs() < 1e-2);
    // The exponent can be computed: x^(ln x / ln x) = x.
    let x = 3.7_f64;
    assert!((x.powf(x.ln() / x.ln()) - x).abs() < 1e-12);
}

/// `powi` takes its exponent from the integer substrate [`Real::Integer`], not a host
/// `i32`, so an integer power law is write-once over the surface.
fn integer_power_law<T: Real>(base: T, exponent: T::Integer) -> T {
    base.powi(exponent)
}

#[test]
fn lazy_real_powi_takes_an_integer_from_the_integer_tier() {
    assert!((integer_power_law(2.0_f64, 10) - 1024.0).abs() < 1e-12);
    assert!((integer_power_law(2.0_f32, 10) - 1024.0).abs() < 1e-3);
    // Exact repeated-multiplication semantics `powf` cannot give: (-2)^3 = -8, 2^-2 = 0.25.
    assert_eq!((-2.0_f64).powi(3), -8.0);
    assert!((2.0_f64.powi(-2) - 0.25).abs() < 1e-12);

    // `f64` uses the full `i64` exponent surface; it must not silently narrow to `i32`.
    let beyond_i32 = i64::from(i32::MAX) + 1;
    assert_eq!(<f64 as Real>::powi(2.0, beyond_i32), f64::INFINITY);
    assert_eq!(<f64 as Real>::powi(0.5, beyond_i32), 0.0);
}

fn square_plus_self<T: Real>(x: T) -> T {
    x.square() + x
}

#[test]
fn lazy_real_square_is_x_times_x() {
    assert_eq!((-3.0_f64).square(), 9.0);
    assert_eq!((-3.0_f32).square(), 9.0);
    assert_eq!(square_plus_self(2.0_f64), 6.0);
    assert_eq!(square_plus_self(2.0_f32), 6.0);
}

#[test]
fn scalar_stable_operations_handle_cancellation_and_tails() {
    let tiny = 1e-16_f64;
    assert_eq!((1.0 + tiny).ln(), 0.0);
    assert!((<f64 as Real>::ln_1p(&tiny) - tiny).abs() < 1e-30);
    assert_eq!(tiny.exp() - 1.0, 0.0);
    assert!((<f64 as Real>::exp_m1(&tiny) - tiny).abs() < 1e-30);
    assert!((tiny.ln_1m() + tiny).abs() < 1e-30);

    let small = 1e-16_f64;
    assert!(small.ln_exp_m1().is_finite());
    assert!((small.ln_exp_m1() - small.ln()).abs() < 1e-12);

    let big = 1000.0_f64;
    assert_eq!(big.sigmoid(), 1.0);
    assert_eq!((-big).sigmoid(), 0.0);
    assert!((big.ln_1m_sigmoid() + big).abs() < 1e-12);
    assert!(((-big).ln_sigmoid() + big).abs() < 1e-12);

    let p = 1.0_f64 - 1e-12;
    assert!((p.logit().sigmoid() - p).abs() < 1e-15);

    assert_eq!(0.0_f64.xlog1py(-1.0), 0.0);
    assert!((2.0_f64.xlog1py(tiny) - 2.0 * tiny).abs() < 1e-30);
    assert!((2.0_f64.xlog1my(tiny) + 2.0 * tiny).abs() < 1e-30);

    let tiny32 = 1e-8_f32;
    assert_eq!((1.0_f32 + tiny32).ln(), 0.0);
    assert!((<f32 as Real>::ln_1p(&tiny32) - tiny32).abs() < 1e-12);
    assert!((<f32 as Real>::exp_m1(&tiny32) - tiny32).abs() < 1e-12);
}

#[test]
fn softplus_is_stable_at_extremes() {
    assert!((f64::ln_1p_exp(&0.0) - core::f64::consts::LN_2).abs() < 1e-12);
    let big = 50.0_f64;
    assert!((big.ln_1p_exp() - big).abs() < 1e-10);
    assert!((0.0_f64.ln_add_exp(0.0) - core::f64::consts::LN_2).abs() < 1e-12);
    assert!((2.0_f64.ln_sub_exp(1.0).exp() - (2.0_f64.exp() - 1.0_f64.exp())).abs() < 1e-12);
    assert_eq!(
        f64::NEG_INFINITY.ln_add_exp(f64::NEG_INFINITY),
        f64::NEG_INFINITY
    );
    assert_eq!(f64::INFINITY.ln_add_exp(1.0), f64::INFINITY);
    assert_eq!(2.0_f64.ln_sub_exp(2.0), f64::NEG_INFINITY);
    assert!(f64::INFINITY.ln_sub_exp(f64::INFINITY).is_nan());
}

/// A formula on [`Real`] using `atan2`, trig, and the `clamp` selector — once a
/// concrete-only API, now compiling for every real carrier.
fn wrapped_angle_unit<T: Real>(x: T, y: T) -> T {
    let angle = y.atan2(x);
    let pythagorean = angle.clone().sin() * angle.clone().sin() + angle.clone().cos() * angle.cos();
    pythagorean.clamp(T::zero(), T::one())
}

#[test]
fn lazy_real_carries_trig_and_select_on_f64_and_f32() {
    // sin²+cos² = 1, clamped to [0, 1], through the Real bound alone.
    assert!((wrapped_angle_unit(0.7_f64, 1.3_f64) - 1.0).abs() < 1e-12);
    assert!((wrapped_angle_unit(0.7_f32, 1.3_f32) - 1.0).abs() < 1e-6);
    let (s, c) = <f64 as Real>::sin_cos(0.7);
    assert!((s * s + c * c - 1.0).abs() < 1e-12);
    let (s, c) = <f32 as Real>::sin_cos(0.7);
    assert!((s * s + c * c - 1.0).abs() < 1e-6);
    assert_eq!(Real::max(Real::min(3.0_f64, 5.0), 1.0), 3.0);
    assert_eq!(Real::clamp(2.0_f32, 0.0, 1.0), 1.0);
}

/// Inverse hyperbolics over [`Real`] invert their forward partner.
fn asinh_then_sinh<T: Real>(x: T) -> T {
    x.asinh().sinh()
}

/// Algebraic spelling of `asinh`, used as a reference at moderate inputs.
fn asinh_via_ln<T: Real>(x: T) -> T {
    (x.clone() + (x.clone() * x + T::one()).sqrt()).ln()
}

#[test]
fn lazy_real_inverse_hyperbolics_invert_and_match_identity() {
    // sinh∘asinh = id, tanh∘atanh = id, acosh∘cosh = id.
    assert!((asinh_then_sinh(1.3_f64) - 1.3).abs() < 1e-12);
    assert!((asinh_then_sinh(1.3_f32) - 1.3).abs() < 1e-5);
    assert!((0.4_f64.atanh().tanh() - 0.4).abs() < 1e-12);
    assert!((2.0_f64.cosh().acosh() - 2.0).abs() < 1e-12);

    // The required implementation agrees with the algebraic identity at moderate inputs.
    assert!((1.3_f64.asinh() - asinh_via_ln(1.3_f64)).abs() < 1e-12);
    assert!((1.3_f32.asinh() - asinh_via_ln(1.3_f32)).abs() < 1e-5);
}

/// The Normal CDF `Φ(z) = ½·erfc(−z/√2)` over [`Real`].
fn normal_cdf<T: Real>(z: T) -> T {
    T::half() * (-z / T::sqrt_2()).erfc()
}

#[test]
fn lazy_real_special_functions_run_on_f64_and_f32() {
    assert!((normal_cdf(0.0_f64) - 0.5).abs() < 1e-15);
    assert!((normal_cdf(0.0_f32) - 0.5).abs() < 1e-6);

    // Spot-check the defining identities through the trait.
    assert!((Real::ln_gamma(5.0_f64) - 24.0_f64.ln()).abs() < 1e-9);
    assert!((Real::digamma(1.0_f64) + f64::euler_gamma()).abs() < 1e-9);
    assert!((Real::ln_beta(2.0_f64, 3.0) - (1.0_f64 / 12.0).ln()).abs() < 1e-10);
    assert!((Real::beta(2.0_f64, 3.0) - 1.0 / 12.0).abs() < 1e-10);
    // P(a, x) + Q(a, x) = 1.
    let p = Real::gamma_lr(3.0_f64, 2.5);
    let q = Real::gamma_ur(3.0_f64, 2.5);
    assert!((p + q - 1.0).abs() < 1e-10);
    // I_x(1, 1) = x.
    assert!((Real::beta_reg(0.37_f64, 1.0, 1.0) - 0.37).abs() < 1e-12);
    assert!((Real::erf(1.0_f32) - 0.842_700_8).abs() < 1e-6);

    // erfcx(x)·e^{-x²} = erfc(x); ln(1−e^x) complements ln(1+e^x); scaled Bessel recovers
    // ln I₀ and the I₁/I₀ ratio.
    let x = 0.6_f64;
    assert!((x.erfcx() * (-x * x).exp() - Real::erfc(x)).abs() < 1e-12);
    assert!((Real::ln_1m_exp(&(-0.5_f64)).exp() - (1.0 - (-0.5_f64).exp())).abs() < 1e-12);
    let k = 2.5_f64;
    assert!((k.bessel_i0e().ln() + k - crate::kernels::bessel::ln_i0(k)).abs() < 1e-9);
    assert!((k.ln_bessel_i0() - crate::kernels::bessel::ln_i0(k)).abs() < 1e-9);
    assert!((k.bessel_i1_over_i0() - crate::kernels::bessel::i1_over_i0(k)).abs() < 1e-9);
    assert!((k.bessel_i1e() / k.bessel_i0e() - crate::kernels::bessel::i1_over_i0(k)).abs() < 1e-9);
    // Native f32 paths stay close to the f64 reference.
    assert!((0.5_f32.erfcx() - 0.5_f64.erfcx() as f32).abs() < 1e-6);
    assert!((Real::beta(2.0_f32, 3.0) - 1.0 / 12.0).abs() < 1e-5);
}
