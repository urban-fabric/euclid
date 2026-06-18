use super::*;

/// A formula written once over the lazy analysis base — pure analysis plus a
/// scalar constant, with no vector/tensor construction.
fn shifted_quadratic<T: LazyReal>(x: T) -> T {
    let half = T::half();
    x.clone() * x.clone() + half * x
}

#[test]
fn real_formula_runs_on_f64_and_f32() {
    assert!((shifted_quadratic(2.5_f64) - 7.5).abs() < 1e-12);
    assert!((shifted_quadratic(2.5_f32) - 7.5).abs() < 1e-5);
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
    // Euler's number names the base of the analysis surface directly: lifting
    // it must agree with recovering it as `exp(1)`.
    assert!((f64::e() - 1.0_f64.exp()).abs() < 1e-12);
    assert!((f32::e() - 1.0_f32.exp()).abs() < 1e-6);
    assert_eq!(f64::e(), <f64 as Real>::E);
    assert_eq!(f32::e(), <f32 as Real>::E);

    // The log-base conversions are mutual inverses through `e` / `10`.
    assert!((f64::log2_e() * f64::ln_2() - 1.0).abs() < 1e-12);
    assert!((f64::log10_e() * f64::ln_10() - 1.0).abs() < 1e-12);
    assert!((f64::log2_10() * f64::log10_2() - 1.0).abs() < 1e-12);

    // Euler–Mascheroni γ, to its leading digits.
    assert!((f64::euler_gamma() - 0.577_215_664_901_532_9).abs() < 1e-15);
    assert!((f32::euler_gamma() - 0.577_215_7).abs() < 1e-6);
}

#[test]
fn softplus_is_stable_at_extremes() {
    assert!((f64::ln_1p_exp(&0.0) - core::f64::consts::LN_2).abs() < 1e-12);
    let big = 50.0_f64;
    assert!((big.ln_1p_exp() - big).abs() < 1e-10);
}

/// A point on the unit circle — written once over the fused [`Real`] trait,
/// which carries algebra (operators), analysis (`LazyReal`), and the real API
/// (`sin_cos`/`PI`) through a single bound.
fn unit_circle_point<T: Real>(theta: T) -> (T, T) {
    let (s, c) = theta.sin_cos();
    (c, s)
}

#[test]
fn fused_real_runs_on_f64_and_f32() {
    // The fused trait at f64: trig + constants.
    let (x, y) = unit_circle_point(f64::PI);
    assert!((x + 1.0).abs() < 1e-12 && y.abs() < 1e-12);
    // ...and at f32, mixing analysis (hypot) with the algebra identity ONE.
    assert!((3.0_f32.hypot(4.0) - 5.0).abs() < 1e-5);
    assert_eq!(<f32 as crate::algebra::HasOne>::ONE, 1.0);
    // A concrete `Real` also satisfies a write-once `LazyReal` formula.
    assert!((shifted_quadratic(2.0_f64) - 5.0).abs() < 1e-12);
}

/// A formula bounded purely on [`LazyReal`] that uses what used to be the
/// concrete-only API — `atan2`, trig, and the `clamp` selector. It now compiles
/// for every lazy backend, not only the concrete reals.
fn wrapped_angle_unit<T: LazyReal>(x: T, y: T) -> T {
    let angle = y.atan2(x);
    let pythagorean = angle.clone().sin() * angle.clone().sin() + angle.clone().cos() * angle.cos();
    pythagorean.clamp(T::zero(), T::one())
}

#[test]
fn lazy_real_carries_trig_and_select_on_f64_and_f32() {
    // sin²+cos² = 1, clamped to [0, 1] — trig + clamp through the LazyReal bound
    // alone, with no `Real` in sight.
    assert!((wrapped_angle_unit(0.7_f64, 1.3_f64) - 1.0).abs() < 1e-12);
    assert!((wrapped_angle_unit(0.7_f32, 1.3_f32) - 1.0).abs() < 1e-6);
    // The `min`/`max`/`clamp` selectors are on the lazy tier too.
    assert_eq!(LazyReal::max(LazyReal::min(3.0_f64, 5.0), 1.0), 3.0);
    assert_eq!(LazyReal::clamp(2.0_f32, 0.0, 1.0), 1.0);
}

/// The Normal CDF `Φ(z) = ½·erfc(−z/√2)`, written once over [`LazyReal`]: it
/// needs `erfc`, which now lives on the lazy surface, so the same source lowers
/// to a tensor backend's `Erfc` op.
fn normal_cdf<T: LazyReal>(z: T) -> T {
    T::half() * (-z / T::sqrt_2()).erfc()
}

#[test]
fn lazy_real_special_functions_run_on_f64_and_f32() {
    // Φ(0) = ½ through the write-once CDF, on both concrete reals.
    assert!((normal_cdf(0.0_f64) - 0.5).abs() < 1e-15);
    assert!((normal_cdf(0.0_f32) - 0.5).abs() < 1e-6);

    // The methods delegate to `crate::special`; spot-check the defining
    // identities directly through the trait.
    assert!((LazyReal::ln_gamma(5.0_f64) - 24.0_f64.ln()).abs() < 1e-9);
    assert!((LazyReal::digamma(1.0_f64) + f64::euler_gamma()).abs() < 1e-9);
    // P(a, x) + Q(a, x) = 1, with `self` the point `x`.
    let p = LazyReal::gamma_lr(3.0_f64, 2.5); // P(2.5, 3.0)
    let q = LazyReal::gamma_ur(3.0_f64, 2.5); // Q(2.5, 3.0)
    assert!((p + q - 1.0).abs() < 1e-10);
    // I_x(1, 1) = x.
    assert!((LazyReal::beta_reg(0.37_f64, 1.0, 1.0) - 0.37).abs() < 1e-12);
    assert!((LazyReal::erf(1.0_f32) - 0.842_700_8).abs() < 1e-6);

    // erfcx(x)·e^{-x²} = erfc(x); ln(1−e^x) complements ln(1+e^x); scaled
    // Bessel recovers ln I₀ and the I₁/I₀ ratio.
    let x = 0.6_f64;
    assert!((x.erfcx() * (-x * x).exp() - LazyReal::erfc(x)).abs() < 1e-12);
    assert!((LazyReal::ln_1m_exp(&(-0.5_f64)).exp() - (1.0 - (-0.5_f64).exp())).abs() < 1e-12);
    let k = 2.5_f64;
    assert!((k.bessel_i0e().ln() + k - crate::special::bessel::ln_i0(k)).abs() < 1e-9);
    assert!((k.bessel_i1e() / k.bessel_i0e() - crate::special::bessel::i1_over_i0(k)).abs() < 1e-9);
    // f32 paths narrow through f64.
    assert!((0.5_f32.erfcx() - 0.5_f64.erfcx() as f32).abs() < 1e-6);
}
