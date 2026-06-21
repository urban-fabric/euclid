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

    // The log-base conversions are mutual inverses through `e` / `10`.
    assert!((f64::log2_e() * f64::ln_2() - 1.0).abs() < 1e-12);
    assert!((f64::log10_e() * f64::ln_10() - 1.0).abs() < 1e-12);
    assert!((f64::log2_10() * f64::log10_2() - 1.0).abs() < 1e-12);

    // Euler–Mascheroni γ, to its leading digits.
    assert!((f64::euler_gamma() - 0.577_215_664_901_532_9).abs() < 1e-15);
    assert!((f32::euler_gamma() - 0.577_215_7).abs() < 1e-6);
}

/// `powf` now takes its exponent in the same real substrate, so a power law can be
/// written once over [`LazyReal`] with the exponent itself a value of the surface —
/// here recovered as `log_b(b) = 1` so the result is just the base. A literal
/// exponent is lifted with [`scalar_like`](LazyReal::scalar_like).
fn power_law<T: LazyReal>(base: T, exponent: T) -> T {
    base.powf(exponent)
}

#[test]
fn lazy_real_powf_takes_a_real_exponent() {
    // Exponent lifted as a `Self`, not an `f64`: the generic surface never sees `f64`.
    assert!((power_law(2.0_f64, 2.0_f64.scalar_like(10.0)) - 1024.0).abs() < 1e-9);
    assert!((power_law(2.0_f32, 2.0_f32.scalar_like(10.0)) - 1024.0).abs() < 1e-2);
    // The exponent can be a computed value of the surface, not just a constant:
    // x^(ln x / ln x) = x^1 = x.
    let x = 3.7_f64;
    assert!((x.powf(x.ln() / x.ln()) - x).abs() < 1e-12);
}

/// `powi` now takes its exponent from the integer substrate [`LazyReal::Integer`],
/// not a host `i32`, so an integer power law is write-once over the surface with the
/// exponent a value of the integer tier (here `T::Integer`, an [`LazyInteger`]).
fn integer_power_law<T: LazyReal>(base: T, exponent: T::Integer) -> T {
    base.powi(exponent)
}

#[test]
fn lazy_real_powi_takes_an_integer_from_the_integer_tier() {
    // Exponent is `<f64 as LazyReal>::Integer` (= i64, a `LazyInteger`), never an `i32`.
    assert!((integer_power_law(2.0_f64, 10) - 1024.0).abs() < 1e-12);
    assert!((integer_power_law(2.0_f32, 10) - 1024.0).abs() < 1e-3);
    // Negative bases and exponents — the exact, repeated-multiplication semantics
    // `powf` cannot give: (-2)^3 = -8, 2^-2 = 0.25.
    assert_eq!((-2.0_f64).powi(3), -8.0);
    assert!((2.0_f64.powi(-2) - 0.25).abs() < 1e-12);
}

#[test]
fn softplus_is_stable_at_extremes() {
    assert!((f64::ln_1p_exp(&0.0) - core::f64::consts::LN_2).abs() < 1e-12);
    let big = 50.0_f64;
    assert!((big.ln_1p_exp() - big).abs() < 1e-10);
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

/// `asinh`/`atanh` written once over [`LazyReal`] — the inverse hyperbolics the
/// conformal map projections (Transverse Mercator) are written in. The native
/// `f64`/`f32` overrides must agree with the `ln`/`sqrt` default, and each must
/// invert its forward partner.
fn asinh_then_sinh<T: LazyReal>(x: T) -> T {
    x.asinh().sinh()
}

/// The default (`ln`/`sqrt`) spelling of `asinh`, to check the native override
/// agrees with it. Mirrors the trait default so the two paths can be compared.
fn asinh_via_ln<T: LazyReal>(x: T) -> T {
    (x.clone() + (x.clone() * x + T::one()).sqrt()).ln()
}

#[test]
fn lazy_real_inverse_hyperbolics_invert_and_match_default() {
    // sinh∘asinh = id and tanh∘atanh = id on both concrete reals.
    assert!((asinh_then_sinh(1.3_f64) - 1.3).abs() < 1e-12);
    assert!((asinh_then_sinh(1.3_f32) - 1.3).abs() < 1e-5);
    assert!((0.4_f64.atanh().tanh() - 0.4).abs() < 1e-12);
    // acosh∘cosh = id for x ≥ 0.
    assert!((2.0_f64.cosh().acosh() - 2.0).abs() < 1e-12);

    // The native libm override agrees with the `ln`/`sqrt` default impl.
    assert!((1.3_f64.asinh() - asinh_via_ln(1.3_f64)).abs() < 1e-12);
    assert!((1.3_f32.asinh() - asinh_via_ln(1.3_f32)).abs() < 1e-5);
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

    // The methods delegate to `crate::numeric::special`; spot-check the defining
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
    assert!((k.bessel_i0e().ln() + k - crate::numeric::special::bessel::ln_i0(k)).abs() < 1e-9);
    assert!(
        (k.bessel_i1e() / k.bessel_i0e() - crate::numeric::special::bessel::i1_over_i0(k)).abs()
            < 1e-9
    );
    // f32 paths narrow through f64.
    assert!((0.5_f32.erfcx() - 0.5_f64.erfcx() as f32).abs() < 1e-6);
}

/// A write-once **ring** formula — `+ - *`, no division — over the lazy ring base.
/// The binomial-coefficient step `(n − k)·(k + 1)` is exactly the count arithmetic a
/// discrete distribution does before it ever reaches the real surface.
fn ring_step<I: LazyInteger>(n: I, k: I) -> I {
    (n - k.clone()) * (k + I::one())
}

#[test]
fn lazy_ring_formula_runs_on_i32_i64_and_i128() {
    // (10 − 3)·(3 + 1) = 28, on every signed integer carrier.
    assert_eq!(ring_step(10_i32, 3), 28);
    assert_eq!(ring_step(10_i64, 3), 28);
    assert_eq!(ring_step(10_i128, 3), 28);
    // The width-matched cast into the real surface: i32 → f32, i64 → f64.
    assert_eq!(3_i32.to_real(), 3.0_f32);
    assert_eq!(3_i64.to_real(), 3.0_f64);
    // The integer surface: lift, sign, abs, Euclidean div/rem.
    assert_eq!(i64::from_i64(7), 7);
    assert_eq!(LazyInteger::signum(-7_i64), -1);
    assert_eq!(LazyInteger::abs(-7_i64), 7);
    assert_eq!(7_i64.div_rem(3), (2, 1)); // 7 = 2·3 + 1
    assert_eq!((-7_i64).div_rem(3), (-2, -1)); // truncates toward zero
}

/// The integer→real bridge: a Poisson-style log-mass kernel whose **count** is a
/// [`LazyInteger`] and whose **parameters** are [`LazyReal`], crossing into the real
/// surface exactly once via [`LazyInteger::to_real`]. This is the shape the discrete
/// `bayes` kernels take once counts are integer-typed on a tensor backend.
fn poisson_log_pmf<I, T>(k: I, lambda: T, ln_factorial_k: T) -> T
where
    I: LazyInteger<Real = T>,
    T: LazyReal,
{
    k.to_real() * lambda.clone().ln() - lambda - ln_factorial_k
}

#[test]
fn lazy_integer_bridges_into_the_real_surface() {
    // Poisson(λ=2) log-pmf at k=3: 3·ln2 − 2 − ln(3!) = 3·ln2 − 2 − ln6.
    let got = poisson_log_pmf(3_i64, 2.0_f64, 6.0_f64.ln());
    let want = 3.0 * 2.0_f64.ln() - 2.0 - 6.0_f64.ln();
    assert!((got - want).abs() < 1e-12);
}

/// A write-once **piecewise** kernel over the mask tier: `relu(x) = select(x > 0, x, 0)`
/// — a comparison to a [`LazyCompare::Mask`] driving a branchless `select`, running on any
/// ordered carrier (real or integer) with no scalar `bool` in the formula.
fn relu<T: LazyCompare>(x: T, zero: T) -> T {
    let positive = x.gt(&zero);
    T::select(&positive, x, zero)
}

#[test]
fn lazy_bool_mask_drives_select_and_logic() {
    // relu on both a real and an integer carrier, write-once.
    assert_eq!(relu(3.0_f64, 0.0), 3.0);
    assert_eq!(relu(-2.0_f64, 0.0), 0.0);
    assert_eq!(relu(5_i64, 0), 5);
    assert_eq!(relu(-7_i64, 0), 0);

    // The Boolean algebra on the masks (trait-qualified to avoid the std `PartialOrd`
    // method of the same name on the concrete scalar).
    let t = LazyCompare::lt(&2.0_f64, &3.0); // true
    let f = LazyCompare::gt(&2.0_f64, &3.0); // false
    assert!(LazyBool::or(&t, &f));
    assert!(!LazyBool::and(&t, &f));
    assert!(LazyBool::xor(&t, &f));
    assert!(LazyBool::not(&f));
}
