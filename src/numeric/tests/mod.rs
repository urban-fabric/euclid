use super::*;

mod control_flow;

/// A formula written once over the lazy analysis base — analysis plus a scalar constant.
fn shifted_quadratic<T: LazyReal>(x: T) -> T {
    let half = T::half();
    x.clone() * x.clone() + half * x
}

fn vector_space_midpoint<V: VectorSpace>(a: V, b: V) -> V {
    let half = V::Scalar::half();
    (a + b).scale(half)
}

fn inner_product_energy<V: InnerProductSpace>(x: V) -> V::Scalar {
    x.clone().dot(x)
}

#[test]
fn real_formula_runs_on_f64_and_f32() {
    assert!((shifted_quadratic(2.5_f64) - 7.5).abs() < 1e-12);
    assert!((shifted_quadratic(2.5_f32) - 7.5).abs() < 1e-5);
}

#[test]
fn scalar_carriers_are_vector_spaces_over_their_real_part() {
    assert_eq!(<f64 as VectorSpace>::zero(), 0.0);
    assert_eq!(<f32 as VectorSpace>::zero(), 0.0);
    assert_eq!(VectorSpace::scale(3.0_f64, 2.0), 6.0);
    assert_eq!(VectorSpace::scale(3.0_f32, 2.0), 6.0);
    assert_eq!(vector_space_midpoint(2.0_f64, 6.0), 4.0);
    assert_eq!(vector_space_midpoint(2.0_f32, 6.0), 4.0);
}

#[test]
fn scalar_carriers_are_inner_product_and_normed_spaces() {
    assert_eq!(InnerProductSpace::dot(3.0_f64, 4.0), 12.0);
    assert_eq!(InnerProductSpace::dot(3.0_f32, 4.0), 12.0);
    assert_eq!(inner_product_energy(-3.0_f64), 9.0);
    assert_eq!(inner_product_energy(-3.0_f32), 9.0);
    assert_eq!(NormedVectorSpace::norm_squared(-3.0_f64), 9.0);
    assert_eq!(NormedVectorSpace::norm_squared(-3.0_f32), 9.0);
    assert_eq!(NormedVectorSpace::norm(-3.0_f64), 3.0);
    assert_eq!(NormedVectorSpace::norm(-3.0_f32), 3.0);
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

    assert!((f64::euler_gamma() - 0.577_215_664_901_532_9).abs() < 1e-15);
    assert!((f32::euler_gamma() - 0.577_215_7).abs() < 1e-6);
}

/// `powf` takes its exponent in the same real substrate, so a power law is write-once over
/// [`LazyReal`] with the exponent itself a value of the surface.
fn power_law<T: LazyReal>(base: T, exponent: T) -> T {
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

/// `powi` takes its exponent from the integer substrate [`LazyReal::Integer`], not a host
/// `i32`, so an integer power law is write-once over the surface.
fn integer_power_law<T: LazyReal>(base: T, exponent: T::Integer) -> T {
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
    assert_eq!(<f64 as LazyReal>::powi(2.0, beyond_i32), f64::INFINITY);
    assert_eq!(<f64 as LazyReal>::powi(0.5, beyond_i32), 0.0);
}

fn square_plus_self<T: LazyReal>(x: T) -> T {
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
    assert!((<f64 as LazyReal>::ln_1p(&tiny) - tiny).abs() < 1e-30);
    assert_eq!(tiny.exp() - 1.0, 0.0);
    assert!((<f64 as LazyReal>::exp_m1(&tiny) - tiny).abs() < 1e-30);
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
    assert!((<f32 as LazyReal>::ln_1p(&tiny32) - tiny32).abs() < 1e-12);
    assert!((<f32 as LazyReal>::exp_m1(&tiny32) - tiny32).abs() < 1e-12);
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

/// A formula on [`LazyReal`] using `atan2`, trig, and the `clamp` selector — once a
/// concrete-only API, now compiling for every lazy backend.
fn wrapped_angle_unit<T: LazyReal>(x: T, y: T) -> T {
    let angle = y.atan2(x);
    let pythagorean = angle.clone().sin() * angle.clone().sin() + angle.clone().cos() * angle.cos();
    pythagorean.clamp(T::zero(), T::one())
}

#[test]
fn lazy_real_carries_trig_and_select_on_f64_and_f32() {
    // sin²+cos² = 1, clamped to [0, 1], through the LazyReal bound alone.
    assert!((wrapped_angle_unit(0.7_f64, 1.3_f64) - 1.0).abs() < 1e-12);
    assert!((wrapped_angle_unit(0.7_f32, 1.3_f32) - 1.0).abs() < 1e-6);
    let (s, c) = <f64 as LazyReal>::sin_cos(0.7);
    assert!((s * s + c * c - 1.0).abs() < 1e-12);
    let (s, c) = <f32 as LazyReal>::sin_cos(0.7);
    assert!((s * s + c * c - 1.0).abs() < 1e-6);
    assert_eq!(LazyReal::max(LazyReal::min(3.0_f64, 5.0), 1.0), 3.0);
    assert_eq!(LazyReal::clamp(2.0_f32, 0.0, 1.0), 1.0);
}

/// Inverse hyperbolics over [`LazyReal`] invert their forward partner.
fn asinh_then_sinh<T: LazyReal>(x: T) -> T {
    x.asinh().sinh()
}

/// Algebraic spelling of `asinh`, used as a reference at moderate inputs.
fn asinh_via_ln<T: LazyReal>(x: T) -> T {
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

/// The Normal CDF `Φ(z) = ½·erfc(−z/√2)` over [`LazyReal`], lowering to a backend's `Erfc`.
fn normal_cdf<T: LazyReal>(z: T) -> T {
    T::half() * (-z / T::sqrt_2()).erfc()
}

#[test]
fn lazy_real_special_functions_run_on_f64_and_f32() {
    assert!((normal_cdf(0.0_f64) - 0.5).abs() < 1e-15);
    assert!((normal_cdf(0.0_f32) - 0.5).abs() < 1e-6);

    // Spot-check the defining identities through the trait.
    assert!((LazyReal::ln_gamma(5.0_f64) - 24.0_f64.ln()).abs() < 1e-9);
    assert!((LazyReal::digamma(1.0_f64) + f64::euler_gamma()).abs() < 1e-9);
    assert!((LazyReal::ln_beta(2.0_f64, 3.0) - (1.0_f64 / 12.0).ln()).abs() < 1e-10);
    assert!((LazyReal::beta(2.0_f64, 3.0) - 1.0 / 12.0).abs() < 1e-10);
    // P(a, x) + Q(a, x) = 1.
    let p = LazyReal::gamma_lr(3.0_f64, 2.5);
    let q = LazyReal::gamma_ur(3.0_f64, 2.5);
    assert!((p + q - 1.0).abs() < 1e-10);
    // I_x(1, 1) = x.
    assert!((LazyReal::beta_reg(0.37_f64, 1.0, 1.0) - 0.37).abs() < 1e-12);
    assert!((LazyReal::erf(1.0_f32) - 0.842_700_8).abs() < 1e-6);

    // erfcx(x)·e^{-x²} = erfc(x); ln(1−e^x) complements ln(1+e^x); scaled Bessel recovers
    // ln I₀ and the I₁/I₀ ratio.
    let x = 0.6_f64;
    assert!((x.erfcx() * (-x * x).exp() - LazyReal::erfc(x)).abs() < 1e-12);
    assert!((LazyReal::ln_1m_exp(&(-0.5_f64)).exp() - (1.0 - (-0.5_f64).exp())).abs() < 1e-12);
    let k = 2.5_f64;
    assert!((k.bessel_i0e().ln() + k - crate::numeric::special::bessel::ln_i0(k)).abs() < 1e-9);
    assert!((k.ln_bessel_i0() - crate::numeric::special::bessel::ln_i0(k)).abs() < 1e-9);
    assert!((k.bessel_i1_over_i0() - crate::numeric::special::bessel::i1_over_i0(k)).abs() < 1e-9);
    assert!(
        (k.bessel_i1e() / k.bessel_i0e() - crate::numeric::special::bessel::i1_over_i0(k)).abs()
            < 1e-9
    );
    // Native f32 paths stay close to the f64 reference.
    assert!((0.5_f32.erfcx() - 0.5_f64.erfcx() as f32).abs() < 1e-6);
    assert!((LazyReal::beta(2.0_f32, 3.0) - 1.0 / 12.0).abs() < 1e-5);
}

/// A write-once ring formula (`+ - *`, no division). The binomial step `(n − k)·(k + 1)` is
/// the count arithmetic a discrete distribution does before reaching the real surface.
fn ring_step<I: LazyInteger>(n: I, k: I) -> I {
    (n - k.clone()) * (k + I::one())
}

#[test]
fn lazy_ring_formula_runs_on_i32_i64_and_i128() {
    // (10 − 3)·(3 + 1) = 28, on every signed integer carrier.
    assert_eq!(ring_step(10_i32, 3), 28);
    assert_eq!(ring_step(10_i64, 3), 28);
    assert_eq!(ring_step(10_i128, 3), 28);
    // The width-matched cast into the real surface.
    assert_eq!(3_i32.to_real(), 3.0_f32);
    assert_eq!(3_i64.to_real(), 3.0_f64);
    // Lift, sign, abs, Euclidean div/rem.
    assert_eq!(i64::from_i64(7), 7);
    assert_eq!(LazyInteger::signum(-7_i64), -1);
    assert_eq!(LazyInteger::abs(-7_i64), 7);
    assert_eq!(7_i64.div_rem(3), (2, 1));
    assert_eq!((-7_i64).div_rem(3), (-3, 2));
    assert_eq!(7_i64.div_rem(-3), (-2, 1));
    assert_eq!((-7_i64).div_rem(-3), (3, 2));
    assert_eq!((-7_i32).div_rem(3), (-3, 2));
    assert_eq!((-7_i128).div_rem(3), (-3, 2));
}

/// The integer→real bridge: a Poisson log-mass kernel with an integer count and real
/// parameters, crossing into the real surface once via [`LazyInteger::to_real`].
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

/// A write-once semiring formula (`+ *`, no subtraction): `n·n + n` — the count arithmetic a
/// formula does staying inside the naturals, before any cast leaves the semiring.
fn semiring_step<N: LazyNatural>(n: N) -> N {
    n.clone() * n.clone() + n
}

#[test]
fn lazy_semiring_formula_runs_on_u32_u64_and_u128() {
    // 6·6 + 6 = 42, on every natural carrier.
    assert_eq!(semiring_step(6_u32), 42);
    assert_eq!(semiring_step(6_u64), 42);
    assert_eq!(semiring_step(6_u128), 42);
    // The width-matched cast into the real surface.
    assert_eq!(3_u32.to_real(), 3.0_f32);
    assert_eq!(3_u64.to_real(), 3.0_f64);
    // Lift, the widen into the signed ring, Euclidean div/rem (floor division, no sign).
    assert_eq!(u64::from_u64(7), 7);
    assert_eq!(7_u32.to_integer(), 7_i64);
    assert_eq!(7_u64.to_integer(), 7_i128);
    assert_eq!(7_u64.div_rem(3), (2, 1));
    assert_eq!(5_u64.factorial(), 120.0);
    assert_eq!(5_u64.binomial(&2), 10.0);
    assert_eq!(5_u32.factorial(), 120.0);
    assert_eq!(5_u32.binomial(&2), 10.0);
    assert_eq!(5_u128.factorial(), 120.0);
    assert_eq!(5_u128.binomial(&2), 10.0);
    assert_eq!(3_u128.binomial(&4), 0.0);
    assert_eq!(3_u128.ln_binomial(&4), f64::NEG_INFINITY);
    assert!((5_u32.ln_factorial() - 120.0_f32.ln()).abs() < 1e-6);
}

/// The natural→integer bridge: a signed difference of counts crosses into the ring via
/// [`LazyNatural::to_integer`], where the subtraction the semiring withholds is available —
/// the result is unrepresentable in the naturals themselves.
fn count_difference<N: LazyNatural>(a: N, b: N) -> N::Integer {
    a.to_integer() - b.to_integer()
}

#[test]
fn lazy_natural_bridges_into_the_integer_ring() {
    // 3 − 7 = −4 in the signed ring; the naturals have no such value.
    assert_eq!(count_difference(3_u32, 7_u32), -4_i64);
    assert_eq!(count_difference(3_u64, 7_u64), -4_i128);
}

/// A write-once piecewise kernel over the mask tier: `relu(x) = select(x > 0, x, 0)`, a
/// comparison to a [`LazyCompare::Mask`] driving a branchless `select` with no scalar `bool`.
fn relu<T: LazyCompare>(x: T, zero: T) -> T {
    let positive = x.gt(&zero);
    T::select(&positive, x, zero)
}

#[test]
fn lazy_bool_mask_drives_select_and_logic() {
    // relu on both a real and an integer carrier, write-once.
    assert_eq!(relu(3.0_f64, 0.0), 3.0);
    assert_eq!(relu(-2.0_f64, 0.0), 0.0);
    assert_eq!(relu(5_i32, 0), 5);
    assert_eq!(relu(-7_i32, 0), 0);
    assert_eq!(relu(5_i64, 0), 5);
    assert_eq!(relu(-7_i64, 0), 0);

    // Naturals are ordered carriers too: min(4, 7) = 4 via comparison + select.
    let lt = LazyCompare::lt(&4_u64, &7_u64);
    assert_eq!(u64::select(&lt, 4, 7), 4);

    // The Boolean algebra on the masks (trait-qualified to avoid std's `PartialOrd`).
    let t = LazyCompare::lt(&2.0_f64, &3.0); // true
    let f = LazyCompare::gt(&2.0_f64, &3.0); // false
    assert!(LazyBool::or(&t, &f));
    assert!(!LazyBool::and(&t, &f));
    assert!(LazyBool::xor(&t, &f));
    assert!(LazyBool::not(&f));
}

#[test]
fn eager_integer_masks_drive_host_while_loop() {
    let signed = <i32 as LazyCompare>::while_loop(0_i32, |n| LazyCompare::lt(n, &5), |n| n + 1);
    assert_eq!(signed, 5);

    let natural = <u64 as LazyCompare>::while_loop(1_u64, |n| LazyCompare::lt(n, &16), |n| n * 2);
    assert_eq!(natural, 16);
}

/// `if`/`then`/`else` over the real surface *alone*. Because `LazyReal: LazyCompare`, a
/// piecewise kernel needs no separate `+ LazyCompare` bound: `signed_clip(x) = if |x| > 1
/// { sign(x) } else { x }` — the branchless saturation a truncated density reaches for — is
/// write-once over [`LazyReal`], lowering to `Abs`/`Gt`/`Select` on a tensor backend.
fn signed_clip<T: LazyReal>(x: T) -> T {
    let outside = x.abs().gt(&T::one());
    T::if_then_else(&outside, x.clone().signum(), x)
}

#[test]
fn lazy_real_carries_comparison_and_if_then_else() {
    // Control flow reached through the `LazyReal` bound alone, on every real tier.
    assert_eq!(signed_clip(0.4_f64), 0.4);
    assert_eq!(signed_clip(2.5_f64), 1.0);
    assert_eq!(signed_clip(-3.0_f32), -1.0);

    // The six comparisons return the mask tier; `select` / `if_then_else` consume it.
    assert!(LazyCompare::ge(&2.0_f64, &1.0));
    assert_eq!(
        f64::if_then_else(&LazyCompare::lt(&1.0_f64, &2.0), 10.0, 20.0),
        10.0
    );
    assert_eq!(
        <f32 as LazyCompare>::select(&LazyCompare::gt(&1.0_f32, &2.0), 10.0, 20.0),
        20.0
    );
}

/// Scrub `NaN` to a fallback — a `LazyReal`-generic kernel composing the `is_nan` predicate
/// with `if_then_else`, the shape a guarded log-density uses outside its support.
fn scrub_nan<T: LazyReal>(x: T, fallback: T) -> T {
    T::if_then_else(&x.is_nan(), fallback, x)
}

#[test]
fn lazy_real_non_finite_sentinels_and_predicates() {
    // The sentinels lift through `scalar` on every real tier.
    assert_eq!(f64::infinity(), f64::INFINITY);
    assert_eq!(f64::neg_infinity(), f64::NEG_INFINITY);
    assert_eq!(f32::infinity(), f32::INFINITY);
    assert!(f64::nan().is_nan());

    // Trait-qualified calls bypass the inherent `f64::is_nan` etc. and exercise the
    // `LazyReal` classifier surface.
    assert!(LazyReal::is_infinite(&f64::infinity()));
    assert!(LazyReal::is_infinite(&f64::neg_infinity()));
    assert!(LazyReal::is_nan(&f64::nan()));
    assert!(!LazyReal::is_finite(&f64::nan()));
    assert!(!LazyReal::is_finite(&f64::infinity()));
    assert!(LazyReal::is_finite(&3.5_f64));
    assert!(!LazyReal::is_nan(&3.5_f64));
    assert!(LazyReal::is_nan(&f32::nan()));

    // The predicates compose with the control-flow surface, write-once over `LazyReal`.
    assert_eq!(scrub_nan(f64::nan(), 0.0), 0.0);
    assert_eq!(scrub_nan(7.0_f64, 0.0), 7.0);
    assert_eq!(scrub_nan(f32::nan(), -1.0), -1.0);
}

#[test]
fn lazy_real_non_finite_recovery_composites() {
    // `nan_to_num` scrubs NaN but preserves an intended ±∞.
    assert_eq!(f64::nan().nan_to_num(0.0), 0.0);
    assert_eq!(5.0_f64.nan_to_num(0.0), 5.0);
    assert_eq!(f64::infinity().nan_to_num(0.0), f64::INFINITY);

    // `finite_or` forces finiteness, replacing NaN *and* ±∞.
    assert_eq!(f64::nan().finite_or(-1.0), -1.0);
    assert_eq!(f64::infinity().finite_or(-1.0), -1.0);
    assert_eq!(f64::neg_infinity().finite_or(-1.0), -1.0);
    assert_eq!(3.0_f64.finite_or(-1.0), 3.0);

    // `mask_invalid` is the out-of-support output guard: keep where valid, else fill.
    assert_eq!(
        7.0_f64.mask_invalid(false, f64::neg_infinity()),
        f64::NEG_INFINITY
    );
    assert_eq!(7.0_f64.mask_invalid(true, f64::neg_infinity()), 7.0);

    // The checkify hook is the identity on a finite value (and is erased in release).
    assert_eq!(2.0_f64.debug_assert_finite(), 2.0);
    assert_eq!(2.0_f32.debug_assert_finite(), 2.0);
}
