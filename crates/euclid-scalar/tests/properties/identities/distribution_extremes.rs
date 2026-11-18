//! Independent distribution identities near the limits of the floating-point formats.

use super::*;

property! {
    #![config(Config::default().cases(200).require("lower half", 10.0).require("upper half", 10.0))]

    fn huge_gamma_quantiles_round_to_the_shape(
        p in generate::float::in_range(0.125, 0.875),
        fraction in generate::float::in_range(0.75, 1.0),
    ) {
        classify!("lower half", p < 0.5);
        classify!("upper half", p > 0.5);
        // Gamma(a, 1) has mean a and variance a. Chebyshev puts these quantiles
        // within 3 sqrt(a) of a, far inside half an ulp for both tested formats.
        let shape = f64::MAX * fraction;
        check!(p.gamma_lr_inv(shape) == shape, "f64 gamma quantile must round to its shape");
        let shape = f32::MAX * fraction as f32;
        check!((p as f32).gamma_lr_inv(shape) == shape, "f32 gamma quantile must round to its shape");
    }
}

property! {
    #![config(Config::default().cases(200).require("subnormal x", 10.0).require("normal x", 10.0))]

    fn beta_with_one_unit_shape_matches_the_integrated_density(
        fraction in generate::float::in_range(0.125, 8.0),
        shape_fraction in generate::float::in_range(0.75, 1.0),
    ) {
        // Integrating the densities b(1-t)^(b-1) and a*t^(a-1) gives these closed forms.
        let shape = f64::MAX * shape_fraction;
        let x = fraction / shape;
        classify!("subnormal x", x.is_subnormal());
        classify!("normal x", x.is_normal());
        let lower = -(shape * (-x).ln_1p()).exp_m1();
        relation::agree("I_x(1,b)", x, |x| x.beta_reg(1.0, shape), |_| lower, &Ulps(2))?;

        relation::agree("I_x(1/2,1)", x, |x| x.beta_reg(0.5, 1.0), f64::sqrt, &Ulps(2))?;
    }
}

property! {
    #![config(Config::default().cases(200))]

    fn negative_half_integer_digamma_matches_the_harmonic_sum(n in generate::uint(1, 1024)) {
        // Apply the digamma recurrence to the known value ψ(1/2) = -γ - 2 ln(2).
        let harmonic: f64 = (1..=n).map(|k| 1.0 / (k as f64 - 0.5)).sum();
        let expected = harmonic - f64::euler_gamma() - 2.0 * core::f64::consts::LN_2;
        let x = 0.5 - n as f64;
        relation::agree("half-integer digamma", x, f64::digamma, |_| expected, &Absolute(4e-14))?;
        relation::agree(
            "single-precision half-integer digamma", x, |x| f64::from((x as f32).digamma()),
            |_| expected, &Absolute(2e-6),
        )?;
    }
}

#[test]
fn gamma_large_shape_tails_saturate_at_the_format_limits() {
    for shape in [3e7, f64::MAX] {
        assert_eq!(f64::MIN_POSITIVE.gamma_lr(shape), 0.0);
        assert_eq!(f64::MIN_POSITIVE.gamma_ur(shape), 1.0);
    }
    assert_eq!(f64::MAX.gamma_lr(3e7), 1.0);
    assert_eq!(f64::MAX.gamma_ur(3e7), 0.0);
    for shape in [1e3, f32::MAX] {
        assert_eq!(f32::MIN_POSITIVE.gamma_lr(shape), 0.0);
        assert_eq!(f32::MIN_POSITIVE.gamma_ur(shape), 1.0);
    }
    assert_eq!(f32::MAX.gamma_lr(1e3), 1.0);
    assert_eq!(f32::MAX.gamma_ur(1e3), 0.0);
}

#[test]
fn gamma_quantile_checks_the_format_spacing_before_snapping_to_the_shape() {
    // At these powers of two half the spacing below the shape is exactly 64 standard deviations.
    let shape = 2.0_f64.powi(120);
    assert_eq!(f64::from_bits(1).gamma_lr_inv(shape), shape);
    assert_eq!(1.0_f64.next_down().gamma_lr_inv(shape), shape);
    let shape = 2.0_f32.powi(62);
    assert_eq!(f32::from_bits(1).gamma_lr_inv(shape), shape);
    assert_eq!(1.0_f32.next_down().gamma_lr_inv(shape), shape);

    // With finer spacing the lower quantile is still distinguishable from the mean.
    let shape = 2.0_f64.powi(110);
    assert!(0.0001_f64.gamma_lr_inv(shape) < shape);
    let shape = 2.0_f32.powi(52);
    assert!(0.0001_f32.gamma_lr_inv(shape) < shape);
}

#[test]
fn gamma_distribution_requires_a_positive_finite_shape() {
    for shape in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(f64::INFINITY.gamma_lr(shape).is_nan());
        assert!(f64::INFINITY.gamma_ur(shape).is_nan());
        assert!(0.5_f64.gamma_lr_inv(shape).is_nan());
        let shape = shape as f32;
        assert!(f32::INFINITY.gamma_lr(shape).is_nan());
        assert!(f32::INFINITY.gamma_ur(shape).is_nan());
        assert!(0.5_f32.gamma_lr_inv(shape).is_nan());
    }

    for x in [-1.0, f64::NEG_INFINITY, f64::NAN] {
        assert!(x.gamma_lr(3e7).is_nan());
        assert!(x.gamma_ur(3e7).is_nan());
        assert!((x as f32).gamma_lr(1e3).is_nan());
        assert!((x as f32).gamma_ur(1e3).is_nan());
    }
    for p in [-0.25, 1.25, f32::NAN, f32::INFINITY] {
        assert!(p.gamma_lr_inv(2.0).is_nan());
    }
}

#[test]
fn beta_distribution_requires_two_positive_finite_shapes() {
    for shape in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(0.0_f64.beta_reg(shape, 1.0).is_nan());
        assert!(1.0_f64.beta_reg(1.0, shape).is_nan());
        assert!(0.5_f64.beta_reg_inv(shape, 1.0).is_nan());
        assert!(0.5_f64.beta_reg_inv(1.0, shape).is_nan());
        assert!(0.5_f32.beta_reg_inv(shape as f32, 1.0).is_nan());
        assert!(0.5_f32.beta_reg_inv(1.0, shape as f32).is_nan());
    }
}

#[test]
fn single_precision_beta_quantile_preserves_the_symmetric_median() {
    // The starting estimate stays usable when multiplying the shapes would overflow or underflow.
    for shape in [1e-23, 1e20, f32::MAX] {
        assert_eq!(0.5_f32.beta_reg_inv(shape, shape), 0.5);
    }
}

#[test]
fn single_precision_erfc_inverse_rounds_to_the_nearest_value() {
    // Include the input found by Bayes and the two smallest positive f32 probabilities.
    for y in [
        f32::from_bits(0x3e84_5270),
        f32::from_bits(1),
        f32::from_bits(2),
    ] {
        let root = y.erfc_inv();
        assert!(root.is_finite());
        let lower = f64::from(root).midpoint(f64::from(root.next_down()));
        let upper = f64::from(root).midpoint(f64::from(root.next_up()));
        assert!(Real::erfc(upper) < f64::from(y));
        assert!(f64::from(y) < Real::erfc(lower));
    }
}

#[test]
fn extremely_asymmetric_large_beta_matches_its_gamma_limit() {
    // b*X tends to Gamma(a, 1) for X ~ Beta(a,b) as b goes to infinity.
    // Here the O(a²/b) correction is negligible beside the CDF error at a=1e7.
    let a = 1e7;
    let b = 1.1e308_f64;
    let x = a / b;
    let actual = x.beta_reg(a, b);
    assert!(
        (actual - (b * x).gamma_lr(a)).abs() < 1e-7,
        "I_x(a,b) = {actual}"
    );
}

#[test]
fn tiny_gamma_upper_tails_match_mpfr() {
    // Regenerate with crates/euclid-scalar/tests/references/extreme_shapes.py.
    for (a, x, expected) in [
        (1e-320, 0.5_f64, 5.6e-321),
        (1e-320, 1.0, 2.194e-321),
        (1e-18, 0.5, 5.597_735_947_761_608e-19),
        (1e-12, 1.0, 2.193_839_343_957_447_4e-13),
        (1e-3, 0.125, 0.001_622_813_890_312_121_4),
        (1e-3, 1.0, 0.000_219_608_357_585_556_4),
        (1e-3, 2.0, 4.897_691_783_098_148e-5),
    ] {
        let actual = x.gamma_ur(a);
        if expected < f64::MIN_POSITIVE {
            assert_eq!(actual, expected);
        } else {
            assert!(
                (actual / expected - 1.0).abs() < 5e-14,
                "Q({a}, {x}) = {actual}"
            );
            assert!((x.gamma_lr(a) + actual - 1.0).abs() <= 2.0 * f64::EPSILON);
        }
    }
    assert!(Real::ln_gamma(f64::from_bits(1)).is_finite());
    assert!(Real::ln_gamma(f32::from_bits(1)).is_finite());
    assert!(0.5_f32.gamma_ur(1e-20) > 0.0);
    assert_eq!(1.0_f32.gamma_ur(f32::from_bits(8)).to_bits(), 2);
}

#[test]
fn extreme_beta_matches_independent_references() {
    // MPFR log gamma and the finite integer-shape sum in crates/euclid-scalar/tests/references/extreme_shapes.py.
    for (a, b, expected) in [
        (1e8_f64, 1e8, -138_629_444.056_817_32),
        (1e16, 1e16, -1.386_294_361_119_892_4e16),
        (1e306, 1e306, -1.386_294_361_119_890_6e306),
        (2e307, 1.7e308, -6.393_419_394_086_805e307),
        (1e300, 1.7e308, -1.995_130_899_795_571_4e301),
    ] {
        let actual = a.ln_beta(b);
        assert!(
            (actual / expected - 1.0).abs() < 4.0 * f64::EPSILON,
            "ln B({a},{b}) = {actual}"
        );
        assert_eq!(actual, b.ln_beta(a));
    }
    assert_eq!(f64::MAX.ln_beta(f64::MAX), f64::NEG_INFINITY);
    assert_eq!(
        1e37_f32.ln_beta(1e37),
        1e37_f64.ln_beta(f64::from(1e37_f32)) as f32
    );
    for (a, b, expected) in [
        (2.0_f64, 1.1e308, 0.593_994_150_290_162),
        (5.0, 1.1e308, 0.559_506_714_934_787_7),
        (10.0, 1e25, 0.542_070_285_528_147_8),
    ] {
        let actual = (a / b).beta_reg(a, b);
        assert!(
            (actual - expected).abs() < 2e-13,
            "I_{{a/b}}({a},{b}) = {actual}"
        );
        assert_eq!(1.0_f64.next_down().beta_reg(b, a), 0.0);
    }
    for (a, b, x, expected) in [
        (5.0, 1e18, 1e-17_f64, 0.970_747_311_923_039),
        (5.0, 1e9, 1e-8, 0.970_747_313_247_203_5),
        (2.0, 1e9, 4e-9, 0.908_421_806_288_954_6),
    ] {
        let actual = x.beta_reg(a, b);
        assert!((actual - expected).abs() < 5e-15, "I_x({a},{b}) = {actual}");
    }
    let reflected = 0.999_999_99_f64.beta_reg(1e9, 5.0);
    assert!((reflected - 0.029_252_685_802_281_02).abs() < 5e-15);
}

#[test]
fn double_precision_erfc_inverse_resolves_subnormal_inputs() {
    // Regenerate with crates/euclid-scalar/tests/references/extreme_shapes.py.
    for (bits, expected) in [(1, 27.213_293_210_812_95), (3, 27.193_114_126_203_973)] {
        let actual = f64::from_bits(bits).erfc_inv();
        assert!((actual - expected).abs() < 4e-14, "erfc inverse = {actual}");
    }
}

#[test]
fn subnormal_beta_shapes_preserve_the_endpoint_mass_ratio() {
    let unit = f64::from_bits(1);
    for x in [unit, 0.125, 0.5, 0.875, 1.0_f64.next_down()] {
        assert_eq!(x.beta_reg(unit, unit), 0.5);
        assert_eq!(x.beta_reg(unit, 3.0 * unit), 0.75);
        assert_eq!(x.beta_reg(3.0 * unit, unit), 0.25);
    }
    assert!(f64::NAN.beta_reg(unit, unit).is_nan());
}

#[test]
fn gamma_quantiles_resolve_the_upper_tail_of_an_exponential() {
    // Gamma(1, 1) has the exact upper quantile -ln(1-p).
    for p in [0.51, 0.99, 0.999_999, 1.0_f64.next_down()] {
        let expected = -(-p).ln_1p();
        assert!((p.gamma_lr_inv(1.0) / expected - 1.0).abs() < 1e-13);
    }
    for p in [0.51, 0.99, 0.999_999, 1.0_f32.next_down()] {
        let expected = -(-p).ln_1p();
        assert!((p.gamma_lr_inv(1.0) / expected - 1.0).abs() < 8.0 * f32::EPSILON);
    }
}

#[test]
fn gamma_medians_resolve_to_their_native_format() {
    // For this tiny f64 shape, P(a, x) is already greater than one-half
    // at half of the least subnormal. The median rounds to zero.
    assert_eq!(0.5_f64.gamma_lr_inv(6.446_704_688_105_846e-20), 0.0);
    // Near shape 2^46 the median differs from the mean by about a third.
    // The spacing below the mean is 2^22 in f32.
    let large = 2.0_f32.powi(46);
    assert_eq!(0.5_f32.gamma_lr_inv(large), large);
    // MPFR gamma upper bisection from the extreme-shape reference generator.
    let shape = 0.009_290_680_289_268_494_f64 as f32;
    let expected = 2.245_615_321_845_595_3e-33_f64 as f32;
    assert_eq!(0.5_f32.gamma_lr_inv(shape), expected);
    let expected_at_cutoff = 4.465_528_099_628_194e-31_f64 as f32;
    assert_eq!(0.5_f32.gamma_lr_inv(0.01), expected_at_cutoff);
}

#[test]
fn digamma_resolves_both_sides_of_a_remote_pole() {
    // At -n + d, the leading pole is -1/d and the remaining finite part is O(ln(n)).
    let pole = -10_000_000_000_000.0_f64;
    for x in [pole.next_down(), pole.next_up()] {
        let value = x.digamma();
        let residual = value + 1.0 / (x - pole);
        assert!((29.0..31.0).contains(&residual), "digamma({x}) = {value}");
    }
    let pole = -10_000.0_f32;
    for x in [pole.next_down(), pole.next_up()] {
        let value = x.digamma();
        let residual = value + 1.0 / (x - pole);
        assert!((9.0..10.0).contains(&residual), "digamma({x}) = {value}");
    }
}

#[test]
fn trigamma_resolves_both_sides_of_a_remote_pole() {
    // Isolate the pole term in the series for ψ₁. As d tends to zero at
    // a remote negative integer, the other terms approach 2ζ(2) = π²/3.
    let pole = -10_000_000_000_000.0_f64;
    for x in [pole.next_down(), pole.next_up()] {
        let offset = x - pole;
        let residual = x.trigamma() - (offset * offset).recip();
        assert!(
            (3.28..3.30).contains(&residual),
            "trigamma({x}) residual {residual}"
        );
    }
    let pole = -10_000.0_f32;
    for x in [pole.next_down(), pole.next_up()] {
        let offset = f64::from(x - pole);
        let residual = f64::from(x.trigamma()) - (offset * offset).recip();
        assert!(
            (3.0..3.5).contains(&residual),
            "trigamma({x}) residual {residual}"
        );
    }
}

#[test]
fn single_precision_digamma_resolves_its_positive_zero() {
    // References evaluated at the exact binary32 inputs by MPFR at 256 bits.
    // The native f32 recurrence previously returned the wrong sign at 0x3fbb_16c3.
    for (bits, expected) in [
        (0x3fbb_16c2, -1.273_915_810_519_784_6e-7),
        (0x3fbb_16c3, -1.203_605_254_910_667e-8),
        (0x3fbb_16c4, 1.033_194_633_696_787_1e-7),
    ] {
        let actual = f64::from(f32::from_bits(bits).digamma());
        assert!((actual - expected).abs() < 2e-14, "{actual} vs {expected}");
    }
}

#[test]
fn digamma_handles_poles_and_signed_zero() {
    for pole in [-1.0, -f64::MAX, f64::NEG_INFINITY] {
        assert!(pole.digamma().is_nan());
    }
    for pole in [-1.0, -f32::MAX, f32::NEG_INFINITY] {
        assert!(pole.digamma().is_nan());
    }
    assert_eq!(0.0_f64.digamma(), f64::NEG_INFINITY);
    assert_eq!((-0.0_f64).digamma(), f64::INFINITY);
    assert_eq!(0.0_f32.digamma(), f32::NEG_INFINITY);
    assert_eq!((-0.0_f32).digamma(), f32::INFINITY);
}
