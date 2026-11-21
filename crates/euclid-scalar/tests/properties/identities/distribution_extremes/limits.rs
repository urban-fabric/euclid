//! The generated laws, and the shape-domain guards.
//!
//! Part of the [`distribution_extremes`](super) suite. What these cover is the *edge of the
//! parameter domain* rather than the edge of the format: a shape that is zero, negative or
//! non-finite has no distribution, and both quantiles have to say so rather than returning a
//! root from a search that never had a bracket.

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
