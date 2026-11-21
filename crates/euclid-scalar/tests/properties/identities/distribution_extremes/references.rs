//! Extreme arguments against independently computed references.
//!
//! Part of the [`distribution_extremes`](super) suite. These are the cases where no identity
//! constrains the answer and a pasted high-precision value is the only instrument — so each one
//! carries its provenance in prose, per the contributing guide's rule for fixtures whose
//! generator does not ship.

use super::*;

/// The single-precision inverse lands within a measured ulp budget of the best `f32` root.
///
/// # This used to assert correct rounding, and no longer can
///
/// The previous form evaluated `erfc` at the midpoints either side of the returned `f32` and
/// required them to bracket `y` — i.e. that the result is the nearest representable root. That
/// held because `erfc_inv` widened its argument to `f64`, solved there, and rounded once.
///
/// It is now native; see [`euclid_scalar`]'s `kernels::f32::normal` for why, and the project
/// direction it follows. Correct rounding is **structurally unreachable** on a native path, and
/// not by a little: the last operation multiplies by `1/√2`, which is irrational and so already
/// inexact as an `f32` constant, so one rounding against an inexact multiplier spends most of an
/// ulp before `ndtri`'s own error is counted. Measured, the three inputs below land 2, 1 and 0
/// ulp from the best `f32` value.
///
/// So the assertion is the weaker achievable one. It still covers what this regression was
/// written for: the downstream-reported input returned a root wrong in its leading digits, which
/// a two-ulp budget catches as easily as correct rounding did. What is genuinely given up is the
/// ability to detect a *one-ulp* drift here, and that is the price of the native path rather than
/// an oversight.
#[test]
fn single_precision_erfc_inverse_lands_within_its_ulp_budget() {
    // Include the input found by Bayes and the two smallest positive f32 probabilities.
    for y in [
        f32::from_bits(0x3e84_5270),
        f32::from_bits(1),
        f32::from_bits(2),
    ] {
        let root = y.erfc_inv();
        assert!(root.is_finite(), "erfc_inv({y:e}) = {root}");
        // The `f64` kernel rounded to `f32` is the best a narrow kernel could return.
        let best = Real::erfc_inv(f64::from(y)) as f32;
        let gap = i64::from(root.to_bits()) - i64::from(best.to_bits());
        assert!(
            gap.abs() <= 2,
            "erfc_inv({y:e}) = {root:e}, best f32 root is {best:e} ({} ulp apart)",
            gap.abs()
        );
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
    // From `extreme_shapes`, a local generator over MPFR at 256 bits, self-validated against
    // closed forms before emitting. Not committed — see CONTRIBUTING.md, "Reference values are
    // pasted in", for why the precision is recorded here rather than a path to the script.
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
    // MPFR log gamma at 256 bits, cross-checked against the finite integer-shape sum evaluated
    // in `Decimal` at 450 digits, both in the local `extreme_shapes` generator.
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
    // From `extreme_shapes`, a local generator over MPFR at 256 bits, evaluated at the exact
    // subnormal doubles below rather than at their decimal renderings.
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
    //
    // Within a measured ulp budget rather than exactly, because `gamma_lr_inv` is native `f32`
    // here and the small-shape inversion has condition number `1/a` — about 108 at this shape.
    // See `kernels::f32::quantile::gamma_lr_inv`: a CDF good to one ulp cannot yield a root
    // better than a hundred, so exact equality was only ever achievable by computing in `f64`.
    // The budget is 160 against a measured worst of 115 across `a ∈ [1e-3, 100]`.
    let shape = 0.009_290_680_289_268_494_f64 as f32;
    let expected = 2.245_615_321_845_595_3e-33_f64 as f32;
    close_in_ulps(
        0.5_f32.gamma_lr_inv(shape),
        expected,
        160,
        "median at tiny shape",
    );
    let expected_at_cutoff = 4.465_528_099_628_194e-31_f64 as f32;
    close_in_ulps(
        0.5_f32.gamma_lr_inv(0.01),
        expected_at_cutoff,
        160,
        "median at the old widening cutoff",
    );
}
