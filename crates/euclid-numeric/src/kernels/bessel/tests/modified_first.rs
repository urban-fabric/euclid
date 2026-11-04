//! `I₀`/`I₁`: the peak-factored series, its scaled forms, the ratio, and `ln I₀`.

use crate::kernels::bessel::asymptotic::i_scaled;
use crate::kernels::bessel::modified_first::{I_ASYMPTOTIC_MIN, I1_NO_LOG_MAX, *};

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(1.0),
        "{a:e} vs {b:e} (tol {tol:e})"
    );
}

/// 60-digit `Decimal` reference values for `e^{-x} I0(x)` and `e^{-x} I1(x)`.
const SCALED: &[(f64, f64, f64)] = &[
    (0.0, 1.0, 0.0),
    (0.5, 6.450_352_704_491_501e-1, 1.564_208_031_848_717e-1),
    (1.0, 4.657_596_075_936_404e-1, 2.079_104_153_497_084e-1),
    (2.0, 3.085_083_225_536_71e-1, 2.152_692_892_489_376e-1),
    (3.74, 2.147_726_486_128_186e-1, 1.831_416_223_205_881e-1),
    (3.76, 2.141_428_724_247_597e-1, 1.827_955_634_912_299e-1),
    (5.0, 1.835_408_126_093_284e-1, 1.639_722_669_445_424e-1),
    (10.0, 1.278_333_371_634_286e-1, 1.212_626_813_844_555e-1),
    (20.0, 8.978_031_188_482_602e-2, 8.750_622_218_328_867e-2),
    (22.1, 8.535_495_638_436_524e-2, 8.340_093_783_969_195e-2),
    (50.0, 5.656_162_664_745_419e-2, 5.599_312_389_289_54e-2),
    (100.0, 3.994_437_929_909_668e-2, 3.974_415_302_513_025e-2),
];

#[test]
fn the_scaled_forms_now_reach_machine_precision() {
    // The point of the rewrite: these were accurate to ~1e-7 before.
    for &(x, i0_ref, i1_ref) in SCALED {
        close(i0e(x), i0_ref, 1e-13);
        close(i1e(x), i1_ref, 1e-13);
    }
}

#[test]
fn there_is_no_longer_a_branch_point_at_3_75() {
    // The old implementation switched formula there and the two sides disagreed. There is
    // no branch now, so the function is smooth across it.
    let step = 1e-8;
    assert!(
        (i0e(3.75 + step) - i0e(3.75 - step)).abs() < 1e-8,
        "join discontinuity at 3.75"
    );
}

#[test]
fn large_arguments_stay_finite_and_accurate() {
    for &x in &[700.0f64, 1e3, 1e4, 1e5] {
        let v = i0e(x);
        assert!(v > 0.0 && v.is_finite(), "i0e({x}) = {v}");
        // I0e(x) ~ (1 + 1/(8x) + 9/(128x^2)) / sqrt(2 pi x)
        let asym =
            (1.0 + 1.0 / (8.0 * x) + 9.0 / (128.0 * x * x)) / (core::f64::consts::TAU * x).sqrt();
        close(v, asym, 1e-8);
    }
}

#[test]
fn unscaled_forms_match_the_scaled_ones() {
    for &(x, _, _) in SCALED.iter().take(8) {
        close(i0(x), i0e(x) * x.exp(), 1e-13);
        if x > 0.0 {
            close(i1(x), i1e(x) * x.exp(), 1e-13);
        }
    }
}

#[test]
fn ln_i0_is_consistent_and_overflow_free() {
    for &(x, i0_ref, _) in SCALED {
        close(ln_i0(x), i0_ref.ln() + x, 1e-13);
    }
    // Far past where I0 itself is representable.
    for &x in &[800.0f64, 1e4, 1e6] {
        let v = ln_i0(x);
        assert!(v.is_finite() && v > 0.0, "ln_i0({x}) = {v}");
        close(v, x - 0.5 * (core::f64::consts::TAU * x).ln(), 1e-6);
    }
}

#[test]
fn the_ratio_is_bounded_monotone_and_odd() {
    assert_eq!(i1_over_i0(0.0), 0.0);
    let mut prev = 0.0f64;
    for i in 1..2000 {
        let x = f64::from(i) * 0.5;
        let r = i1_over_i0(x);
        assert!((0.0..1.0).contains(&r), "i1_over_i0({x}) = {r}");
        assert!(r >= prev, "the ratio must increase: {prev} then {r}");
        prev = r;
    }
    assert!(i1_over_i0(1e5) > 0.99999);
    close(i1_over_i0(-3.0), -i1_over_i0(3.0), 1e-15);
    for &(x, _, _) in SCALED.iter().skip(1) {
        close(i1_over_i0(x), i1e(x) / i0e(x), 1e-13);
    }
}

#[test]
fn the_derivative_relation_holds() {
    // d/dx ln I0(x) = I1(x)/I0(x). Now that both sides are accurate to ~1e-14 rather
    // than ~1e-7, this can be checked at a far tighter tolerance than before.
    for i in 1..40 {
        let x = f64::from(i) * 0.5;
        let h = 1e-5 * x;
        let d = (ln_i0(x + h) - ln_i0(x - h)) / (2.0 * h);
        close(d, i1_over_i0(x), 1e-9);
    }
}

#[test]
fn parity_and_non_finite_inputs() {
    for &(x, _, _) in SCALED.iter().skip(1) {
        close(i0e(-x), i0e(x), 1e-15);
        close(i1e(-x), -i1e(x), 1e-15);
    }
    assert!(i0(f64::NAN).is_nan());
    assert!(i1(f64::NAN).is_nan());
    assert_eq!(i0e(f64::INFINITY), 0.0);
    assert_eq!(i1e(f64::INFINITY), 0.0);
    assert_eq!(ln_i0(f64::INFINITY), f64::INFINITY);
}

#[test]
fn the_stirling_threshold_leaves_no_step_at_ulp_resolution() {
    // `stirling_remainder` switches from `ln_gamma` to its asymptotic series at a peak index
    // of 20. The peak index is `round(x/2)`, so the switch happens as `x` crosses 39 — an
    // internal boundary with no outward sign, and one no reference table would ever probe.
    // Measured disagreement across the switch is 1.5e-14 relative, and it is the *lower*
    // side that carries it: below the threshold `mu(k)` is formed as a difference of two
    // O(k ln k) quantities, costing about `eps*42/0.0042 = 1.5e-14`, while above it the
    // asymptotic series is exact to rounding. That is the documented 1.7e-14 worst case for
    // this family, and it is bounded here rather than merely asserted elsewhere.
    use euclid_utils::no_step_across;
    let threshold = 39.0;
    no_step_across(
        "I0e",
        i0e,
        threshold,
        i0e(threshold) - i1e(threshold),
        1e-13,
    );
    no_step_across("I1e", i1e, threshold, i1e(threshold), 1e-13);
    no_step_across("ln_i0", ln_i0, threshold, i1_over_i0(threshold), 1e-13);
    no_step_across("i1_over_i0", i1_over_i0, threshold, 1.0, 1e-13);
    // The peak index also steps by one every time x crosses an odd integer, since it is a
    // rounding of x/2. Those are threshold crossings too, and there are many of them.
    for odd in [5.0f64, 11.0, 21.0, 41.0, 101.0] {
        no_step_across("I0e at a peak step", i0e, odd, i0e(odd) - i1e(odd), 1e-13);
        no_step_across("ln_i0 at a peak step", ln_i0, odd, i1_over_i0(odd), 1e-13);
    }
}

#[test]
fn the_first_order_is_exact_in_the_small_argument_limit() {
    // I1(x) = (x/2)(1 + x^2/8 + ...), so I1/(x/2) - 1 should *be* x^2/8 and nothing else.
    //
    // Before `i1_no_log` existed the general route went through `exp(ln(x/2) + ...)`, and the
    // residual was the round-trip error rather than the mathematics:
    //
    //     x = 1e-10  1.3e-15      x = 1e-150  1.0e-14
    //     x = 1e-50  6.7e-15      x = 1e-300  3.5e-14
    //
    // all of it `eps * |ln(x/2)|` converted from absolute error in the log into relative
    // error by the exp. Now the residual is exactly the next series term, and exactly zero
    // once that term falls below the rounding of 1.
    // Compared against the series *directly* rather than by forming `ratio - 1`. That
    // subtraction is itself a cancellation — it resolves the residual only to `eps/residual`,
    // which is 9e-6 relative by x = 1e-5 — so a test written that way would be reporting its
    // own arithmetic. Three terms of `I1 = (x/2)(1 + x^2/8 + x^4/192)` are good to `x^6/9216`.
    for &x in &[1e-2f64, 1e-4, 1e-5] {
        let series = (x * 0.5) * (1.0 + x * x / 8.0 + x.powi(4) / 192.0);
        assert!(
            ((i1(x) - series) / series).abs() < 1e-15,
            "I1({x:e}) = {:e} should match its series {series:e}",
            i1(x)
        );
    }
    for &x in &[1e-10f64, 1e-50, 1e-150, 1e-300, f64::MIN_POSITIVE] {
        assert_eq!(
            i1(x) / (x * 0.5),
            1.0,
            "I1({x:e}) should be exactly x/2 once x^2/8 is below rounding"
        );
        // The *scaled* form is not x/2 but (x/2)e^{-x}, and at x = 1e-10 that factor is a
        // perfectly visible 1 - 1e-10. Exactness therefore only applies once e^{-x} itself
        // rounds to 1, which happens below eps/2.
        if x < f64::EPSILON * 0.5 {
            assert_eq!(
                i1e(x) / (x * 0.5),
                1.0,
                "I1e({x:e}) should be exactly x/2 where e^-x is 1"
            );
        } else {
            let scaled = (x * 0.5) * (-x).exp();
            assert!(
                ((i1e(x) - scaled) / scaled).abs() <= 2.0 * f64::EPSILON,
                "I1e({x:e}) should be (x/2)e^-x"
            );
        }
        // Oddness must survive the new branch.
        assert_eq!(i1(-x), -i1(x));
        assert_eq!(i1e(-x), -i1e(x));
    }
    // The new branch meets the logarithmic one without a step.
    use euclid_utils::no_step_across;
    no_step_across("i1", i1, I1_NO_LOG_MAX, i0(I1_NO_LOG_MAX), 1e-15);
    no_step_across("i1e", i1e, I1_NO_LOG_MAX, 1.0, 1e-15);
}

#[test]
fn the_scaled_forms_hold_all_the_way_to_the_top_of_the_format() {
    // The asymptotic branch exists because the series could not reach here. Measured against
    // the expansion before it was added, `I₀ᵉ` carried 4.9e-14 at x = 1e8 from drift over
    // 20,000 terms, and then lost its value entirely as the walk hit its cap:
    //
    //     x = 1e9   2.5e-10        x = 1e11   5.3e-1
    //     x = 1e10  4.5e-2         x = 1e12   8.4e-1
    //
    // — finite, silent, and wrong by half.
    //
    // The leading term is the whole answer at these magnitudes, since the first correction
    // `1/(8x)` is already below half an ulp past x = 1e17. So `1/√(2πx)` is an exact reference
    // rather than an approximation, and it is also the form that proves the square root was
    // split: `2πx` overflows past 2.9e307 while `√(2π)·√x` does not.
    use crate::constants::f64::SQRT_TAU;
    for &x in &[1e17f64, 1e30, 1e100, 1e300, 2.9e307, 1e308, f64::MAX] {
        let expected = 1.0 / (SQRT_TAU * x.sqrt());
        let got = i0e(x);
        assert!(
            got > 0.0 && ((got - expected) / expected).abs() <= 2.0 * f64::EPSILON,
            "i0e({x:e}) = {got:e}, expected {expected:e}"
        );
        let got_one = i1e(x);
        assert!(
            got_one > 0.0 && ((got_one - expected) / expected).abs() <= 2.0 * f64::EPSILON,
            "i1e({x:e}) = {got_one:e}, expected {expected:e}"
        );
        // I1/I0 -> 1 from below, and `ln I0 -> x` once the -ln(2 pi x)/2 correction falls
        // below x's own last bit.
        assert!(
            (0.0..=1.0).contains(&i1_over_i0(x)),
            "the ratio left [0, 1] at {x:e}"
        );
        assert!(ln_i0(x).is_finite() && ln_i0(x) > 0.0, "ln_i0({x:e})");
    }
}

#[test]
fn the_series_and_the_asymptotic_branch_agree_where_they_meet() {
    // The two routes share no code — a peak-factored power series against a divergent
    // asymptotic expansion — so agreement across the threshold is real evidence. Probed at
    // the two representable values straddling it, where no derivative slack is needed.
    use euclid_utils::no_step_across;
    let x = I_ASYMPTOTIC_MIN;
    no_step_across("i0e", i0e, x, i0e(x) - i1e(x), 1e-14);
    no_step_across("i1e", i1e, x, i1e(x), 1e-14);
    no_step_across("ln_i0", ln_i0, x, i1_over_i0(x), 1e-15);
    no_step_across("i1_over_i0", i1_over_i0, x, 1.0, 1e-14);
    // And across a whole band below the threshold, where the series is still the one in use
    // and must already be converging on the expansion.
    for i in 0..40 {
        let x = 3e3 + f64::from(i) * 175.0;
        let expected = i_scaled(0.0, x);
        assert!(
            ((i0e(x) - expected) / expected).abs() < 5e-15,
            "series i0e({x:e}) vs the expansion"
        );
    }
}
