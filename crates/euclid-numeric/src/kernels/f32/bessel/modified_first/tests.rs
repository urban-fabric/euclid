//! Unit tests for the single-precision `I₀`/`I₁`.
//!
//! A child module in its own file so implementation plus tests stay inside the crate's
//! 400-line budget, while `use super::*` still reaches the private peak-factored sums.

use super::*;

fn agrees(a: f32, b: f64, tol: f32) {
    let b = b as f32;
    assert!(
        (a - b).abs() <= tol * b.abs().max(1.0),
        "{a:e} vs {b:e} (tol {tol:e})"
    );
}

#[test]
fn the_scaled_forms_match_the_f64_reference_to_single_precision() {
    // The f32 path is now limited by f32 itself rather than by a 1e-7 approximation, so
    // it tracks the f64 implementation to a few ulp instead of coincidentally sharing its
    // error.
    for &(x, i0_ref, i1_ref) in &[
        (
            0.5f32,
            6.450_352_704_491_501e-1f64,
            1.564_208_031_848_717e-1f64,
        ),
        (1.0, 4.657_596_075_936_404e-1, 2.079_104_153_497_084e-1),
        (3.74, 2.147_726_486_128_186e-1, 1.831_416_223_205_881e-1),
        (3.76, 2.141_428_724_247_597e-1, 1.827_955_634_912_299e-1),
        (10.0, 1.278_333_371_634_286e-1, 1.212_626_813_844_555e-1),
        (50.0, 5.656_162_664_745_419e-2, 5.599_312_389_289_54e-2),
    ] {
        agrees(i0e(x), i0_ref, 1e-5);
        agrees(i1e(x), i1_ref, 1e-5);
    }
}

#[test]
fn large_arguments_stay_finite_in_f32() {
    for &x in &[50.0f32, 100.0, 1e3, 1e4] {
        let v = i0e(x);
        assert!(v > 0.0 && v.is_finite(), "i0e({x}) = {v}");
        let asym = 1.0 / (core::f32::consts::TAU * x).sqrt();
        assert!((v - asym).abs() < 1e-2 * asym);
    }
    // ln I0 stays finite far past where I0 is representable in f32.
    assert!(ln_i0(1e4f32).is_finite());
}

#[test]
fn the_ratio_is_bounded_and_monotone() {
    let mut prev = 0.0f32;
    for i in 1..400 {
        let x = (i as f32) * 0.5;
        let r = i1_over_i0(x);
        assert!((0.0..1.0).contains(&r), "i1_over_i0({x}) = {r}");
        assert!(r >= prev - 1e-6);
        prev = r;
    }
}

// --- the large-argument and subnormal regressions ------------------------

/// Relative error against the `f64` reference, which is machine precision here.
fn rel(got: f32, want: f64) -> f64 {
    ((f64::from(got) - want) / want).abs()
}

#[test]
fn the_scaled_forms_track_f64_at_every_magnitude() {
    // The regression test for the log-domain cancellation. Before `ln_peak_core_scaled`
    // existed, the naive peak exponent gave
    //
    //     x = 1e1  1.8e-6      x = 1e4  3.6e-3
    //     x = 1e2  3.1e-5      x = 1e5  3.5e-2      x = 1e6  4.7e-2  (i1e: 4.2e-1)
    //
    // of relative error — up to 42% on a surface documented to track `f64` within an ulp.
    // It survived because the existing parity test sweeps only to x = 50 and judges with a
    // mixed absolute/relative criterion, which forgives anything about a value as small as
    // `I₀ᵉ`. This test is relative, and it goes to 1e6.
    for &x in &[10.0f32, 100.0, 1e3, 1e4, 1e5, 1e6] {
        let wide = f64::from(x);
        let (i0_ref, i1_ref) = (
            crate::kernels::bessel::modified_first::i0e(wide),
            crate::kernels::bessel::modified_first::i1e(wide),
        );
        // What remains is the length of the sum itself: `i0_rel_sum` walks O(sqrt x)
        // terms, which is ~1000 of them at x = 1e6, accumulating about sqrt(n)*eps.
        assert!(
            rel(i0e(x), i0_ref) < 1e-5,
            "i0e({x:e}) rel {:e}",
            rel(i0e(x), i0_ref)
        );
        assert!(
            rel(i1e(x), i1_ref) < 1e-5,
            "i1e({x:e}) rel {:e}",
            rel(i1e(x), i1_ref)
        );
        // `ln I0` is the form that has to stay good when `I0` itself has long overflowed
        // f32 (which it does by x = 89).
        let ln_ref = crate::kernels::bessel::modified_first::ln_i0(wide);
        assert!(rel(ln_i0(x), ln_ref) < 1e-6, "ln_i0({x:e})");
    }
}

#[test]
fn subnormal_arguments_do_not_produce_nan() {
    // `2k·ln h` is `0 × −∞` once `h = x/2` underflows, which it does at the smallest f32
    // subnormal — and that took `I₀`, `I₀ᵉ` and `ln I₀` to NaN. Each has a perfectly
    // ordinary limit at the origin.
    for &x in &[1e-45f32, 1.4e-45, 1e-44, 1e-40, f32::MIN_POSITIVE] {
        assert_eq!(i0(x), 1.0, "I0({x:e}) -> 1");
        assert_eq!(i0e(x), 1.0, "I0e({x:e}) -> 1");
        assert_eq!(ln_i0(x), 0.0, "ln I0({x:e}) -> 0");
        assert!(!i1(x).is_nan(), "I1({x:e}) is NaN");
        assert!(i1(x) >= 0.0 && i1(x) <= x, "I1(x) ~ x/2 must lie in [0, x]");
        // The ratio is close to I1 here, since I0 is 1 — but not bit-identical, because
        // it is formed from the peak-relative sums directly rather than as a quotient.
        assert!(
            (i1_over_i0(x) - i1(x)).abs() <= 4.0 * f32::EPSILON * x,
            "the ratio should be I1 to a few ulps when I0 is 1"
        );
    }
}

#[test]
fn the_unscaled_forms_overflow_only_where_f32_itself_runs_out() {
    // I0 grows like e^x/sqrt(2 pi x), so it leaves f32's range near x = 89 — far earlier
    // than the f64 x = 714. The scaled forms and the log must carry on past it.
    // The boundary is near x = 91.5, where ln I0 = x - ln(2 pi x)/2 reaches f32's 88.7.
    assert!(
        i0(90.0).is_finite(),
        "ln I0(90) is 86.8, so I0(90) still fits in f32"
    );
    assert!(i0(95.0).is_infinite(), "by x = 95 it does not");
    for &x in &[92.0f32, 100.0, 1e4, 1e30, 1e38] {
        assert!(i0e(x) > 0.0 && i0e(x).is_finite(), "i0e({x:e})");
        assert!(ln_i0(x).is_finite(), "ln_i0({x:e})");
        // The ratio tends to 1 from below as 1 - 1/(2x); past x = 1/(2 eps) = 4e6 that
        // is no longer distinguishable from 1 in f32, so saturating to exactly 1.0 is
        // the correctly-rounded answer rather than an escape from the open interval.
        let ratio = i1_over_i0(x);
        assert!(
            (0.0..=1.0).contains(&ratio),
            "the ratio left [0, 1] at {x:e}"
        );
    }
}
