//! `K₀`/`K₁`: the cosh-integral quadrature, the small-argument series, and the join between
//! them. The two cross-family checks at the end belong here rather than with `I`, because what
//! they constrain is `K`'s consistency *with* `I` rather than anything about `I` alone.

use crate::constants::f64::EULER_GAMMA;
use crate::kernels::bessel::modified_first::{i0e, i1e};
use crate::kernels::bessel::modified_second::*;

/// 600-digit `Decimal` references: `(x, K₀, K₁, K₀ᵉ, K₁ᵉ)`.
const K_REFS: &[(f64, f64, f64, f64, f64)] = &[
    (
        0.1,
        2.427_069_024_702_017e0,
        9.853_844_780_870_606e0,
        2.682_326_102_262_894_4e0,
        1.089_018_268_304_969_6e1,
    ),
    (
        0.5,
        9.244_190_712_276_659e-1,
        1.656_441_120_003_301e0,
        1.524_109_385_773_909_4e0,
        2.731_009_708_211_786e0,
    ),
    (
        1.0,
        4.210_244_382_407_083_4e-1,
        6.019_072_301_972_346e-1,
        1.144_463_079_806_895e0,
        1.636_153_486_263_258_3e0,
    ),
    (
        2.0,
        1.138_938_727_495_334_4e-1,
        1.398_658_818_165_224_3e-1,
        8.415_682_150_707_714e-1,
        1.033_476_847_068_688_6e0,
    ),
    (
        5.0,
        3.691_098_334_042_594_2e-3,
        4.044_613_445_452_165e-3,
        5.478_075_643_135_19e-1,
        6.002_738_587_883_126e-1,
    ),
    (
        10.0,
        1.778_006_231_616_765e-5,
        1.864_877_345_382_558_5e-5,
        3.916_319_344_365_986_6e-1,
        4.107_665_705_957_887_5e-1,
    ),
    (
        20.0,
        5.741_237_815_336_525e-10,
        5.883_057_969_557_038e-10,
        2.785_448_766_571_822_5e-1,
        2.854_254_969_407_264_4e-1,
    ),
    (
        50.0,
        3.410_167_749_789_495_6e-23,
        3.444_102_226_717_555_5e-23,
        1.768_071_558_574_293_4e-1,
        1.785_665_585_588_155_6e-1,
    ),
    (
        100.0,
        4.656_628_229_175_902e-45,
        4.679_853_735_636_909_5e-45,
        1.251_756_216_591_265_7e-1,
        1.257_999_504_795_785_4e-1,
    ),
];

/// Relative error, which is the right measure for `K`: the unscaled values span `1e0` down
/// to `1e-45`, so the mixed absolute/relative criterion used elsewhere would wave through
/// a completely wrong `K₀(100)`.
fn rel_close(got: f64, want: f64, tol: f64) {
    let err = ((got - want) / want).abs();
    assert!(
        err <= tol,
        "{got:e} vs {want:e} (rel err {err:e} > {tol:e})"
    );
}

#[test]
fn the_second_kind_matches_its_reference_values() {
    for &(x, k0_r, k1_r, k0e_r, k1e_r) in K_REFS {
        rel_close(k0(x), k0_r, 1e-14);
        rel_close(k1(x), k1_r, 1e-14);
        rel_close(k0e(x), k0e_r, 1e-14);
        rel_close(k1e(x), k1e_r, 1e-14);
    }
}

#[test]
fn the_modified_wronskian_ties_i_to_k() {
    // I0(x)K1(x) + I1(x)K0(x) = 1/x, written in the *scaled* forms so it is checkable at
    // any argument: the e^{-x} and e^{+x} factors cancel pairwise, where the unscaled
    // product would overflow times underflow past x = 710.
    for x in [
        0.01f64, 0.1, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 50.0, 100.0, 500.0, 1e3, 1e5,
    ] {
        let residual = i0e(x) * k1e(x) + i1e(x) * k0e(x) - 1.0 / x;
        // Judged *relative* to the Wronskian's own value of 1/x. The floor here is set by
        // the `I` family rather than by `K`: `i0e`/`i1e` carry about `1.7e-14` relative
        // error (measured, and documented on the trait method), which flows straight into
        // the products. `K` is an order better.
        assert!(
            (residual * x).abs() <= 1e-13,
            "I/K Wronskian broken at x = {x}: residual {residual:e}"
        );
    }
}

#[test]
fn the_second_kind_joins_its_two_regimes_smoothly() {
    // K0' = -K1 and K1' = -(K0 + K1/x), which bound a legitimate change across the join
    // between the power series and the quadrature.
    let x = K_SERIES_MAX;
    let h = 1e-9;
    for (name, f, derivative) in [
        ("K0", k0 as fn(f64) -> f64, -k1(x)),
        ("K1", k1 as fn(f64) -> f64, -(k0(x) + k1(x) / x)),
    ] {
        let (below, above) = (f(x - h), f(x + h));
        let implied = derivative.abs() * 2.0 * h;
        assert!(
            (above - below).abs() <= implied * 1.05 + 1e-16,
            "{name} steps at the join: {below:e} then {above:e}"
        );
    }
    // And the two routes agree throughout a band rather than only at the switch. The band
    // has to stay narrow because it is the *series* that decays, not the quadrature: the
    // `ln(x/2)·I₀` subtraction loses about one digit at x = 2, 1.6 by x = 2.5 and 3.1 by
    // x = 4, where its relative error reaches 1.4e-13. Measured disagreement across
    // (2, 2.5] is 1.2e-14, all of it the series'. That growth is exactly why
    // `K_SERIES_MAX` sits at 2 rather than higher.
    for i in 1..=10 {
        let x = K_SERIES_MAX + f64::from(i) * 0.05;
        rel_close(k_nu_scaled(0.0, x), k0_series(x) * x.exp(), 5e-14);
        rel_close(k_nu_scaled(1.0, x), k1_series(x) * x.exp(), 5e-14);
    }
}

#[test]
fn the_second_kind_satisfies_the_modified_bessel_equation() {
    // x^2 y'' + x y' - x^2 y = 0 for K0, by finite differences.
    for i in 2..80 {
        let x = f64::from(i) * 0.25;
        let h = 1e-4 * x;
        let second = (k0(x + h) - 2.0 * k0(x) + k0(x - h)) / (h * h);
        let first = (k0(x + h) - k0(x - h)) / (2.0 * h);
        let residual = x * x * second + x * first - x * x * k0(x);
        assert!(
            residual.abs() < 1e-5 * k0(x).max(1e-12) * x.max(1.0),
            "modified equation broken for K0 at {x}: {residual:e}"
        );
    }
}

#[test]
fn the_second_kind_decays_monotonically_and_dominates_k0_by_k1() {
    let mut previous_k0 = f64::INFINITY;
    let mut previous_k1 = f64::INFINITY;
    for i in 1..1000 {
        let x = f64::from(i) * 0.05;
        let (a, b) = (k0(x), k1(x));
        assert!(a > 0.0 && b > 0.0, "K must stay positive at {x}");
        assert!(a < previous_k0, "K0 must decrease at {x}");
        assert!(b < previous_k1, "K1 must decrease at {x}");
        // K1 > K0 for every positive argument, since K is increasing in its order.
        assert!(b > a, "K1 must exceed K0 at {x}");
        previous_k0 = a;
        previous_k1 = b;
    }
}

#[test]
fn the_scaled_second_kind_survives_where_the_unscaled_one_underflows() {
    // K0(746) underflows to zero, while K0e(746) is a perfectly ordinary 0.046.
    assert_eq!(
        k0(800.0),
        0.0,
        "the unscaled form underflows, as the true value does"
    );
    for &x in &[700.0f64, 746.0, 1e3, 1e5, 1e8, 1e300] {
        let (a, b) = (k0e(x), k1e(x));
        assert!(a > 0.0 && a.is_finite(), "k0e({x}) = {a}");
        assert!(b > 0.0 && b.is_finite(), "k1e({x}) = {b}");
        // Ke(x) ~ sqrt(pi/2x)·Sum a_k(nu)/x^k, with a_k(nu) the same coefficients the
        // Hankel expansion uses. Three terms are needed for this comparison to say
        // anything at 1e-9: truncating after two leaves 9/(128x^2), which is itself
        // 1.3e-7 at x = 746.
        let base = (core::f64::consts::PI / (2.0 * x)).sqrt();
        let inv = 1.0 / x;
        rel_close(a, base * (1.0 - inv / 8.0 + 9.0 * inv * inv / 128.0), 1e-9);
        rel_close(
            b,
            base * (1.0 + 3.0 * inv / 8.0 - 15.0 * inv * inv / 128.0),
            1e-9,
        );
    }
}

#[test]
fn the_second_kind_handles_its_poles_and_non_finite_inputs() {
    // A logarithmic singularity for K0 and a simple pole for K1.
    assert_eq!(k0(0.0), f64::INFINITY);
    assert_eq!(k1(0.0), f64::INFINITY);
    assert_eq!(k0e(0.0), f64::INFINITY);
    assert_eq!(k1e(0.0), f64::INFINITY);
    // Approaching the pole, K0 ~ -ln(x/2) - gamma and K1 ~ 1/x. The next correction to
    // K0 is O(x^2 ln x), so these start at 1e-8 where that is already below 1e-16 —
    // at 1e-4 it would be 3e-9 and the comparison would be measuring the truncated
    // reference rather than the implementation.
    for &x in &[1e-8f64, 1e-12, 1e-30, 1e-100, 1e-300] {
        rel_close(k0(x), -(x * 0.5).ln() - EULER_GAMMA, 1e-10);
        rel_close(k1(x), 1.0 / x, 1e-10);
    }
    assert_eq!(k0(f64::INFINITY), 0.0);
    assert_eq!(k1(f64::INFINITY), 0.0);
    assert!(k0(-1.0).is_nan(), "the real K is undefined below zero");
    assert!(k1(-1.0).is_nan());
    assert!(k0e(-1.0).is_nan());
    assert!(k0(f64::NAN).is_nan());
    assert!(k1(f64::NAN).is_nan());
    assert!(k0e(f64::NAN).is_nan());
    assert!(k1e(f64::NAN).is_nan());
}

/// `Iνᵉ` and `Kνᵉ` from their asymptotic expansions, summed to the minimum term. At large
/// `x` these are exact to ~`e^{-2x}`, so they are a sound independent reference.
fn scaled_asymptotic(nu: f64, x: f64, modified_first: bool) -> f64 {
    let four_nu_sq = 4.0 * nu * nu;
    let mut a = 1.0f64;
    let mut inv = 1.0f64;
    let mut total = 0.0f64;
    let mut prev = f64::INFINITY;
    for m in 0..200u32 {
        if m > 0 {
            let mf = f64::from(m);
            let odd = 2.0 * mf - 1.0;
            a *= (four_nu_sq - odd * odd) / (8.0 * mf);
            inv /= x;
        }
        let mag = (a * inv).abs();
        if mag >= prev {
            break;
        }
        prev = mag;
        // K uses a_k/x^k; I uses (-1)^k a_k/x^k.
        total += if modified_first && m % 2 == 1 {
            -a * inv
        } else {
            a * inv
        };
        if mag < 1e-22 {
            break;
        }
    }
    if modified_first {
        total / (core::f64::consts::TAU * x).sqrt()
    } else {
        total * (core::f64::consts::PI / (2.0 * x)).sqrt()
    }
}

#[test]
fn both_scaled_kinds_stay_machine_precision_at_large_argument() {
    // This is the test that caught the log-domain cancellation in `ln_peak`. Before the
    // peak core was rewritten to cancel symbolically, `i0e`/`i1e` carried
    //
    //     x = 1e2  8.7e-16     x = 1e4  7.8e-12
    //     x = 1e3  1.8e-14     x = 1e5  9.1e-11      x = 1e6  1.5e-9
    //
    // of relative error -- five orders worse than the surface's documented 1.7e-14 -- all
    // of it from subtracting two O(k ln k) logarithms. `K` was unaffected throughout,
    // which is what localised the fault to the first kind.
    for &x in &[100.0f64, 1e3, 1e4, 1e5, 1e6] {
        rel_close(i0e(x), scaled_asymptotic(0.0, x, true), 1e-14);
        rel_close(i1e(x), scaled_asymptotic(1.0, x, true), 1e-14);
        rel_close(k0e(x), scaled_asymptotic(0.0, x, false), 1e-14);
        rel_close(k1e(x), scaled_asymptotic(1.0, x, false), 1e-14);
    }
    // What remains for `I` at the very top of the range is the length of the sum itself,
    // not cancellation: `i0_rel_sum` walks `O(√x)` terms, which is 1e4 of them at x = 1e8,
    // and accumulates about `√n·ε ≈ 5e-14`. `K`'s quadrature is a fixed ~40 nodes and so
    // does not drift at all.
    rel_close(i0e(1e8), scaled_asymptotic(0.0, 1e8, true), 1e-13);
    rel_close(k0e(1e8), scaled_asymptotic(0.0, 1e8, false), 1e-15);
    rel_close(k1e(1e8), scaled_asymptotic(1.0, 1e8, false), 1e-15);
}

#[test]
fn the_series_quadrature_threshold_leaves_no_step_at_ulp_resolution() {
    // The series -> quadrature switch at x = 2, probed at the two representable values that
    // straddle it. K decays exponentially, so the comparison is scaled by the value itself.
    // The two routes agree to about 1e-14 relative here, all of it the series' `ln(x/2)·I₀`
    // subtraction; see `the_second_kind_joins_its_two_regimes_smoothly` for the measurement.
    use euclid_utils::no_step_across;
    let x = K_SERIES_MAX;
    no_step_across("K0", k0, x, -k1(x), 1e-13);
    no_step_across("K1", k1, x, -(k0(x) + k1(x) / x), 1e-13);
    no_step_across("K0e", k0e, x, k0e(x) - k1e(x), 1e-13);
    no_step_across("K1e", k1e, x, k1e(x) - k0e(x) - k1e(x) / x, 1e-13);
}
