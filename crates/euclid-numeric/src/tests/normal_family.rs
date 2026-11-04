//! The standard normal family (`ndtr`, `log_ndtr`, `ndtri`, `erfc_inv`) and the elementary
//! additions alongside it (`log10`, `exp2`, and the exact integer operations).
//!
//! The normal CDF is the most-used function in applied statistics and the one whose naive
//! composition fails soonest, so it gets the same four-way treatment as the rest: reference
//! values, tail stability, round-trips, and edges into the relationship web.

// Reference digits are written out in full as provenance; see the note in `accuracy`.
#![allow(clippy::excessive_precision)]

use crate::{Integer, Natural, Real};
use euclid_utils::{
    assert_close, assert_close_f32, assert_exact, assert_nondecreasing, assert_probability,
    derivative, integrate, log_sweep, rel, sweep,
};

// --- ndtr: reference values and shape -----------------------------------

#[test]
fn ndtr_matches_reference_values() {
    // Reference: Python's statistics.NormalDist().cdf, an independent implementation.
    let table = [
        (-3.0f64, 1.349_898_031_630_095e-3),
        (-1.959_963_984_540_054, 0.025),
        (-1.0, 0.158_655_253_931_457_05),
        (0.0, 0.5),
        (1.0, 0.841_344_746_068_542_9),
        (1.959_963_984_540_054, 0.975),
        (3.0, 0.998_650_101_968_369_9),
        (5.0, 0.999_999_713_348_428_2),
    ];
    for (x, want) in table {
        assert_close(Real::ndtr(x), want, 1e-12, &format!("ndtr({x})"));
    }
}

#[test]
fn ndtr_is_a_cdf() {
    let xs: Vec<f64> = sweep(-10.0, 10.0, 400).collect();
    let values: Vec<f64> = xs.iter().map(|&x| Real::ndtr(x)).collect();
    assert_probability(values.iter().copied(), "ndtr stays in [0,1]");
    assert_nondecreasing(values.iter().copied(), 0.0, "ndtr is monotone");
    assert_exact(Real::ndtr(f64::NEG_INFINITY), 0.0, "ndtr(-inf) = 0");
    assert_exact(Real::ndtr(f64::INFINITY), 1.0, "ndtr(+inf) = 1");
    assert!(Real::ndtr(f64::NAN).is_nan());
}

#[test]
fn ndtr_is_symmetric_about_a_half() {
    // Phi(-x) = 1 - Phi(x).
    for x in sweep(0.0, 8.0, 80) {
        assert_close(
            Real::ndtr(-x),
            1.0 - Real::ndtr(x),
            1e-12,
            "Phi(-x) = 1 - Phi(x)",
        );
    }
}

// --- log_ndtr: the reason it exists -------------------------------------

#[test]
fn log_ndtr_stays_finite_where_the_naive_composition_dies() {
    // Phi underflows to exactly zero around x = -38, so ln(ndtr(x)) is -inf from there on
    // while the true value is a perfectly ordinary number.
    assert_exact(Real::ndtr(-40.0f64), 0.0, "Phi underflows");
    assert!(Real::ndtr(-40.0f64).ln().is_infinite(), "so its log dies");

    for x in [-40.0f64, -100.0, -1e3, -1e6, -1e8] {
        let v = Real::log_ndtr(x);
        assert!(v.is_finite() && v < 0.0, "log_ndtr({x}) = {v}");
        // ln Phi(x) ~ -x^2/2 - ln|x| - ln sqrt(2 pi) in the far left tail.
        let asym = -0.5 * x * x - (-x).ln() - <f64 as Real>::half_ln_tau();
        assert!(
            rel(v, asym) < 1e-6,
            "log_ndtr({x}) = {v}, asymptotic {asym}"
        );
    }
}

#[test]
fn log_ndtr_agrees_with_the_direct_log_where_both_work() {
    for x in sweep(-30.0, 8.0, 200) {
        let direct = Real::ndtr(x).ln();
        if !direct.is_finite() {
            continue;
        }
        assert_close(Real::log_ndtr(x), direct, 1e-11, &format!("log_ndtr({x})"));
    }
}

#[test]
fn log_ndtr_is_monotone_and_crosses_its_branch_seamlessly() {
    let values: Vec<f64> = sweep(-50.0, 10.0, 600).map(Real::log_ndtr).collect();
    assert_nondecreasing(values.iter().copied(), 0.0, "log_ndtr is monotone");
    // The implementation switches formula at x = -1; a mismatched join shows up as a jump.
    let left = Real::log_ndtr(-1.0f64 - 1e-9);
    let right = Real::log_ndtr(-1.0f64 + 1e-9);
    assert!(
        (left - right).abs() < 1e-8,
        "discontinuity at the branch: {left} vs {right}"
    );
    assert!(Real::log_ndtr(0.0f64) < 0.0);
    assert_close(
        Real::log_ndtr(0.0f64),
        -<f64 as Real>::ln_2(),
        1e-14,
        "ln Phi(0) = -ln 2",
    );
}

#[test]
fn a_truncated_normal_log_likelihood_stays_finite() {
    // The motivating use. A censored observation contributes ln Phi((c - mu)/sigma); built
    // from ln(ndtr(..)) it becomes -inf for any observation far enough out, silently turning
    // the whole log-likelihood into -inf.
    let (mu, sigma) = (0.0f64, 1.0f64);
    let censoring_points = [-5.0f64, -20.0, -50.0, -200.0];
    let total: f64 = censoring_points
        .iter()
        .map(|&c| Real::log_ndtr((c - mu) / sigma))
        .sum();
    assert!(total.is_finite(), "log-likelihood = {total}");
    // The naive version is -inf as soon as any point is past about -38.
    let naive: f64 = censoring_points
        .iter()
        .map(|&c| Real::ndtr((c - mu) / sigma).ln())
        .sum();
    assert!(
        naive.is_infinite(),
        "the naive form is expected to collapse, got {naive}"
    );
}

// --- ndtri: the quantile ------------------------------------------------

#[test]
fn ndtri_matches_reference_values() {
    let table = [
        (0.025f64, -1.959_963_984_540_054),
        (0.05, -1.644_853_626_951_472_7),
        (0.1, -1.281_551_565_544_600_5),
        (0.5, 0.0),
        (0.9, 1.281_551_565_544_600_5),
        (0.975, 1.959_963_984_540_054),
        (0.995, 2.575_829_303_548_900_7),
    ];
    for (p, want) in table {
        assert_close(Real::ndtri(p), want, 1e-11, &format!("ndtri({p})"));
    }
}

#[test]
fn ndtri_round_trips_through_ndtr() {
    for p in sweep(0.001, 0.999, 500) {
        assert_close(
            Real::ndtr(Real::ndtri(p)),
            p,
            1e-11,
            &format!("ndtr(ndtri({p}))"),
        );
    }
}

#[test]
fn ndtri_round_trips_deep_in_the_tail_through_the_log_cdf() {
    // Below about p = 1e-16 the probability itself is the only usable handle, so the
    // round-trip has to be checked in the log domain.
    for p in log_sweep(1e-300, 1e-16, 60) {
        let x = Real::ndtri(p);
        assert!(x.is_finite() && x < 0.0, "ndtri({p}) = {x}");
        assert_close(
            Real::log_ndtr(x),
            p.ln(),
            1e-10,
            &format!("tail round trip at {p}"),
        );
    }
}

#[test]
fn ndtri_is_monotone_antisymmetric_and_handles_its_endpoints() {
    let values: Vec<f64> = sweep(1e-6, 1.0 - 1e-6, 500).map(Real::ndtri).collect();
    assert_nondecreasing(values.iter().copied(), 0.0, "ndtri is monotone");
    for p in sweep(0.01, 0.49, 40) {
        assert_close(
            Real::ndtri(1.0 - p),
            -Real::ndtri(p),
            1e-10,
            "ndtri is antisymmetric",
        );
    }
    assert_exact(Real::ndtri(0.5f64), 0.0, "ndtri(1/2) = 0");
    assert_exact(Real::ndtri(0.0f64), f64::NEG_INFINITY, "ndtri(0) = -inf");
    assert_exact(Real::ndtri(1.0f64), f64::INFINITY, "ndtri(1) = +inf");
    assert!(Real::ndtri(1.5f64).is_nan(), "out of domain");
    assert!(Real::ndtri(-0.5f64).is_nan(), "out of domain");
    assert!(Real::ndtri(f64::NAN).is_nan());
}

// --- erfc_inv -----------------------------------------------------------

#[test]
fn erfc_inv_inverts_erfc_across_its_domain() {
    for y in sweep(0.01, 1.99, 200) {
        assert_close(
            Real::erfc(Real::erfc_inv(y)),
            y,
            1e-11,
            &format!("erfc_inv({y})"),
        );
    }
    assert_exact(Real::erfc_inv(1.0f64), 0.0, "erfc_inv(1) = 0");
    assert_exact(Real::erfc_inv(0.0f64), f64::INFINITY, "erfc_inv(0) = +inf");
    assert_exact(
        Real::erfc_inv(2.0f64),
        f64::NEG_INFINITY,
        "erfc_inv(2) = -inf",
    );
    assert!(Real::erfc_inv(2.5f64).is_nan());
    assert!(Real::erfc_inv(-0.1f64).is_nan());
}

#[test]
fn erfc_inv_reaches_tails_that_erf_inv_cannot() {
    // `erf_inv(1 - y)` is the obvious composition and it is useless for small y: 1 - y rounds
    // to 1 and erf_inv returns +inf.
    let y = 1e-20f64;
    assert_exact(1.0 - y, 1.0, "1 - y rounds away");
    assert!(
        Real::erf_inv(1.0f64 - y).is_infinite(),
        "so erf_inv gives inf"
    );
    let x = Real::erfc_inv(y);
    assert!(x.is_finite() && x > 0.0, "erfc_inv({y}) = {x}");
    // Verify through the scaled form, since erfc(x) itself has underflowed.
    assert_close(
        Real::erfcx(x).ln() - x * x,
        y.ln(),
        1e-9,
        "erfc_inv tail check via erfcx",
    );
    for y in log_sweep(1e-280, 1e-20, 40) {
        let x = Real::erfc_inv(y);
        assert!(x.is_finite() && x > 0.0, "erfc_inv({y}) = {x}");
        assert_close(Real::erfcx(x).ln() - x * x, y.ln(), 1e-9, "erfc_inv tail");
    }
}

// --- relationship-web edges for the new functions -----------------------

#[test]
fn ndtr_differentiates_to_the_normal_density() {
    // Phi'(x) = phi(x). Ties ndtr to exp and half_ln_tau.
    for x in sweep(-5.0, 5.0, 60) {
        let want = (-0.5 * x * x - <f64 as Real>::half_ln_tau()).exp();
        assert_close(
            derivative(Real::ndtr, x),
            want,
            1e-8,
            &format!("ndtr'({x})"),
        );
    }
}

#[test]
fn log_ndtr_differentiates_to_the_inverse_mills_ratio() {
    // d/dx ln Phi(x) = phi(x)/Phi(x). Ties log_ndtr to both ndtr and the density.
    for x in sweep(-6.0, 4.0, 60) {
        let ln_pdf = -0.5 * x * x - <f64 as Real>::half_ln_tau();
        let want = (ln_pdf - Real::log_ndtr(x)).exp();
        // As with `ln_bessel_i0`, differentiation amplifies the underlying function's error
        // by 1/h: `log_ndtr` runs through `erfcx` in the left tail, and the step is ~5e-4, so
        // an erfcx error of 1e-9 becomes microscopically visible here.
        assert_close(
            derivative(Real::log_ndtr, x),
            want,
            1e-5,
            &format!("log_ndtr'({x})"),
        );
    }
}

#[test]
fn the_normal_density_integrates_to_ndtr() {
    // Quadrature of the density reproduces the CDF, pinning the normalizer independently.
    let pdf = |t: f64| (-0.5 * t * t - <f64 as Real>::half_ln_tau()).exp();
    for x in sweep(-4.0, 4.0, 20) {
        let area = integrate(pdf, -40.0, x, 20_000);
        assert_close(area, Real::ndtr(x), 1e-10, &format!("int phi to {x}"));
    }
}

#[test]
fn ndtr_agrees_with_the_erf_and_incomplete_gamma_routes() {
    // Three independent routes to the same quantity, closing a triangle with the existing
    // web: ndtr, the erf form, and the chi-squared-with-1-degree-of-freedom form.
    for x in sweep(0.05, 6.0, 80) {
        let via_erf = 0.5 * (1.0 + Real::erf(x * <f64 as Real>::frac_1_sqrt_2()));
        let via_gamma = 0.5 * (1.0 + Real::gamma_lr(x * x / 2.0, 0.5));
        assert_close(
            Real::ndtr(x),
            via_erf,
            1e-12,
            &format!("ndtr vs erf at {x}"),
        );
        assert_close(
            Real::ndtr(x),
            via_gamma,
            1e-11,
            &format!("ndtr vs gamma at {x}"),
        );
    }
}

#[test]
fn ndtri_and_erfc_inv_are_consistent() {
    // erfc(x) = 2 Phi(-x sqrt 2), so erfc_inv(y) = -ndtri(y/2)/sqrt 2.
    for y in sweep(0.01, 1.9, 60) {
        assert_close(
            Real::erfc_inv(y),
            -Real::ndtri(0.5 * y) * <f64 as Real>::frac_1_sqrt_2(),
            1e-12,
            "erfc_inv via ndtri",
        );
    }
    // And erf_inv agrees where it is usable.
    for p in sweep(0.02, 0.98, 60) {
        assert_close(
            Real::ndtri(p),
            Real::erf_inv(2.0 * p - 1.0) / <f64 as Real>::frac_1_sqrt_2(),
            1e-10,
            "ndtri via erf_inv",
        );
    }
}

// --- log10 and exp2 -----------------------------------------------------

#[test]
fn log10_and_exp2_are_exact_at_their_powers() {
    for k in -300..301 {
        let p = f64::from(k);
        // log10 is exact at exact powers of ten within range.
        if (-300..=300).contains(&k) {
            assert_exact(Real::log10(10f64.powi(k)), p, &format!("log10(1e{k})"));
        }
    }
    for k in -1000..1001 {
        let p = f64::from(k);
        assert_exact(Real::exp2(p), 2f64.powi(k), &format!("exp2({k})"));
        assert_exact(Real::log2(Real::exp2(p)), p, &format!("log2(exp2({k}))"));
    }
}

#[test]
fn log10_and_exp2_relate_to_the_natural_forms() {
    for x in sweep(0.1, 100.0, 200) {
        assert_close(
            Real::log10(x),
            Real::ln(&x) * <f64 as Real>::log10_e(),
            1e-14,
            &format!("log10({x})"),
        );
        assert_close(
            Real::log10(x) * <f64 as Real>::log2_10(),
            Real::log2(x),
            1e-14,
            &format!("log10 to log2 at {x}"),
        );
    }
    for x in sweep(-20.0, 20.0, 200) {
        assert_close(
            Real::exp2(x),
            Real::exp(&(x * <f64 as Real>::ln_2())),
            1e-14,
            &format!("exp2({x})"),
        );
    }
    // Round trip both ways.
    for x in sweep(0.01, 1000.0, 100) {
        assert_close(Real::exp2(Real::log2(x)), x, 1e-14, "exp2(log2(x)) = x");
    }
}

// --- exact integer arithmetic -------------------------------------------

#[test]
fn gcd_satisfies_its_defining_properties() {
    for a in 0..60u64 {
        for b in 0..60u64 {
            let g = Natural::gcd(a, b);
            if a == 0 && b == 0 {
                assert_eq!(g, 0, "gcd(0,0) = 0");
                continue;
            }
            assert!(g > 0, "gcd({a},{b}) must be positive");
            assert_eq!(a % g, 0, "gcd({a},{b}) must divide {a}");
            assert_eq!(b % g, 0, "gcd({a},{b}) must divide {b}");
            assert_eq!(g, Natural::gcd(b, a), "gcd is commutative");
            // Maximality: nothing larger divides both.
            for d in (g + 1)..=a.max(b) {
                assert!(
                    !(a % d == 0 && b % d == 0),
                    "{d} divides both {a} and {b} but exceeds gcd {g}"
                );
            }
        }
    }
    assert_eq!(Natural::gcd(12u64, 0), 12, "gcd(n,0) = n");
    assert_eq!(Natural::gcd(1_071u64, 462), 21, "the classic example");
}

#[test]
fn lcm_satisfies_the_product_identity() {
    // gcd(a,b) * lcm(a,b) = a * b
    for a in 1..50u64 {
        for b in 1..50u64 {
            let (g, l) = (Natural::gcd(a, b), Natural::lcm(a, b));
            assert_eq!(g * l, a * b, "gcd*lcm = a*b at ({a},{b})");
            assert_eq!(l % a, 0, "lcm must be a multiple of {a}");
            assert_eq!(l % b, 0, "lcm must be a multiple of {b}");
        }
    }
    assert_eq!(Natural::lcm(0u64, 5), 0, "lcm with zero is zero");
    assert_eq!(Natural::lcm(4u64, 6), 12);
}

#[test]
fn lcm_divides_before_multiplying_so_large_operands_survive() {
    // (2^62) and 4 share a factor of 4, so the lcm is 2^62 and fits — but a*b would not.
    let a = 1u64 << 62;
    assert_eq!(
        Natural::lcm(a, 4),
        a,
        "no overflow on the way to a small answer"
    );
    assert_eq!(Natural::lcm(a, 2), a);
}

#[test]
fn isqrt_is_exact_where_the_float_round_trip_is_not() {
    for n in 0..1_000u64 {
        let r = Natural::isqrt(n);
        assert!(r * r <= n, "isqrt({n}) = {r} too large");
        assert!((r + 1) * (r + 1) > n, "isqrt({n}) = {r} too small");
    }
    // Perfect squares.
    for r in 0..1_000u64 {
        assert_eq!(Natural::isqrt(r * r), r, "isqrt of a perfect square");
        if r > 0 {
            assert_eq!(Natural::isqrt(r * r - 1), r - 1, "just below");
        }
    }
    // Above 2^52 an f64 can no longer represent every integer, so a `sqrt` round trip stops
    // being exact. A root near 2^30 puts the square near 2^60 — past that threshold, and
    // still inside u64.
    for k in [30u32, 31] {
        let root = (1u64 << k) + 1;
        let n = root * root;
        assert!(
            n > (1u64 << 52),
            "n must exceed 2^52 for this to be the hard case"
        );
        assert_eq!(Natural::isqrt(n), root, "isqrt is exact at n = {n}");
        assert_eq!(Natural::isqrt(n - 1), root - 1, "and just below it");
    }
}

#[test]
fn signed_gcd_and_lcm_return_non_negative_representatives() {
    for a in -30..30i64 {
        for b in -30..30i64 {
            let g = Integer::gcd(a, b);
            assert!(g >= 0, "gcd({a},{b}) = {g} must be non-negative");
            if a != 0 || b != 0 {
                assert_eq!(a % g, 0);
                assert_eq!(b % g, 0);
            }
            let l = Integer::lcm(a, b);
            assert!(l >= 0, "lcm({a},{b}) = {l} must be non-negative");
        }
    }
    assert_eq!(Integer::gcd(-12i64, 18), 6);
    assert_eq!(Integer::gcd(12i64, -18), 6);
    assert_eq!(Integer::gcd(-12i64, -18), 6);
    assert_eq!(Integer::lcm(-4i64, 6), 12);
}

// --- the f32 carrier ----------------------------------------------------

#[test]
fn the_normal_family_works_on_f32() {
    assert_close_f32(Real::ndtr(0.0f32), 0.5, 8.0, "ndtr(0)");
    assert_close_f32(Real::ndtr(1.0f32), 0.841_344_746_068_542_9, 64.0, "ndtr(1)");
    assert_close_f32(Real::ndtri(0.975f32), 1.959_963_984_540_054, 512.0, "ndtri");
    // The f32 exponent range is far narrower, so log_ndtr matters sooner.
    assert_eq!(Real::ndtr(-20.0f32), 0.0, "Phi underflows in f32 here");
    let v = Real::log_ndtr(-20.0f32);
    assert!(v.is_finite() && v < 0.0, "log_ndtr(-20) = {v}");
    // Round trip in the tail.
    for p in [1e-10f32, 1e-20, 1e-30] {
        let x = Real::ndtri(p);
        assert!(x.is_finite() && x < 0.0, "ndtri({p}) = {x}");
        assert!((Real::log_ndtr(x) - p.ln()).abs() < 1e-3 * p.ln().abs());
    }
    // log10 / exp2 on f32.
    assert_eq!(Real::log10(1000.0f32), 3.0);
    assert_eq!(Real::exp2(10.0f32), 1024.0);
    // And the exact integer surface on the narrow widths.
    assert_eq!(Natural::gcd(12u32, 18), 6);
    assert_eq!(Natural::lcm(4u32, 6), 12);
    assert_eq!(Natural::isqrt(99u32), 9);
    assert_eq!(Integer::gcd(-12i32, 18), 6);
}
