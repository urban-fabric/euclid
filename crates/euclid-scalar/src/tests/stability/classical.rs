//! Stability of the classical special functions.
//!
//! Each test here pairs a function against the *obvious* way to compute it, at an argument
//! where the obvious way demonstrably fails — and asserts the failure as well as the success.
//! That is what makes these tests documentation rather than decoration: they record why the
//! implementation is shaped the way it is, in a form that breaks if someone simplifies it back.
//!
//! Every naive form below is one a reasonable person would write first.

use crate::Real;
use euclid_utils::{assert_close, rel};

/// `Si(x)` by its defining power series, summed as written.
///
/// `Si(x) = Σ (−1)^k x^{2k+1} / ((2k+1)(2k+1)!)`, which converges for every `x` — the series is
/// *mathematically* fine at any argument, which is exactly what makes it a trap.
fn naive_si_series(x: f64) -> f64 {
    let mut power_over_factorial = x;
    let mut total = x;
    for k in 1..400u32 {
        let m = 2.0 * f64::from(k);
        power_over_factorial *= x * x / (m * (m + 1.0));
        let term = power_over_factorial / (m + 1.0);
        total += if k % 2 == 0 { term } else { -term };
    }
    total
}

/// `J₀(x)` by its defining power series, summed as written.
fn naive_j0_series(x: f64) -> f64 {
    let h = x * 0.5;
    let mut term = 1.0f64;
    let mut total = 1.0f64;
    for k in 1..400u32 {
        let kf = f64::from(k);
        term *= -h * h / (kf * kf);
        total += term;
    }
    total
}

#[test]
fn the_trigonometric_integral_series_is_destroyed_by_cancellation() {
    // The series' terms peak near k = x/2 before decaying, so the partial sums must cancel
    // enormously: at x = 45 they reach 4e20 while Si(45) is about 1.56. The rounding error
    // floor is eps times the largest intermediate -- 4e4 -- so there is nothing left.
    //
    // This is not a hypothetical. It is what this crate's own Si returned before the continued
    // fraction replaced the series: 10.46 instead of 1.5587.
    for &(x, reference) in &[
        (45.0f64, 1.558_715_000_896_412_8e0),
        (50.0, 1.551_617_072_485_936e0),
    ] {
        let naive = naive_si_series(x);
        assert!(
            (naive - reference).abs() > 1.0,
            "the naive series is supposed to fail at x = {x}, but returned {naive:e}"
        );
        assert_close(
            Real::sin_int(x),
            reference,
            1e-15,
            &format!("Si({x}) from the continued fraction"),
        );
    }
    // Below the crossover the series is perfectly good, which is why it is still used there.
    for &x in &[0.5f64, 1.0, 2.0, 3.5] {
        assert_close(
            naive_si_series(x),
            Real::sin_int(x),
            1e-15,
            &format!("the series still agrees at x = {x}"),
        );
    }
}

#[test]
fn the_bessel_j_series_is_destroyed_by_the_same_cancellation() {
    // J0's series terms sum in magnitude to I0(x), so the error floor is eps*I0(x): already
    // 5e-14 by x = 8, and 1.4e3 by x = 50. The aliased trapezoidal rule has no such floor,
    // being an average of bounded cosines.
    let reference_j0_50 = 5.581_232_766_925_182e-2;
    let naive = naive_j0_series(50.0);
    assert!(
        (naive - reference_j0_50).abs() > 1.0,
        "the naive J0 series is supposed to fail at x = 50, but returned {naive:e}"
    );
    assert_close(
        Real::bessel_j0(50.0),
        reference_j0_50,
        1e-15,
        "J0(50) from the Hankel expansion",
    );
    // And at a middle argument, where the series has lost some digits but not all: this is the
    // range the quadrature exists for. At x = 16 the terms sum in magnitude to I0(16) = 8.9e5,
    // so the series' floor is about 5.6e-10 relative, while the quadrature is unaffected.
    let reference_j0_16 = -1.748_990_739_836_292e-1;
    assert!(
        rel(naive_j0_series(16.0), reference_j0_16) > 1e-12,
        "the naive series should already have lost digits at x = 16"
    );
    assert!(
        rel(Real::bessel_j0(16.0), reference_j0_16) < 1e-14,
        "the quadrature should not have"
    );
}

#[test]
fn the_scaled_forms_survive_where_scaling_by_hand_cannot() {
    // The naive way to get e^x K0(x) is to compute K0(x) and multiply. Past x ~ 746 that is
    // zero times infinity.
    let x = 800.0f64;
    assert_eq!(Real::bessel_k0(x), 0.0, "K0 has underflowed");
    assert!(x.exp().is_infinite(), "e^x has overflowed");
    assert!(
        (Real::bessel_k0(x) * x.exp()).is_nan(),
        "so scaling by hand gives NaN"
    );
    // The scaled form is computed directly and is an ordinary number.
    let scaled = Real::bessel_k0e(x);
    assert!(scaled > 0.0 && scaled.is_finite(), "K0e({x}) = {scaled}");
    // Pinned against the *unscaled* form where that still works, which fixes the scaling
    // convention without needing an asymptotic expansion as a reference. (Comparing to a
    // truncated asymptotic series here would be measuring the truncation: three terms leave
    // `a₃/x³ = 1.4e-10` at x = 800.)
    for &moderate in &[1.0f64, 5.0, 20.0, 100.0] {
        assert_close(
            Real::bessel_k0e(moderate) / (Real::bessel_k0(moderate) * moderate.exp()),
            1.0,
            1e-14,
            &format!("K0e({moderate}) = e^x K0(x) where both are representable"),
        );
    }

    // The value at x = 800 is then pinned by the Wronskian I0 K1 + I1 K0 = 1/x. Written with
    // the unscaled functions that identity is unevaluable past 710 -- one factor overflows
    // while the other underflows -- yet the identity itself is perfectly finite there, and the
    // scaled forms deliver it to full precision.
    assert!(
        (Real::bessel_i0(x) * Real::bessel_k1(x)).is_nan(),
        "the unscaled product is inf * 0"
    );
    assert_close(
        (Real::bessel_i0e(x) * Real::bessel_k1e(x) + Real::bessel_i1e(x) * Real::bessel_k0e(x)) * x,
        1.0,
        1e-13,
        "the scaled Wronskian at x = 800",
    );
}

#[test]
fn the_alternating_zeta_series_needs_acceleration_inside_the_critical_strip() {
    // eta(s) = sum (-1)^{k-1} k^{-s} converges for every s > 0, but only conditionally: its
    // terms decay like k^{-s}, so at s = 1/2 the partial sums wander by about N^{-1/2}.
    // Reaching f64 precision this way would take 1e32 terms.
    let s = 0.5f64;
    let mut partial = 0.0f64;
    for k in 1..1_000_000u32 {
        let term = f64::from(k).powf(-s);
        partial += if k % 2 == 1 { term } else { -term };
    }
    let naive_zeta = partial / (1.0 - 2f64.powf(1.0 - s));
    let reference = -1.460_354_508_809_586_8e0;
    assert!(
        (naive_zeta - reference).abs() > 1e-5,
        "a million terms should still be short by ~1e-4, got {naive_zeta}"
    );
    // The Cohen-Rodriguez Villegas-Zagier acceleration reaches the same value from 24 terms.
    assert_close(Real::zeta(s), reference, 1e-13, "zeta(1/2), accelerated");
}

#[test]
fn halving_before_taking_a_logarithm_loses_the_smallest_subnormals() {
    // `ln(x/2)` appears in the small-argument series of Y0, Y1, K0 and K1. Spelled
    // `(x * 0.5).ln()` it is a latent trap: at the smallest subnormal the halving rounds to
    // exactly zero, so the logarithm is -inf and the whole series collapses -- even though the
    // true ln(x/2) is about -745, an entirely ordinary number.
    let tiny = 5e-324f64;
    assert_eq!(tiny * 0.5, 0.0, "the halving underflows");
    assert!((tiny * 0.5).ln().is_infinite(), "so the naive form is -inf");
    assert!(
        (tiny.ln() - core::f64::consts::LN_2).is_finite(),
        "the subtraction form is finite"
    );
    // So the functions built on it stay finite and correct at the bottom of the range.
    for &x in &[tiny, 1e-323, f64::MIN_POSITIVE] {
        let log_term = x.ln() - core::f64::consts::LN_2 + euclid_utils::EULER_GAMMA;
        assert!(Real::bessel_k0(x).is_finite(), "K0({x:e}) must be finite");
        assert_close(
            Real::bessel_k0(x) / -log_term,
            1.0,
            1e-12,
            &format!("K0({x:e}) ~ -ln(x/2) - gamma"),
        );
        assert!(Real::bessel_y0(x).is_finite(), "Y0({x:e}) must be finite");
        assert!(Real::bessel_i0(x) == 1.0, "I0({x:e}) must be exactly 1");
    }
}

#[test]
fn the_small_argument_bessel_limits_are_exact_rather_than_merely_small() {
    // J1(x) ~ x/2 as x -> 0 is a perfectly conditioned limit, so it should come back with full
    // *relative* accuracy, not just a small absolute error. The quadrature cannot do that --
    // its summand cos(x sin t - t) loses the x entirely once x falls below the ulp of pi/2 --
    // which is why a series branch covers the small end.
    // The sweep starts at 1e-14 because each comparison is against a *leading* term only, and
    // the next corrections are relative: x^2/8 for J1, x^2/18 for Si, and x for W. The last of
    // those is the binding one, so anything above ~1e-13 would be testing the reference.
    for &x in &[1e-14f64, 1e-20, 1e-100, 1e-300, f64::MIN_POSITIVE] {
        assert_close(
            Real::bessel_j1(x) / (x * 0.5),
            1.0,
            1e-12,
            &format!("J1({x:e}) ~ x/2 relatively"),
        );
        assert_eq!(Real::bessel_j0(x), 1.0, "J0 -> 1 exactly");
        assert_close(
            Real::sin_int(x) / x,
            1.0,
            1e-12,
            &format!("Si({x:e}) ~ x relatively"),
        );
        assert_close(
            Real::lambert_w(x) / x,
            1.0,
            1e-12,
            &format!("W({x:e}) ~ x relatively"),
        );
    }
}
