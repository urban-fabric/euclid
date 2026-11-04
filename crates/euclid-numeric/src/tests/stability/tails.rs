//! Tail-regime special functions and log-domain CDFs.

use crate::Real;
use euclid_utils::rel;

// --- tail-regime special functions --------------------------------------

#[test]
fn erfcx_carries_the_gaussian_tail_where_erfc_underflows() {
    // erfc(30) underflows to exactly 0, so any ratio built on it dies. erfcx does not.
    assert_eq!(Real::erfc(30.0), 0.0, "erfc underflows here");
    let scaled = Real::erfcx(30.0);
    assert!(scaled.is_finite() && scaled > 0.0, "erfcx(30) = {scaled}");
    // erfcx(x) ~ 1/(x√π) for large x.
    let asymptotic = 1.0 / (30.0 * core::f64::consts::PI.sqrt());
    assert!(rel(scaled, asymptotic) < 1e-3);

    // Still usable far beyond that.
    for &x in &[100.0f64, 1e4, 1e8] {
        let v = Real::erfcx(x);
        assert!(v > 0.0 && v.is_finite());
        assert!(rel(v, 1.0 / (x * core::f64::consts::PI.sqrt())) < 1e-3);
    }
}

#[test]
fn erfc_tail_does_not_underflow_prematurely() {
    // A naive `1 − erf(x)` is exactly 0 from x ≈ 6; the real value is ~2e-17.
    assert_eq!(1.0 - Real::erf(6.0), 0.0, "naive complement underflows");
    assert!(rel(Real::erfc(6.0), 2.151_973_671_249_891e-17) < 1e-11);
    // And keeps going well past that.
    assert!(Real::erfc(20.0) > 0.0);
    assert!(Real::erfc(25.0) > 0.0);
}

#[test]
fn scaled_bessel_survives_arguments_where_the_unscaled_form_overflows() {
    // I₀(800) overflows f64 completely; the scaled form is ~1/√(2πx).
    assert!(800.0f64.exp().is_infinite(), "e^x overflows at this x");
    for &x in &[100.0f64, 800.0, 1e5] {
        let i0e = Real::bessel_i0e(x);
        assert!(i0e > 0.0 && i0e.is_finite(), "bessel_i0e({x}) = {i0e}");
        // Asymptotically e^{-x} I₀(x) → 1/√(2πx).
        let asym = 1.0 / (core::f64::consts::TAU * x).sqrt();
        assert!(rel(i0e, asym) < 1e-2, "i0e({x})={i0e} vs {asym}");
    }
    // ln I₀ stays finite where I₀ itself cannot be represented.
    for &x in &[5.0f64, 100.0, 800.0, 1e5] {
        let direct = Real::ln_bessel_i0(x);
        assert!(direct.is_finite() && direct > 0.0);
        // `ln_bessel_i0` and the composed `|x| + ln I₀ᵉ(x)` take different internal paths at
        // large `x`, so they agree only to the accuracy of the underlying A&S polynomials.
        // In log space that is an *absolute* gap of ~1e-7 (measured 1.3e-7 at x = 800), which
        // is the same 1e-7 relative error on I₀ itself — not an independent defect.
        let composed = x + Real::bessel_i0e(x).ln();
        assert!(
            (direct - composed).abs() < 1e-5,
            "ln_bessel_i0({x}) = {direct} vs composed {composed}"
        );
    }
}

#[test]
fn bessel_ratio_stays_bounded_and_monotone() {
    // I₁/I₀ ∈ [0,1), the von Mises mean resultant length — must never exceed 1 or go negative.
    let mut prev = 0.0f64;
    for i in 0..400 {
        let x = f64::from(i) * 2.5;
        let r = Real::bessel_i1_over_i0(x);
        assert!(
            (0.0..1.0).contains(&r),
            "i1_over_i0({x}) = {r} escaped [0,1)"
        );
        assert!(
            r >= prev - 1e-6,
            "ratio must be non-decreasing: {prev} -> {r}"
        );
        prev = r;
    }
    // And it approaches 1 for large concentration rather than overflowing.
    assert!(Real::bessel_i1_over_i0(1e4) > 0.999);
}

#[test]
fn incomplete_gamma_and_beta_stay_in_the_unit_interval() {
    // These are CDFs: escaping [0,1] is a hard error, not a precision issue.
    for i in 0..200 {
        let x = f64::from(i) * 0.25;
        for &a in &[0.5f64, 1.0, 2.5, 10.0, 50.0] {
            let lower = Real::gamma_lr(x, a);
            let upper = Real::gamma_ur(x, a);
            assert!((0.0..=1.0).contains(&lower), "gamma_lr({x},{a}) = {lower}");
            assert!((0.0..=1.0).contains(&upper), "gamma_ur({x},{a}) = {upper}");
        }
    }
    for i in 0..=100 {
        let x = f64::from(i) * 0.01;
        for &(a, b) in &[(0.5f64, 0.5f64), (1.0, 1.0), (2.0, 5.0), (50.0, 20.0)] {
            let v = Real::beta_reg(x, a, b);
            assert!((0.0..=1.0).contains(&v), "beta_reg({x},{a},{b}) = {v}");
        }
    }
}

// --- log-domain CDFs ----------------------------------------------------

#[test]
fn normal_log_cdf_is_usable_deep_in_the_left_tail() {
    // The common construction: ln Φ(x) = ln(erfc(−x/√2)/2). Built on plain erfc this dies
    // around x = −9; routed through erfcx it keeps going.
    let ln_phi = |x: f64| {
        let z = -x * core::f64::consts::FRAC_1_SQRT_2;
        // ln(erfcx(z) e^{-z²} / 2)
        Real::erfcx(z).ln() - z * z - core::f64::consts::LN_2
    };
    for &x in &[-5.0f64, -10.0, -30.0, -100.0] {
        let v = ln_phi(x);
        assert!(v.is_finite() && v < 0.0, "ln_phi({x}) = {v}");
        // ln Φ(x) ≈ −x²/2 − ln(−x√(2π)) for very negative x.
        let asym = -x * x / 2.0 - (-x * core::f64::consts::TAU.sqrt()).ln();
        assert!(rel(v, asym) < 1e-2, "ln_phi({x}) = {v}, asymptotic {asym}");
    }
    // Monotone increasing.
    let mut prev = f64::NEG_INFINITY;
    for i in -500..0 {
        let v = ln_phi(f64::from(i) * 0.2);
        assert!(v > prev, "ln Phi must increase");
        prev = v;
    }
}
