//! Overflow and saturation: the stable forms where the naive expression returns inf or 0.

use crate::{Interpolate, Real};
use euclid_utils::rel;

// --- overflow / saturation ----------------------------------------------

#[test]
fn ln_1p_exp_is_stable_at_both_extremes() {
    // softplus. For large x, e^x overflows and the naive form gives inf; the answer is ~x.
    let big = 800.0f64;
    assert!(big.exp().is_infinite(), "naive e^x overflows here");
    assert!(rel(Real::ln_1p_exp(&big), big) < 1e-15);

    // For very negative x, softplus(x) ≈ e^x, which must not underflow to a bogus 0 early.
    let small = -40.0f64;
    assert!(rel(Real::ln_1p_exp(&small), small.exp()) < 1e-12);

    // Monotone and non-negative throughout.
    let mut prev = f64::NEG_INFINITY;
    for i in -200..200 {
        let v = Real::ln_1p_exp(&(f64::from(i) * 5.0));
        assert!(v >= 0.0 && v >= prev && v.is_finite());
        prev = v;
    }
}

#[test]
fn sigmoid_and_its_logs_do_not_overflow() {
    for &x in &[-800.0f64, -50.0, 0.0, 50.0, 800.0] {
        let s = Real::sigmoid(&x);
        assert!((0.0..=1.0).contains(&s), "sigmoid({x}) = {s}");
        // The log forms must stay finite where sigmoid itself has saturated to 0 or 1.
        assert!(Real::ln_sigmoid(&x).is_finite() || x < 0.0);
        assert!(Real::ln_1m_sigmoid(&x).is_finite() || x > 0.0);
    }
    // ln_sigmoid(-800) ≈ -800 even though sigmoid(-800) underflows to 0.
    assert_eq!(Real::sigmoid(&-800.0), 0.0);
    assert!(rel(Real::ln_sigmoid(&-800.0), -800.0) < 1e-12);
}

#[test]
fn ln_add_exp_beats_the_naive_log_sum_exp() {
    // Both terms overflow individually; the shifted form is exact.
    let (a, b) = (1000.0f64, 1000.0f64);
    assert!(
        (a.exp() + b.exp()).ln().is_infinite(),
        "naive log-sum-exp overflows here"
    );
    assert!(rel(Real::ln_add_exp(a, b), 1000.0 + core::f64::consts::LN_2) < 1e-15);

    // And underflow at the other end: e^-800 is 0, but the answer is ~-800.
    let (c, d) = (-800.0f64, -810.0f64);
    assert_eq!(c.exp() + d.exp(), 0.0, "naive form underflows to zero");
    let got = Real::ln_add_exp(c, d);
    assert!(
        got.is_finite() && (got - c).abs() < 1.0,
        "ln_add_exp = {got}"
    );
}

#[test]
fn ln_sub_exp_inverts_ln_add_exp() {
    // log(e^a − e^b) recovered from the sum, across magnitudes that would overflow naively.
    for &(a, b) in &[(1000.0f64, 999.0f64), (5.0, 1.0), (-700.0, -705.0)] {
        let sum = Real::ln_add_exp(a, b);
        let recovered = Real::ln_sub_exp(sum, b);
        assert!(
            (recovered - a).abs() < 1e-9 * a.abs().max(1.0),
            "ln_sub_exp(ln_add_exp({a},{b}),{b}) = {recovered}"
        );
    }
}

#[test]
fn hypot_does_not_overflow_on_large_operands() {
    // sqrt(x² + y²) overflows for x near the f64 ceiling; hypot must not.
    let x = 1e300f64;
    assert!((x * x + x * x).sqrt().is_infinite(), "naive form overflows");
    let h = Real::hypot(x, x);
    assert!(h.is_finite());
    assert!(rel(h, x * core::f64::consts::SQRT_2) < 1e-15);

    // And no underflow at the bottom.
    let t = 1e-300f64;
    assert!(rel(Real::hypot(t, t), t * core::f64::consts::SQRT_2) < 1e-15);
}

#[test]
fn midpoint_does_not_overflow_on_opposite_extremes() {
    // The classic `a + (b − a)/2` overflows for large opposite-signed operands.
    assert_eq!(Interpolate::midpoint(1e308_f64, -1e308), 0.0);
    assert_eq!(Interpolate::midpoint(f64::MAX, f64::MAX), f64::MAX);
    assert!(Interpolate::midpoint(f64::MAX, f64::MIN).abs() < 1.0);
}

/// `Interpolate` at `f32`, which is a **separate body** and was riding along untested.
///
/// Three mutants lived here: `midpoint` replaceable with `Default::default()`, and `lerp`'s
/// subtraction replaceable with an addition. The test above covers `f64` only, and the f32 impl
/// is its own `impl` block — cause 3 in [MUTANTS.md](../../../../../MUTANTS.md), the same reason
/// the `NormedDivisionAlgebra` tier and the f32 rising factorial went unchecked.
///
/// `lerp` is `self + (target − self) · t`, so the endpoints and the halfway point are what pin
/// the sign: with `−` flipped to `+`, `lerp(a, b, 1)` becomes `a + (b + a)` rather than `b`.
#[test]
fn interpolation_is_stable_at_f32_too() {
    // The endpoints are exact, which is what makes lerp usable as a parameterisation.
    for &(a, b) in &[(0.0f32, 1.0f32), (-2.0, 3.0), (1e30, -1e30), (0.25, 0.75)] {
        assert_eq!(
            Interpolate::lerp(a, b, 0.0),
            a,
            "lerp at t = 0 is the start"
        );
        assert_eq!(Interpolate::lerp(a, b, 1.0), b, "lerp at t = 1 is the end");
        // Halfway must agree with the midpoint, which is computed by a different route.
        let half = Interpolate::lerp(a, b, 0.5);
        let mid = Interpolate::midpoint(a, b);
        assert!(
            (half - mid).abs() <= 1e-6 * mid.abs().max(1.0),
            "lerp(a, b, 1/2) = {half:e} disagrees with midpoint = {mid:e}"
        );
    }
    // The overflow stability the f32 impl claims, mirroring the f64 case above.
    assert_eq!(Interpolate::midpoint(1e38_f32, -1e38), 0.0);
    assert_eq!(Interpolate::midpoint(f32::MAX, f32::MAX), f32::MAX);
    assert!(Interpolate::midpoint(f32::MAX, f32::MIN).abs() < 1.0);
    // And midpoint is not the constant a `Default::default()` body would make it.
    assert_eq!(Interpolate::midpoint(2.0f32, 4.0), 3.0);
}
