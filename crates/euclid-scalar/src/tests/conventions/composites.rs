//! Composite methods against their defining expressions.

use crate::Real;

// --- composite consistency ----------------------------------------------

#[test]
fn square_recip_and_cbrt_agree_with_their_definitions() {
    for i in 1..80 {
        let x = f64::from(i) * 0.37;
        assert!((Real::square(&x) - x * x).abs() < 1e-13 * x * x);
        assert!((Real::recip(x) - 1.0 / x).abs() < 1e-15 / x);
        let c = Real::cbrt(x);
        assert!((c * c * c - x).abs() < 1e-12 * x, "cbrt^3 != x at {x}");
    }
    // cbrt is defined for negative inputs, unlike sqrt.
    assert!((Real::cbrt(-8.0f64) + 2.0).abs() < 1e-14);
}

#[test]
fn log2_agrees_with_the_natural_log_scaled() {
    for i in 1..100 {
        let x = f64::from(i) * 0.5;
        let want = Real::ln(&x) * <f64 as Real>::log2_e();
        assert!(
            (Real::log2(x) - want).abs() < 1e-13 * want.abs().max(1.0),
            "log2 disagrees at {x}"
        );
    }
    // Exact at powers of two.
    for p in -20..20 {
        assert_eq!(Real::log2(2.0f64.powi(p)), f64::from(p));
    }
}

#[test]
fn mul_add_matches_the_unfused_expression_to_within_one_rounding() {
    // `mul_add` may be a single fused instruction, so it can be *more* accurate than
    // `a*b + c`. The contract is that it is not less accurate.
    for i in 1..40 {
        let (a, b, c) = (f64::from(i) * 0.3, 1.7, -0.9);
        let fused = Real::mul_add(a, b, c);
        let unfused = a * b + c;
        assert!(
            (fused - unfused).abs() <= 4.0 * f64::EPSILON * fused.abs().max(1.0),
            "mul_add diverged at {a}"
        );
    }
}

#[test]
fn powi_matches_repeated_multiplication() {
    for base in [-3.0f64, -0.5, 0.5, 2.0, 7.0] {
        for n in 0..10i64 {
            let mut want = 1.0f64;
            for _ in 0..n {
                want *= base;
            }
            let got = Real::powi(base, n);
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "powi({base}, {n}) = {got}, want {want}"
            );
        }
        // Negative exponents are the reciprocal.
        let got = Real::powi(base, -3);
        let want = 1.0 / (base * base * base);
        assert!((got - want).abs() <= 1e-12 * want.abs().max(1.0));
    }
    // powi is defined for negative bases where powf is not.
    assert_eq!(Real::powi(-2.0f64, 2), 4.0);
    assert!(
        Real::powf(&-2.0f64, 0.5).is_nan(),
        "powf of a negative base"
    );
}

#[test]
fn rem_euclid_is_always_non_negative() {
    // The convention that matters for angle wrapping: `rem_euclid` returns a value in
    // `[0, |m|)`, unlike `%` which keeps the sign of the dividend.
    for i in -60..60 {
        let x = f64::from(i) * 0.7;
        let m = 3.0f64;
        let r = Real::rem_euclid(x, m);
        assert!(
            (0.0..m).contains(&r),
            "rem_euclid({x}, {m}) = {r} escaped [0, {m})"
        );
        // And it is a true remainder: x = q*m + r for some integer q.
        let q = Real::round((x - r) / m);
        assert!((q * m + r - x).abs() < 1e-12, "not a remainder at {x}");
    }
    // Contrast with `%`, which keeps the dividend's sign. Read through variables so the
    // comparison is not constant-folded away.
    let (dividend, modulus) = (-1.0f64, 3.0f64);
    assert!(dividend % modulus < 0.0, "`%` keeps the dividend's sign");
    assert_eq!(Real::rem_euclid(dividend, modulus), 2.0);
}

#[test]
fn angle_wrapping_through_rem_euclid_is_stable_over_many_turns() {
    // The practical use: normalizing an accumulated angle. Must stay in range even after
    // many turns, where naive subtraction loops would drift or run long.
    let tau = <f64 as Real>::tau();
    for turns in [1.0f64, 10.0, 1e3, 1e6] {
        let angle = core::f64::consts::FRAC_PI_4 + turns * tau;
        let wrapped = Real::rem_euclid(angle, tau);
        assert!((0.0..tau).contains(&wrapped));
        // At 1e6 turns the input has ~6 digits of the fraction left, so compare loosely.
        assert!(
            (wrapped - core::f64::consts::FRAC_PI_4).abs() < 1e-8,
            "wrapped {wrapped} after {turns} turns"
        );
    }
}

#[test]
fn degree_radian_conversions_match_the_turn_constants() {
    assert_eq!(Real::to_radians(0.0f64), 0.0);
    let half = <f64 as Real>::half_turn_degrees();
    assert!((Real::to_radians(half) - <f64 as Real>::pi()).abs() < 1e-15);
    assert!(
        (Real::to_degrees(<f64 as Real>::pi()) - half).abs() < 1e-13,
        "to_degrees(pi) should be 180"
    );
    // Quarter turn is pi/2.
    assert!(
        (Real::to_radians(<f64 as Real>::quarter_turn_degrees()) - <f64 as Real>::frac_pi_2())
            .abs()
            < 1e-15
    );
}
