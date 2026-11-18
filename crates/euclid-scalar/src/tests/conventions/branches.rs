//! Branch and quadrant conventions for the inverse and trigonometric functions.

use crate::Real;

// --- atan2 branch conventions -------------------------------------------

#[test]
fn atan2_follows_the_standard_quadrant_convention() {
    use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    let close = |a: f64, b: f64, what: &str| {
        assert!((a - b).abs() < 1e-15, "{what}: {a} vs {b}");
    };
    // atan2(y, x) with `self` = y.
    close(Real::atan2(0.0f64, 1.0), 0.0, "atan2(0, 1)");
    close(Real::atan2(1.0f64, 1.0), FRAC_PI_4, "atan2(1, 1)");
    close(Real::atan2(1.0f64, 0.0), FRAC_PI_2, "atan2(1, 0)");
    close(Real::atan2(0.0f64, -1.0), PI, "atan2(0, -1) is +pi");
    close(Real::atan2(-1.0f64, 0.0), -FRAC_PI_2, "atan2(-1, 0)");
    close(
        Real::atan2(-1.0f64, -1.0),
        -3.0 * FRAC_PI_4,
        "third quadrant",
    );
    // Range is (-pi, pi].
    for i in -50..50 {
        for j in -50..50 {
            if i == 0 && j == 0 {
                continue;
            }
            let a = Real::atan2(f64::from(i), f64::from(j));
            assert!(
                a > -PI - 1e-15 && a <= PI + 1e-15,
                "atan2 out of range: {a}"
            );
        }
    }
}

#[test]
fn atan2_at_the_origin_is_defined_not_nan() {
    // Both arguments zero has no mathematical answer; IEEE still defines it by sign.
    assert_eq!(Real::atan2(0.0f64, 0.0), 0.0);
    assert_eq!(Real::atan2(-0.0f64, 0.0), -0.0);
    assert!((Real::atan2(0.0f64, -0.0) - core::f64::consts::PI).abs() < 1e-15);
}

#[test]
fn atan2_recovers_the_angle_of_a_vector() {
    // The motivating use: converting a vector back to its polar angle.
    for i in 1..72 {
        let theta = f64::from(i) * 0.08 - 2.8; // within (-pi, pi)
        let (s, c) = Real::sin_cos(theta);
        let recovered = Real::atan2(s, c);
        assert!(
            (recovered - theta).abs() < 1e-14,
            "atan2(sin t, cos t) != t at {theta}"
        );
    }
}

// --- trigonometric conventions ------------------------------------------

#[test]
fn trig_functions_take_radians_and_hit_their_exact_points() {
    use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    let close = |a: f64, b: f64, what: &str| assert!((a - b).abs() < 1e-15, "{what}: {a}");
    close(Real::sin(0.0f64), 0.0, "sin 0");
    close(Real::cos(0.0f64), 1.0, "cos 0");
    close(Real::sin(FRAC_PI_2), 1.0, "sin pi/2");
    close(Real::cos(PI), -1.0, "cos pi");
    close(Real::tan(FRAC_PI_4), 1.0, "tan pi/4");
    // Inverse functions return the principal branch.
    close(Real::asin(1.0f64), FRAC_PI_2, "asin 1");
    close(Real::acos(1.0f64), 0.0, "acos 1");
    close(Real::atan(1.0f64), FRAC_PI_4, "atan 1");
    // asin/acos ranges.
    for i in -20..=20 {
        let v = f64::from(i) * 0.05;
        let a = Real::asin(v);
        assert!((-FRAC_PI_2..=FRAC_PI_2).contains(&a), "asin range");
        let c = Real::acos(v);
        assert!((0.0..=PI).contains(&c), "acos range");
    }
}

#[test]
fn hyperbolic_functions_satisfy_their_identity() {
    for i in -30..30 {
        let x = f64::from(i) * 0.2;
        // cosh² − sinh² = 1. The identity is itself cancellation-prone: by |x| = 6 both
        // squares are ~4e4, so the difference loses ~5 digits before it is compared to 1.
        // The tolerance therefore scales with the magnitude being cancelled, not with 1.
        let (s, c) = (Real::sinh(x), Real::cosh(x));
        let cancelled = c * c;
        assert!(
            (c * c - s * s - 1.0).abs() < 1e-14 * cancelled.max(1.0),
            "cosh²-sinh²=1 at {x}"
        );
        // tanh = sinh / cosh
        assert!((Real::tanh(x) - s / c).abs() < 1e-14, "tanh at {x}");
    }
}
