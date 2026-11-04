//! Trigonometric and hyperbolic identities.

use crate::Real;
use euclid_utils::assert_close;

// --- trigonometry -------------------------------------------------------

#[test]
fn pythagorean_identity_holds() {
    for i in -200..200 {
        let x = f64::from(i) * 0.1;
        let (s, c) = Real::sin_cos(x);
        assert_close(s * s + c * c, 1.0, 1e-14, "sin² + cos² = 1");
        // sin_cos must agree with the separate functions.
        assert_close(s, Real::sin(x), 1e-15, "sin_cos matches sin");
        assert_close(c, Real::cos(x), 1e-15, "sin_cos matches cos");
    }
}

#[test]
fn inverse_hyperbolics_invert() {
    for i in -50..50 {
        let x = f64::from(i) * 0.2;
        assert_close(Real::asinh(Real::sinh(x)), x, 1e-13, "asinh(sinh(x)) = x");
    }
    // `tanh` saturates toward ±1, so the round trip is information-destroying outside a
    // modest range: by |x| = 10, tanh(x) is within 4e-9 of 1 and `atanh` cannot recover the
    // argument. Checked where the round trip is actually invertible.
    for i in -15..15 {
        let x = f64::from(i) * 0.2;
        assert_close(Real::atanh(Real::tanh(x)), x, 1e-12, "atanh(tanh(x)) = x");
    }
    for i in 1..50 {
        let x = 1.0 + f64::from(i) * 0.2;
        assert_close(Real::acosh(Real::cosh(x)), x, 1e-12, "acosh(cosh(x)) = x");
    }
}

#[test]
fn degree_radian_conversions_round_trip() {
    for i in -36..36 {
        let deg = f64::from(i) * 10.0;
        assert_close(
            Real::to_degrees(Real::to_radians(deg)),
            deg,
            1e-14,
            "degrees -> radians -> degrees",
        );
    }
}
