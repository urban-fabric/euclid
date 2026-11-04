//! Sign conventions: signum, copysign, abs.

use crate::Real;

// --- sign family --------------------------------------------------------

#[test]
fn signum_returns_plus_or_minus_one_and_never_zero() {
    // The ambiguity: some languages define sign(0) = 0. Rust's `f64::signum` returns ±1.0
    // even at zero, carrying the sign bit.
    assert_eq!(Real::signum(3.0f64), 1.0);
    assert_eq!(Real::signum(-3.0f64), -1.0);
    assert_eq!(Real::signum(0.0f64), 1.0, "signum(+0) is +1, not 0");
    assert_eq!(Real::signum(-0.0f64), -1.0, "signum(-0) is -1, not 0");
    assert!(Real::signum(f64::NAN).is_nan());
}

#[test]
fn copysign_transfers_only_the_sign_bit() {
    assert_eq!(Real::copysign(3.0f64, 1.0), 3.0);
    assert_eq!(Real::copysign(3.0f64, -1.0), -3.0);
    assert_eq!(Real::copysign(-3.0f64, 1.0), 3.0);
    // Signed zero is a valid sign source — this is how the sign of a zero is observable.
    assert_eq!(Real::copysign(1.0f64, -0.0), -1.0);
    assert_eq!(Real::copysign(1.0f64, 0.0), 1.0);
    // Magnitude is preserved exactly, including for infinities.
    assert_eq!(Real::copysign(f64::INFINITY, -1.0), f64::NEG_INFINITY);
    // abs is the |·| half of the same decomposition.
    for i in -30..30 {
        let x = f64::from(i) * 0.7;
        assert_eq!(Real::copysign(Real::abs(&x), x), x, "copysign(|x|, x) = x");
    }
}
