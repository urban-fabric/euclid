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

// --- zero signs from degree-argument trigonometry -----------------------

#[test]
fn sin_cos_degrees_follows_the_ieee_sign_conventions() {
    // F.10.1.12/13: a vanished cosine is `+0`, and a vanished sine carries the
    // argument's sign. Checked on the bits, since `-0.0 == 0.0`.
    let (_, cos) = 90.0_f64.sin_cos_degrees();
    assert!(
        cos.is_sign_positive(),
        "a zero cosine must be +0, got {cos}"
    );
    let (sin_neg, _) = (-360.0_f64).sin_cos_degrees();
    assert!(
        sin_neg.is_sign_negative(),
        "a zero sine takes the argument's sign, got {sin_neg}"
    );
    let (sin_pos, _) = 360.0_f64.sin_cos_degrees();
    assert!(sin_pos.is_sign_positive());
}
