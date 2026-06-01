#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;

#[test]
fn length_div_time_is_speed() {
    let l = LengthMeters(100.0);
    let t = TimeSeconds(10.0);
    let v: SpeedMs = l / t;
    assert_eq!(v.0, 10.0);
}

#[test]
fn length_div_speed_is_time() {
    let l = LengthMeters(100.0);
    let v = SpeedMs(20.0);
    let t: TimeSeconds = l / v;
    assert_eq!(t.0, 5.0);
}

#[test]
fn speed_mul_time_is_length_both_orderings() {
    let v = SpeedMs(20.0);
    let t = TimeSeconds(5.0);
    let a: LengthMeters = v * t;
    let b: LengthMeters = t * v;
    assert_eq!(a.0, 100.0);
    assert_eq!(b.0, 100.0);
}

#[test]
fn algebra_identity_roundtrip() {
    let v = SpeedMs(7.5);
    let t = TimeSeconds(3.0);
    let v2 = (v * t) / t;
    assert!((v2.0 - v.0).abs() < 1e-12);

    let l = LengthMeters(123.5);
    let v = SpeedMs(2.5);
    let l2 = (l / v) * v;
    assert!((l2.0 - l.0).abs() < 1e-12);
}

#[test]
fn factor_preserves_unit() {
    let half = Factor(0.5);
    let l = LengthMeters(100.0);
    let scaled: LengthMeters = half * l;
    assert_eq!(scaled.0, 50.0);
    let scaled2: LengthMeters = l * half;
    assert_eq!(scaled2.0, 50.0);

    let kph = SpeedKph(50.0);
    let scaled3: SpeedKph = half * kph;
    assert_eq!(scaled3.0, 25.0);
}

#[test]
fn factor_identity_is_neutral() {
    let l = LengthMeters(42.0);
    assert_eq!((Factor::IDENTITY * l).0, 42.0);
    assert_eq!((l * Factor::IDENTITY).0, 42.0);
}
