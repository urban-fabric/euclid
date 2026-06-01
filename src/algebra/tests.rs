#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;

#[test]
fn length_div_time_is_speed() {
    let l = LengthMeters::new(100.0);
    let t = TimeSeconds::new(10.0);
    let v: SpeedMs = l / t;
    assert_eq!(v.as_ms(), 10.0);
}

#[test]
fn length_div_speed_is_time() {
    let l = LengthMeters::new(100.0);
    let v = SpeedMs::new(20.0);
    let t: TimeSeconds = l / v;
    assert_eq!(t.as_seconds(), 5.0);
}

#[test]
fn speed_mul_time_is_length_both_orderings() {
    let v = SpeedMs::new(20.0);
    let t = TimeSeconds::new(5.0);
    let a: LengthMeters = v * t;
    let b: LengthMeters = t * v;
    assert_eq!(a.as_meters(), 100.0);
    assert_eq!(b.as_meters(), 100.0);
}

#[test]
fn algebra_identity_roundtrip() {
    let v = SpeedMs::new(7.5);
    let t = TimeSeconds::new(3.0);
    let v2 = (v * t) / t;
    assert!((v2.as_ms() - v.as_ms()).abs() < 1e-12);

    let l = LengthMeters::new(123.5);
    let v = SpeedMs::new(2.5);
    let l2 = (l / v) * v;
    assert!((l2.as_meters() - l.as_meters()).abs() < 1e-12);
}

#[test]
fn factor_preserves_unit() {
    let half = Factor::new(0.5);
    let l = LengthMeters::new(100.0);
    let scaled: LengthMeters = half * l;
    assert_eq!(scaled.as_meters(), 50.0);
    let scaled2: LengthMeters = l * half;
    assert_eq!(scaled2.as_meters(), 50.0);

    let kph = SpeedKph::new(50.0);
    let scaled3: SpeedKph = half * kph;
    assert_eq!(scaled3.as_kph(), 25.0);
}

#[test]
fn factor_identity_is_neutral() {
    let l = LengthMeters::new(42.0);
    assert_eq!((Factor::IDENTITY * l).as_meters(), 42.0);
    assert_eq!((l * Factor::IDENTITY).as_meters(), 42.0);
}
