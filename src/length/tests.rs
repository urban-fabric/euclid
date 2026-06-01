#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;

#[test]
fn length_meters_constants_and_quantity_impl() {
    assert_eq!(LengthMeters::ZERO.0, 0.0);
    assert_eq!(LengthMeters::INFINITY.0, f64::INFINITY);
    assert_eq!(LengthMeters::DIMENSION, Dimension::Length);
    assert_eq!(LengthMeters::UNIT_NAME, "m");
    assert_eq!(LengthMeters::new(42.5).raw(), 42.5);
    assert_eq!(LengthMeters::from_raw(7.0), LengthMeters(7.0));
}

#[test]
fn length_meters_arithmetic() {
    let a = LengthMeters(100.0);
    let b = LengthMeters(25.0);
    assert_eq!((a + b).0, 125.0);
    assert_eq!((a - b).0, 75.0);
    assert_eq!((-a).0, -100.0);
    let mut c = a;
    c += b;
    assert_eq!(c.0, 125.0);
    c -= b;
    assert_eq!(c.0, 100.0);
    assert_eq!((a * 2.0).0, 200.0);
    assert_eq!((2.0 * a).0, 200.0);
    assert_eq!((a / 4.0).0, 25.0);
    let sum: LengthMeters = [a, b, b].into_iter().sum();
    assert_eq!(sum.0, 150.0);
}

#[test]
fn length_km_round_trip() {
    let m = LengthMeters(1500.0);
    let km = m.to_km();
    assert!((km.0 - 1.5).abs() < 1e-12);
    assert!((km.to_meters().0 - 1500.0).abs() < 1e-9);
}

#[test]
fn length_feet_round_trip() {
    let m = LengthMeters(1.0);
    let ft = m.to_feet();
    assert!((ft.0 - 3.280_839_895_013_123).abs() < 1e-12);
    assert!((ft.to_meters().0 - 1.0).abs() < 1e-12);
}

#[test]
fn length_km_arithmetic_is_in_dimension() {
    let a = LengthKm(1.0);
    let b = LengthKm(0.5);
    assert_eq!((a + b).0, 1.5);
    assert_eq!((a * 2.0).0, 2.0);
}
