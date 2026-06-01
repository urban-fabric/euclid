#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;

#[test]
fn angle_round_trip() {
    let r = AngleRadians(std::f64::consts::PI);
    let d = r.to_degrees();
    assert!((d.0 - 180.0).abs() < 1e-12);
    assert!((d.to_radians().0 - std::f64::consts::PI).abs() < 1e-12);
}

#[test]
fn angle_quantity_impl() {
    assert_eq!(AngleRadians::DIMENSION, Dimension::Angle);
    assert_eq!(AngleRadians::UNIT_NAME, "rad");
    assert_eq!(AngleDegrees::UNIT_NAME, "deg");
}
