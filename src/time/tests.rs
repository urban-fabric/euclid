#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;

#[test]
fn time_seconds_constants() {
    assert_eq!(TimeSeconds::ZERO.as_seconds(), 0.0);
    assert_eq!(TimeSeconds::DIMENSION, Dimension::Time);
    assert_eq!(TimeSeconds::UNIT_NAME, "s");
}

#[test]
fn time_seconds_arithmetic() {
    let a = TimeSeconds::new(60.0);
    let b = TimeSeconds::new(15.0);
    assert_eq!((a + b).as_seconds(), 75.0);
    assert_eq!((a - b).as_seconds(), 45.0);
    assert_eq!((a * 2.0).as_seconds(), 120.0);
    let sum: TimeSeconds = [a, b].into_iter().sum();
    assert_eq!(sum.as_seconds(), 75.0);
}

#[test]
fn time_round_trips() {
    let s = TimeSeconds::new(3_600.0);
    assert!((s.to_minutes().as_minutes() - 60.0).abs() < 1e-12);
    assert!((s.to_hours().as_hours() - 1.0).abs() < 1e-12);
    assert!((s.to_hours().to_seconds().as_seconds() - 3_600.0).abs() < 1e-9);
}
