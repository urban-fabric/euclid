#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;

#[test]
fn time_seconds_constants() {
    assert_eq!(TimeSeconds::ZERO.0, 0.0);
    assert_eq!(TimeSeconds::DIMENSION, Dimension::Time);
    assert_eq!(TimeSeconds::UNIT_NAME, "s");
}

#[test]
fn time_seconds_arithmetic() {
    let a = TimeSeconds(60.0);
    let b = TimeSeconds(15.0);
    assert_eq!((a + b).0, 75.0);
    assert_eq!((a - b).0, 45.0);
    assert_eq!((a * 2.0).0, 120.0);
    let sum: TimeSeconds = [a, b].into_iter().sum();
    assert_eq!(sum.0, 75.0);
}

#[test]
fn time_round_trips() {
    let s = TimeSeconds(3_600.0);
    assert!((s.to_minutes().0 - 60.0).abs() < 1e-12);
    assert!((s.to_hours().0 - 1.0).abs() < 1e-12);
    assert!((s.to_hours().to_seconds().0 - 3_600.0).abs() < 1e-9);
}
