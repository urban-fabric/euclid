#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;

#[test]
fn constants() {
    assert_eq!(Percentage::ZERO.as_percent(), 0.0);
    assert_eq!(Percentage::FULL.as_percent(), 100.0);
    assert_eq!(Percentage::DIMENSION, Dimension::Dimensionless);
    assert_eq!(Percentage::UNIT_NAME, "%");
}

#[test]
fn fraction_round_trip() {
    let p = Percentage::from_fraction(Factor::new(0.42));
    assert!((p.as_percent() - 42.0).abs() < 1e-12);
    let f = p.to_fraction();
    assert!((f.as_f64() - 0.42).abs() < 1e-12);
}

#[test]
fn predicates() {
    assert!(Percentage::ZERO.is_zero());
    assert!(Percentage::ZERO.is_non_negative());
    assert!(!Percentage::ZERO.is_positive());
    assert!(Percentage::FULL.is_in_range());
    assert!(Percentage::new(50.0).is_in_range());
    assert!(!Percentage::new(101.0).is_in_range());
    assert!(!Percentage::new(-1.0).is_in_range());
}
