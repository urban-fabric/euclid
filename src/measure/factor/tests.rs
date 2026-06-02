#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;

#[test]
fn factor_constants() {
    assert_eq!(Factor::IDENTITY.as_f64(), 1.0);
    assert_eq!(Factor::ZERO.as_f64(), 0.0);
    assert_eq!(Factor::DIMENSION, Dimension::Dimensionless);
    assert_eq!(Factor::UNIT_NAME, "(unitless)");
}

#[test]
fn factor_default_is_identity() {
    assert_eq!(Factor::default(), Factor::IDENTITY);
}

#[test]
fn factor_composes() {
    let a = Factor::new(0.5);
    let b = Factor::new(2.0);
    assert_eq!((a * b).as_f64(), 1.0);
    assert_eq!((a * Factor::IDENTITY).as_f64(), 0.5);
    assert_eq!((Factor::IDENTITY * a).as_f64(), 0.5);
    assert_eq!((b / a).as_f64(), 4.0);
}

#[test]
fn factor_scalar() {
    let a = Factor::new(0.5);
    assert_eq!((a * 4.0).as_f64(), 2.0);
    assert_eq!((2.0 * a).as_f64(), 1.0);
    assert_eq!((a + a).as_f64(), 1.0);
}
