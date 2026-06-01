#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;

#[test]
fn factor_constants() {
    assert_eq!(Factor::IDENTITY.0, 1.0);
    assert_eq!(Factor::ZERO.0, 0.0);
    assert_eq!(Factor::DIMENSION, Dimension::Dimensionless);
    assert_eq!(Factor::UNIT_NAME, "(unitless)");
}

#[test]
fn factor_default_is_identity() {
    assert_eq!(Factor::default(), Factor::IDENTITY);
}

#[test]
fn factor_composes() {
    let a = Factor(0.5);
    let b = Factor(2.0);
    assert_eq!((a * b).0, 1.0);
    assert_eq!((a * Factor::IDENTITY).0, 0.5);
    assert_eq!((Factor::IDENTITY * a).0, 0.5);
    assert_eq!((b / a).0, 4.0);
}

#[test]
fn factor_scalar() {
    let a = Factor(0.5);
    assert_eq!((a * 4.0).0, 2.0);
    assert_eq!((2.0 * a).0, 1.0);
    assert_eq!((a + a).0, 1.0);
}
