#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Tests for the uniform quantity APIs. These use
//! a spread of unit types to confirm the provided methods behave identically
//! regardless of which quantity they are called on.

use crate::{
    Factor, LengthMeters, LinearQuantity, Percentage, Quantity, QuantityError, ScalarQuantity,
    SpeedMs, TimeSeconds,
};

#[test]
fn from_raw_finite_accepts_finite_and_rejects_the_rest() {
    assert_eq!(
        LengthMeters::from_raw_finite(3.5).unwrap(),
        LengthMeters::new(3.5)
    );
    assert_eq!(
        LengthMeters::from_raw_finite(f64::NAN),
        Err(QuantityError::NotFinite)
    );
    assert_eq!(
        TimeSeconds::from_raw_finite(f64::INFINITY),
        Err(QuantityError::NotFinite)
    );
}

#[test]
fn abs_is_available_uniformly_even_without_an_inherent_method() {
    assert_eq!(SpeedMs::new(-4.0).abs(), SpeedMs::new(4.0));
    assert_eq!(Factor::new(-2.0).abs(), Factor::new(2.0));
}

#[test]
fn signum_follows_sign() {
    assert_eq!(LengthMeters::new(-7.0).signum(), -1.0);
    assert_eq!(LengthMeters::new(7.0).signum(), 1.0);
}

#[test]
fn min_max_and_clamp() {
    let a = LengthMeters::new(2.0);
    let b = LengthMeters::new(5.0);
    assert_eq!(a.min(b), a);
    assert_eq!(a.max(b), b);
    assert_eq!(LengthMeters::new(7.0).clamp(a, b), b);
    assert_eq!(LengthMeters::new(1.0).clamp(a, b), a);
    assert_eq!(LengthMeters::new(3.0).clamp(a, b), LengthMeters::new(3.0));
}

#[test]
fn rounding_family_stays_in_unit() {
    let x = TimeSeconds::new(2.6);
    assert_eq!(x.floor(), TimeSeconds::new(2.0));
    assert_eq!(x.ceil(), TimeSeconds::new(3.0));
    assert_eq!(x.round(), TimeSeconds::new(3.0));
    assert_eq!(TimeSeconds::new(-2.6).trunc(), TimeSeconds::new(-2.0));
}

#[test]
fn finiteness_predicates() {
    assert!(Percentage::new(50.0).is_finite());
    let inf = Percentage::from_raw(f64::INFINITY);
    let nan = Percentage::from_raw(f64::NAN);
    assert!(!inf.is_finite());
    assert!(inf.is_infinite());
    assert!(nan.is_nan());
}

#[test]
fn ratio_is_dimensionless() {
    assert_eq!(LengthMeters::new(10.0).ratio(LengthMeters::new(4.0)), 2.5);
}

#[test]
fn lerp_and_map() {
    let a = LengthMeters::new(0.0);
    let b = LengthMeters::new(10.0);
    assert_eq!(a.lerp(b, 0.25), LengthMeters::new(2.5));
    assert_eq!(b.map(|m| m * 2.0), LengthMeters::new(20.0));
}

#[test]
fn approx_eq_uses_a_typed_tolerance() {
    let a = LengthMeters::new(1.000_1);
    let b = LengthMeters::new(1.000_2);
    assert!(a.approx_eq(b, LengthMeters::new(0.001)));
    assert!(!a.approx_eq(b, LengthMeters::new(0.000_01)));
    assert!(a.approx_eq(b, LengthMeters::new(-0.001)));
}
