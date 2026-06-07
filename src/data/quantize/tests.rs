#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::{ConversionError, DistanceWeight, LengthMeters, TimeSeconds, TimeWeight};

#[test]
fn try_from_seconds_rounds_like_the_infallible_path() {
    let checked = TimeWeight::try_from_seconds(1.234).unwrap();
    assert_eq!(checked, TimeWeight::from_seconds(1.234));
    assert_eq!(checked.as_centiseconds(), 123);
}

#[test]
fn try_from_seconds_rejects_non_finite() {
    assert_eq!(
        TimeWeight::try_from_seconds(f64::NAN),
        Err(ConversionError::NotFinite)
    );
    assert_eq!(
        TimeWeight::try_from_seconds(f64::INFINITY),
        Err(ConversionError::NotFinite)
    );
}

#[test]
fn try_from_seconds_rejects_negative_instead_of_panicking() {
    assert_eq!(
        TimeWeight::try_from_seconds(-0.5),
        Err(ConversionError::Negative)
    );
}

#[test]
fn try_from_seconds_reports_overflow_at_the_inf_sentinel() {
    let too_big = f64::from(u32::MAX) / 100.0;
    assert_eq!(
        TimeWeight::try_from_seconds(too_big),
        Err(ConversionError::Overflow)
    );
}

#[test]
fn try_from_meters_matches_distance_weight_quantization() {
    let checked = DistanceWeight::try_from_meters(42.0).unwrap();
    assert_eq!(checked, DistanceWeight::from_meters(42.0));
    assert_eq!(checked.as_decimeters(), 420);
}

#[test]
fn quantity_helpers_delegate_to_weight_constructors() {
    assert_eq!(
        LengthMeters::new(5.0).try_to_distance_weight().unwrap(),
        DistanceWeight::try_from_meters(5.0).unwrap()
    );
    assert_eq!(
        TimeSeconds::new(5.0).try_to_time_weight().unwrap(),
        TimeWeight::try_from_seconds(5.0).unwrap()
    );
}

#[test]
fn error_renders_a_message() {
    assert_eq!(ConversionError::Negative.to_string(), "value is negative");
}
