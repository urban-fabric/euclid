use super::*;

#[test]
fn time_weight_quantizes_seconds_to_centiseconds() {
    assert_eq!(TimeWeight::from_seconds(0.0), TimeWeight::ZERO);
    assert_eq!(TimeWeight::from_seconds(1.234).as_centiseconds(), 123);
    assert_eq!(TimeWeight::from_seconds(1.236).as_centiseconds(), 124);

    let seconds = TimeWeight::from_centiseconds(12_345).as_seconds();
    assert!((seconds - 123.45).abs() < 1e-12);
}

#[test]
fn time_weight_infinity_sentinel_is_reserved() {
    assert_eq!(TimeWeight::from_seconds(f64::INFINITY), TimeWeight::INF);
    assert_eq!(TimeWeight::from_seconds(f64::NAN), TimeWeight::INF);
    assert_eq!(
        TimeWeight::from_seconds(u32::MAX as f64 / 100.0),
        TimeWeight::INF,
    );

    assert!(TimeWeight::INF.is_inf());
    assert!(!TimeWeight::INF.is_positive());
    assert!(!TimeWeight::INF.is_zero());
}

#[test]
fn time_weight_saturating_add_clamps_to_infinity() {
    assert_eq!(
        TimeWeight::from_centiseconds(u32::MAX - 5)
            .saturating_add(TimeWeight::from_centiseconds(10)),
        TimeWeight::INF,
    );
    assert_eq!(
        TimeWeight::INF.saturating_add(TimeWeight::ZERO),
        TimeWeight::INF
    );
}

#[test]
fn distance_weight_quantizes_meters_to_decimeters() {
    assert_eq!(DistanceWeight::from_meters(0.0), DistanceWeight::ZERO);
    assert_eq!(DistanceWeight::from_meters(1.24).as_decimeters(), 12);
    assert_eq!(DistanceWeight::from_meters(1.26).as_decimeters(), 13);

    let meters = DistanceWeight::from_decimeters(1_234).as_meters();
    assert!((meters - 123.4).abs() < 1e-12);
}

#[test]
fn distance_weight_infinity_sentinel_is_reserved() {
    assert_eq!(
        DistanceWeight::from_meters(f64::INFINITY),
        DistanceWeight::INF,
    );
    assert_eq!(DistanceWeight::from_meters(f64::NAN), DistanceWeight::INF);
    assert_eq!(
        DistanceWeight::from_meters(u32::MAX as f64 / 10.0),
        DistanceWeight::INF,
    );

    assert!(DistanceWeight::INF.is_inf());
    assert!(!DistanceWeight::INF.is_positive());
    assert!(!DistanceWeight::INF.is_zero());
}
