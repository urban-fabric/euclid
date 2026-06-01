#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;

#[test]
fn distance_weight_implements_quantity() {
    assert_eq!(DistanceWeight::DIMENSION, Dimension::Length);
    assert_eq!(DistanceWeight::UNIT_NAME, "dm");
    let d = DistanceWeight(500);
    assert_eq!(d.raw(), 500u32);
    assert_eq!(DistanceWeight::from_raw(500), d);
}

#[test]
fn time_weight_implements_quantity() {
    assert_eq!(TimeWeight::DIMENSION, Dimension::Time);
    assert_eq!(TimeWeight::UNIT_NAME, "cs");
    let t = TimeWeight(6000);
    assert_eq!(t.raw(), 6000u32);
}

#[test]
fn route_weights_serde_as_bare_u32_contract() {
    let time = TimeWeight::from_centiseconds(12_345);
    let time_json = serde_json::to_string(&time).unwrap();
    assert_eq!(time_json, "12345");
    assert_eq!(
        serde_json::from_str::<TimeWeight>(&time_json).unwrap(),
        time
    );

    let distance = DistanceWeight::from_decimeters(678);
    let distance_json = serde_json::to_string(&distance).unwrap();
    assert_eq!(distance_json, "678");
    assert_eq!(
        serde_json::from_str::<DistanceWeight>(&distance_json).unwrap(),
        distance,
    );
}

#[test]
fn route_weights_order_by_storage_repr() {
    let mut times = [
        TimeWeight::from_centiseconds(30),
        TimeWeight::ZERO,
        TimeWeight::from_centiseconds(10),
    ];
    times.sort();
    assert_eq!(
        times,
        [
            TimeWeight::ZERO,
            TimeWeight::from_centiseconds(10),
            TimeWeight::from_centiseconds(30),
        ],
    );

    let mut distances = [
        DistanceWeight::from_decimeters(9),
        DistanceWeight::from_decimeters(1),
        DistanceWeight::from_decimeters(5),
    ];
    distances.sort();
    assert_eq!(
        distances,
        [
            DistanceWeight::from_decimeters(1),
            DistanceWeight::from_decimeters(5),
            DistanceWeight::from_decimeters(9),
        ],
    );
}

#[test]
fn length_meters_roundtrips_via_distance_weight() {
    let m = LengthMeters(50.5);
    let dw = m.to_distance_weight();
    assert_eq!(dw.0, 505);
    let m2: LengthMeters = LengthMeters::from(dw);
    assert!((m2.0 - 50.5).abs() < 1e-9);
}

#[test]
fn time_seconds_roundtrips_via_time_weight() {
    let s = TimeSeconds(60.0);
    let tw = s.to_time_weight();
    assert_eq!(tw.0, 6000);
    let s2: TimeSeconds = TimeSeconds::from(tw);
    assert!((s2.0 - 60.0).abs() < 1e-9);
}

#[test]
fn travel_time_at_helper_uses_canonical_operand_order() {
    let l = LengthMeters(1500.0);
    let speed = SpeedKph(50.0);
    let helper = l.travel_time_at(speed);
    let expected = TimeSeconds(1500.0 * 3.6 / 50.0);
    assert_eq!(helper.0, expected.0);
}

#[test]
fn distance_at_helper_uses_canonical_operand_order() {
    let t = TimeSeconds(120.0);
    let speed = SpeedKph(60.0);
    let helper = t.distance_at(speed);
    let expected = LengthMeters(60.0 * 120.0 / 3.6);
    assert_eq!(helper.0, expected.0);
}

#[test]
fn travel_time_at_handles_zero_and_negative_speed() {
    let l = LengthMeters(100.0);
    assert_eq!(l.travel_time_at(SpeedKph(0.0)), TimeSeconds::INFINITY);
    assert_eq!(l.travel_time_at(SpeedKph(-5.0)), TimeSeconds::INFINITY);
    assert_eq!(l.travel_time_at(SpeedKph(f64::NAN)), TimeSeconds::INFINITY);
}

#[test]
fn quantization_clamps_overflow() {
    let huge = LengthMeters(1.0e15);
    let dw = huge.to_distance_weight();
    assert_eq!(dw, DistanceWeight::INF);

    let huge_t = TimeSeconds(1.0e15);
    let tw = huge_t.to_time_weight();
    assert_eq!(tw, TimeWeight::INF);
}
