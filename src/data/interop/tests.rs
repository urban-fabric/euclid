use crate::{DistanceWeight, LengthMeters, Quantity, SpeedKph, TimeSeconds, TimeWeight};

#[test]
fn length_meters_roundtrips_via_distance_weight() {
    let m = LengthMeters::new(50.5);
    let dw = m.to_distance_weight();
    assert_eq!(dw.as_decimeters(), 505);
    let m2: LengthMeters = LengthMeters::from(dw);
    assert!((m2.as_meters() - 50.5).abs() < 1e-9);
}

#[test]
fn time_seconds_roundtrips_via_time_weight() {
    let s = TimeSeconds::new(60.0);
    let tw = s.to_time_weight();
    assert_eq!(tw.as_centiseconds(), 6000);
    let s2: TimeSeconds = TimeSeconds::from(tw);
    assert!((s2.as_seconds() - 60.0).abs() < 1e-9);
}

#[test]
fn travel_time_at_helper_uses_canonical_operand_order() {
    let l = LengthMeters::new(1500.0);
    let speed = SpeedKph::new(50.0);
    let helper = l.travel_time_at(speed);
    let expected = TimeSeconds::new(1500.0 * 3.6 / 50.0);
    assert_eq!(helper.as_seconds(), expected.as_seconds());
}

#[test]
fn distance_at_helper_uses_canonical_operand_order() {
    let t = TimeSeconds::new(120.0);
    let speed = SpeedKph::new(60.0);
    let helper = t.distance_at(speed);
    let expected = LengthMeters::new(60.0 * 120.0 / 3.6);
    assert_eq!(helper.as_meters(), expected.as_meters());
}

#[test]
fn travel_time_at_handles_zero_and_negative_speed() {
    let l = LengthMeters::new(100.0);
    assert_eq!(l.travel_time_at(SpeedKph::new(0.0)), TimeSeconds::INFINITY);
    assert_eq!(l.travel_time_at(SpeedKph::new(-5.0)), TimeSeconds::INFINITY);
    assert_eq!(
        l.travel_time_at(SpeedKph::from_raw(f64::NAN)),
        TimeSeconds::INFINITY,
    );
}

#[test]
fn quantization_clamps_overflow() {
    let huge = LengthMeters::new(1.0e15);
    let dw = huge.to_distance_weight();
    assert_eq!(dw, DistanceWeight::INF);

    let huge_t = TimeSeconds::new(1.0e15);
    let tw = huge_t.to_time_weight();
    assert_eq!(tw, TimeWeight::INF);
}
