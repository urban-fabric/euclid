use crate::{LengthMeters, Quantity, SpeedKph, TimeSeconds};

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
